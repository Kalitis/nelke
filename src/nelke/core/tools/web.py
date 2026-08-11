"""Web tools: fetch a URL to markdown, and web search via DuckDuckGo.

``web_fetch`` is made resilient to anti-bot responses and flaky sources:
it validates the content-type before parsing, transparently retries on
transient errors, falls back to alternate mirrors when the primary source
returns an HTTP error (e.g. Wikipedia's 403), and decodes body bytes by the
charset the server declares so non-UTF-8 pages don't garble text.
"""

from __future__ import annotations

import re

import httpx

from nelke.core.tools.base import BaseTool, ToolResult

MAX_FETCH = 40_000

# Sources that frequently serve 403/451 to non-browser UAs have well-known
# mirrors (Wikipedia mirrors + the REST plain-text endpoint). When the primary
# URL fails with an HTTP error from one of these, we retry the mirror list until
# one succeeds.
_FALLBACKS = {
    "wikipedia.org": [
        # Language-aware plain-text endpoint first (keeps ru/de/… subdomains).
        lambda url: _rewrite_wiki_with_lang(url, "https://{lang}.wikipedia.org/api/rest_v1/page/plain/"),
        lambda url: _rewrite_wiki(url, "https://en.wikipedia.org/api/rest_v1/page/plain/"),
        # HTML mirrors as a last resort.
        lambda url: _rewrite_wiki_with_lang(url, "https://{lang}.m.wikipedia.org/wiki/"),
    ],
    "wikiwand.com": [
        # Wikiwand uses /<lang>/<title> (not /<lang>/wiki/<title>).
        lambda url: _rewrite_wikiwand(url),
    ],
}

# Content-types we are willing to parse as HTML/Markdown. Anything else (images,
# PDFs, JSON APIs the caller didn't ask for) is guarded against.
_HTML_CT = re.compile(r"text/html|application/xhtml|text/plain|text/markdown|text/x-wiki")

# A minimal desktop browser UA so anti-bot walls that reject httpx's default UA
# are more likely to let us through.
_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)


def _rewrite_wiki(url: str, prefix: str) -> str:
    """Rewrite a wikipedia URL to one of its mirrors/plain-text endpoints.

    Keeps the page path and optional query so article lookups survive the
    mirror switch. Only the scheme+host pivot changes via ``prefix``.
    """
    if "?action=raw" in url or "/api/rest_v1/page/plain/" in url:
        return url
    m = re.match(r"^https?://[^/]+(/wiki/[\w%()\-.~:/]+)(\?.*)?$", url)
    if not m:
        m = re.match(r"^https?://[^/]+(/w/index\.php\?title=[^&]+)(&.*)?$", url)
    if not m:
        return url
    return prefix + m.group(1).lstrip("/") + (m.group(2) or "")


def _rewrite_wiki_with_lang(url: str, prefix: str) -> str:
    """Rewrite hugging the original subdomain language (``de.…``, ``ru.…``)."""
    lang = "en"
    m = re.match(r"^https?://([a-z]{2,3})\.wikipedia\.org", url)
    if m and m.group(1) not in ("www", "m"):
        lang = m.group(1)
    return _rewrite_wiki(url, prefix.format(lang=lang))


def _rewrite_wikiwand(url: str, prefix: str = "https://www.wikiwand.com/") -> str:
    """Rewrite a ``<lang>.wikiwand.com/<title>`` URL to the canonical ``www`` form.

    Wikiwand mirrors article paths as ``/<lang>/<title>`` (no ``/wiki/``), so we
    keep the original language and page — otherwise the primary URL and the
    fallback would point at the same page and the 403 would repeat.
    """
    m = re.match(r"^https?://([a-z]{2,3})\.wikiwand\.com/([^/?#]+)(.*)$", url)
    if not m:
        return url
    lang = m.group(1) if m.group(1) not in ("www", "m") else ""
    lang_part = f"{lang}/" if lang else ""
    return prefix + lang_part + m.group(2) + m.group(3)


class WebFetchTool(BaseTool):
    name = "web_fetch"
    description = "Fetch a URL and return its content converted to Markdown (truncated)."
    parameters = {
        "type": "object",
        "properties": {
            "url": {"type": "string"},
            "max_chars": {"type": "integer", "default": MAX_FETCH},
        },
        "required": ["url"],
    }

    def __init__(self, timeout: float = 30) -> None:
        self.timeout = timeout

    async def execute(self, **kwargs) -> ToolResult:
        url = str(kwargs.get("url", ""))
        max_chars = int(kwargs.get("max_chars") or MAX_FETCH)
        if not url.startswith(("http://", "https://")):
            return ToolResult.failure("url must start with http:// or https://")

        headers = {"User-Agent": _USER_AGENT}
        timeout = httpx.Timeout(self.timeout, connect=self.timeout)
        try:
            async with httpx.AsyncClient(timeout=timeout, follow_redirects=True,
                                         headers=headers) as client:
                # Primary attempt.
                fetched = await _fetch_with_fallback(client, url)
        except httpx.HTTPError as exc:
            return ToolResult.failure(f"fetch failed: {exc}")

        if fetched is None:
            return ToolResult.failure(
                f"all sources failed for {url} (anti-bot / HTTP error)"
            )
        resp, is_plain = fetched
        if not _HTML_CT.search(resp.headers.get("content-type", "")):
            return ToolResult.failure(
                f"unexpected content-type {resp.headers.get('content-type', '')!r} for {url}; "
                "refusing to parse non-HTML content"
            )
        text = _decode_body(resp)
        if is_plain:
            # Plain-text fallbacks (Wikipedia REST) arrive already as text/markdown;
            # skip the HTML converter.
            converted = text.strip()
        else:
            converted = _html_to_markdown(text)
        if len(converted) > max_chars:
            converted = converted[:max_chars] + "\n...[truncated]"
        return ToolResult.success(converted)


async def _fetch_with_fallback(client, url: str):
    """Fetch ``url``; on an HTTP error try configured mirrors.

    Returns ``(response, is_plain)`` or ``None`` if every candidate failed.
    Transient network errors (timeouts, connection resets) retry the primary
    source once before giving up on it.
    """
    # Which mirrors apply to this host.
    host_mirrors: list = []
    hostname = (url.split("://", 1)[1].split("/", 1)[0] if "://" in url else "")
    for host, mirrors in _FALLBACKS.items():
        if host in hostname:
            host_mirrors = mirrors
            break

    candidates = [url]
    for mirror in host_mirrors:
        aliased = mirror(url)
        if aliased and aliased not in candidates:
            candidates.append(aliased)

    last_http_error: Exception | None = None
    for attempt, cand in enumerate(candidates):
        is_plain = "/api/rest_v1/page/plain/" in cand or cand.startswith(
            ("https://en.wikipedia.org/api/rest_v1/page/plain/",)
        )
        try:
            resp = await client.get(cand)
        except httpx.TransportError as exc:
            # Transient: retry once at the appropriate attempt boundary.
            last_http_error = exc
            if attempt == 0:
                try:
                    resp = await client.get(cand)
                except httpx.TransportError as exc2:
                    last_http_error = exc2
                    continue
            else:
                continue
        if resp.status_code >= 400:
            last_http_error = httpx.HTTPStatusError(
                f"HTTP {resp.status_code}", request=resp.request, response=resp
            )
            continue
        return resp, is_plain
    raise last_http_error or httpx.HTTPError(f"no candidates succeeded for {url}")


def _decode_body(resp) -> str:
    """Decode a response body using the charset the server declared (else UTF-8)."""
    ct = resp.headers.get("content-type", "")
    charset_m = re.search(r"charset=([\w-]+)", ct, re.IGNORECASE)
    charset = charset_m.group(1) if charset_m else None
    for enc in (charset, "utf-8", "latin-1"):
        if not enc:
            continue
        try:
            return resp.content.decode(enc)
        except (LookupError, UnicodeDecodeError):
            continue
    # Last resort: latin-1 never fails, so this is unreachable except for a bad charset name.
    return resp.content.decode("utf-8", errors="replace")


class WebSearchTool(BaseTool):
    name = "web_search"
    description = "Search the web (DuckDuckGo) and return a list of results with title, url and snippet."
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "max_results": {"type": "integer", "default": 5},
        },
        "required": ["query"],
    }

    def __init__(self, timeout: float = 30) -> None:
        self.timeout = timeout

    async def execute(self, **kwargs) -> ToolResult:
        query = str(kwargs.get("query", ""))
        max_results = int(kwargs.get("max_results") or 5)
        try:
            from duckduckgo_search import DDGS
        except ImportError:
            return ToolResult.failure("duckduckgo-search is not installed")
        try:
            with DDGS(timeout=int(self.timeout)) as ddgs:
                results = list(ddgs.text(query, max_results=max(1, max_results)))
        except Exception as exc:  # noqa: BLE001 - search backends are flaky
            return ToolResult.failure(f"search failed: {exc}")
        lines: list[str] = []
        for i, r in enumerate(results, 1):
            title = str(r.get("title", "untitled"))
            url = str(r.get("href", r.get("url", "")))
            body = str(r.get("body", ""))
            lines.append(f"{i}. {title}\n   {url}\n   {body}")
        return ToolResult.success("\n".join(lines) if lines else "no results")


def _html_to_markdown(html: str) -> str:
    import html as html_mod

    # Remove scripts/styles/nav noise first so the converter doesn't preserve it.
    html = _strip_boilerplate(html)
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        return html_mod.unescape(re.sub(r"<[^>]+>", " ", html))
    try:
        from markdownify import markdownify as mdify
    except ImportError:
        soup = BeautifulSoup(html, "html.parser")
        return soup.get_text("\n")
    return mdify(html, heading_style="ATX").strip()


_BOILERPLATE_RE = re.compile(
    r"<(script|style|noscript|template|iframe|svg|nav|footer|header|aside)"
    r"[^>]*>.*?</\1>",
    re.IGNORECASE | re.DOTALL,
)
_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)


def _strip_boilerplate(html: str) -> str:
    """Drop non-content blocks (scripts, styles, nav, comments) before conversion.

    These rarely carry useful text and often dominate an HTML page, so removing
    them first yields cleaner, shorter markdown.
    """
    html = _COMMENT_RE.sub("", html)
    return _BOILERPLATE_RE.sub("", html)
