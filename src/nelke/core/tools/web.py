"""Web tools: fetch a URL to markdown, and web search via DuckDuckGo."""

from __future__ import annotations

import re

from nelke.core.tools.base import BaseTool, ToolResult

MAX_FETCH = 40_000


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
        import httpx

        url = str(kwargs.get("url", ""))
        max_chars = int(kwargs.get("max_chars") or MAX_FETCH)
        if not url.startswith(("http://", "https://")):
            return ToolResult.failure("url must start with http:// or https://")
        try:
            async with httpx.AsyncClient(timeout=self.timeout, follow_redirects=True) as client:
                resp = await client.get(url)
                resp.raise_for_status()
        except httpx.HTTPError as exc:
            return ToolResult.failure(f"fetch failed: {exc}")
        text = _html_to_markdown(resp.text)
        if len(text) > max_chars:
            text = text[:max_chars] + "\n...[truncated]"
        return ToolResult.success(text)


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
