"""Markdown memory store: durable git-tracked knowledge + auto INDEX + recall.

``memory/*.md`` files are the durable, human-readable self-memory. ``INDEX.md`` is
auto-generated (titles + one-line summaries + tags), kept under a token budget and
injected into the agent system prompt. Session/task logs belong in SQLite, not here.

Recall combines lexical term-frequency/fuzzy scoring with an embedding-similarity
pass, so queries that share *meaning* but no exact words can still surface the
right file. The async paths (``arecall`` / ``auto_link_async``) use a pluggable
:class:`~nelke.core.embeddings.Embedder` — by default the offline hash embedder,
switching automatically to dense multilingual embeddings from an OpenAI-compatible
endpoint (LM Studio via ``[embeddings]``) when configured and reachable. Related
memory files are auto-linked with relative ``[title](file.md)`` cross-references.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from nelke.core.embeddings import (
    Embedder,
    _embed_similarity,  # noqa: F401  (re-exported for tests/back-compat)
    _tokenize,
    build_embedder,
)
from nelke.core.tools.base import ToolError, resolve_within

_INDEX_NAME = "INDEX.md"
_HEADING_RE = re.compile(r"^#\s+(.+)$")
_TAGS_RE = re.compile(r"^tags\s*:\s*(.+)$", re.IGNORECASE)

_EMBED_MIN_SIM = 0.35  # local threshold for "related" (sync recall / auto_link)


@dataclass
class MemoryHit:
    name: str
    score: int
    snippet: str


def _titles(name: str, content: str) -> str:
    for line in content.splitlines()[:20]:
        m = _HEADING_RE.match(line.strip())
        if m:
            return m.group(1).strip()
    return name.replace("_", " ").replace("-", " ").removesuffix(".md").strip()


def _summary(content: str) -> str:
    for line in content.splitlines()[1:]:
        s = line.strip()
        if not s:
            continue
        if s.startswith("#") or s.startswith("tags:"):
            continue
        return s
    return ""


def _score_terms(all_terms: list[str], query_terms: list[str]) -> tuple[int, int]:
    """Return (score, first_hit_pos) via term-frequency ranking.

    Each query term contributes ``min(count_in_doc, count_in_query)`` weighted by
    its ``1 + log(count)`` term frequency, so documents that repeat a query term
    rank higher, while exact matches are naturally weighted more than sparse
    ones. No inverse-document-frequency is applied (bodies are short).
    """
    if not all_terms:
        return 0, -1
    qfreq = {t: query_terms.count(t) for t in dict.fromkeys(query_terms)}
    total = 0.0
    first_hit = -1
    for term, qn in qfreq.items():
        cn = all_terms.count(term)
        if cn == 0:
            continue
        contrib = min(cn, qn) * (1.0 + math.log(1 + cn))
        total += contrib
        idx = -1
        for i, t in enumerate(all_terms):
            if t == term:
                idx = i
                break
        if first_hit < 0 or (idx >= 0 and idx < first_hit):
            first_hit = idx
    return int(round(total * 100)), first_hit


def _fuzzy_tokenize(text: str) -> list[str]:
    """Tokenize, then stem simple plurals (suffix ``s``/``es``) for fuzzy matches."""
    out = []
    for w in _tokenize(text):
        if w.endswith("ies") and len(w) > 4:
            out.append(w[:-3] + "y")
        elif w.endswith("es") and len(w) > 3:
            out.append(w[:-2])
        elif w.endswith("s") and len(w) > 3 and not w.endswith("ss"):
            out.append(w[:-1])
        out.append(w)
    return out


def _recall_one(content: str, query_terms: list[str]) -> tuple[int, int]:
    """Combined exact + stemmed-fuzzy scoring for a single document."""
    exact = _tokenize(content)
    exact_score, exact_pos = _score_terms(exact, query_terms)
    fuzzy = _fuzzy_tokenize(content)
    fuzzy_terms = list(dict.fromkeys(query_terms))
    fuzzy_score, fuzzy_pos = _score_terms(fuzzy, fuzzy_terms)
    # boost exact matches over fuzzy-only ones
    score = max(exact_score, fuzzy_score)
    if score == 0:
        return 0, -1
    pos = exact_pos if exact_pos >= 0 else fuzzy_pos
    return score, pos


class MemoryStore:
    def __init__(self, memory_dir: Path, embedder: Embedder | None = None) -> None:
        self.memory_dir = memory_dir.resolve()
        # Explicit embedder (tests/DI) or lazily-built from config: local by
        # default, dense (LM Studio-ish) when `[embeddings]` is configured.
        self._embedder = embedder
        self._resolved_embedder: Embedder | None = None

    def _get_embedder(self) -> Embedder:
        if self._embedder is not None:
            return self._embedder
        if self._resolved_embedder is None:
            self._resolved_embedder = build_embedder()
        return self._resolved_embedder

    def _resolve(self, name: str) -> Path:
        if not name.endswith(".md"):
            raise ToolError("memory files must be .md")
        return resolve_within(self.memory_dir, name)

    def files(self) -> list[Path]:
        """All markdown files under the memory dir except INDEX.md (relative paths)."""
        if not self.memory_dir.exists():
            return []
        out = [
            p.relative_to(self.memory_dir)
            for p in self.memory_dir.rglob("*.md")
            if p.name != _INDEX_NAME
        ]
        return sorted(out)

    def read(self, name: str) -> str:
        path = self._resolve(name)
        if not path.is_file():
            raise FileNotFoundError(str(path))
        return path.read_text(encoding="utf-8", errors="replace")

    def write(self, name: str, content: str, *, overwrite: bool = False) -> Path:
        """Write `content` to ``<memory_dir>/<name>``; by default appends to an existing file."""
        path = self._resolve(name)
        if path.exists() and not overwrite:
            existing = path.read_text(encoding="utf-8", errors="replace")
            chunk = existing.rstrip() + "\n\n" + content.strip() + "\n"
        else:
            chunk = content.strip() + "\n"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(chunk, encoding="utf-8")
        return path

    def build_index(self, *, max_tokens: int = 2000) -> str:
        """Rebuild a compact INDEX.md body and return its text."""
        budget_chars = max_tokens * 4
        lines: list[str] = ["# Nelke Memory Index"]
        groups: dict[str, list[Path]] = {}
        for rel in self.files():
            parts = rel.parts
            group = parts[0] if len(parts) > 1 else "."
            groups.setdefault(group, []).append(rel)

        used = 0
        for group in sorted(groups):
            if used >= budget_chars:
                break
            lines.append(f"\n## {group}")
            used += 4 + len(group)
            for rel in groups[group]:
                content = self.read(rel.as_posix())
                title = _titles(rel.name, content)
                summary = _summary(content)
                tags = self._extract_tags(content)
                body = f"- [{title}]({rel.as_posix()}) — {summary}"
                if tags:
                    body += f" #{tags}"
                body = body[: 2000]
                if used + len(body) > budget_chars:
                    if used >= budget_chars:
                        break
                    body = body[: budget_chars - used]
                lines.append(body)
                used += len(body)
        index_text = "\n".join(lines)
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        (self.memory_dir / _INDEX_NAME).write_text(index_text + "\n", encoding="utf-8")
        return index_text

    @staticmethod
    def _extract_tags(content: str) -> str:
        for line in content.splitlines()[:15]:
            m = _TAGS_RE.match(line.strip())
            if m:
                tags = [t.strip().lstrip("#").strip() for t in m.group(1).split(",")]
                return " ".join(t for t in tags if t)
        return ""

    def index_text(self) -> str:
        path = self.memory_dir / _INDEX_NAME
        return path.read_text(encoding="utf-8") if path.exists() else ""

    def recall(self, query: str, top_k: int = 8) -> list[MemoryHit]:
        """Synchronous relevance search using the *local* (offline) embedder.

        Scoring combines term-frequency ranking (with log dampening) over the raw
        body, a stemmed fuzzy pass (so plurals/queries still match), a boost when
        the query terms appear in the file's INDEX entry (title/tags/summary), and
        an embedding-similarity pass (word/stem hashing) that surfaces files
        sharing *meaning* but no exact words — e.g. "how do I recover a deleted
        branch" matching a file about ``git checkout``.

        Prefer :meth:`arecall` from async callers: it uses dense multilingual
        embeddings when an endpoint is configured, falling back to this path.
        """
        if not query.strip():
            return []
        query_terms = _tokenize(query)
        if not query_terms:
            return []
        index_text = self._load_index()
        hits: list[MemoryHit] = []
        for rel in self.files():
            content = self.read(rel.as_posix())
            body_score, body_pos = _recall_one(content, query_terms)
            # Semantic similarity — catches meaningful-but-lexically-distinct hits.
            semantic = _embed_similarity(query, content)
            if body_score == 0 and semantic < _EMBED_MIN_SIM:
                continue  # no lexical nor semantic signal
            # boost by matching the INDEX entry (title+tags+summary)
            entry_score = _recall_one(_index_entry(index_text, rel.as_posix()), query_terms)[0]
            score = body_score + entry_score + int(round(semantic * 100))
            pos = body_pos if body_pos >= 0 else 0
            snippet = _snippet(content, pos)
            hits.append(MemoryHit(name=rel.as_posix(), score=score, snippet=snippet))
        hits.sort(key=lambda h: (-h.score, h.name))
        return hits[:top_k]

    async def arecall(self, query: str, top_k: int = 8) -> list[MemoryHit]:
        """Recall using the configured embedder (dense + multilingual when
        available), with an automatic fallback to the local hashing path.

        Any embedder failure (endpoint down, no embedding model loaded, timeout)
        degrades cleanly to :meth:`recall` — memory recall is never blocked by a
        network dependency.
        """
        if not query.strip():
            return []
        query_terms = _tokenize(query)
        if not query_terms:
            return []
        embedder = self._get_embedder()
        try:
            if not await embedder.ensure_ready() or not embedder.available:
                return self.recall(query, top_k)
            rels = self.files()
            bodies = {r.as_posix(): self.read(r.as_posix()) for r in rels}
            vectors = await embedder.embed([query] + [bodies[r.as_posix()] for r in rels])
            if vectors is None or len(vectors) != len(rels) + 1:
                return self.recall(query, top_k)
            qvec = vectors[0]
            index_text = self._load_index()
            hits: list[MemoryHit] = []
            for rel, vec in zip(rels, vectors[1:], strict=True):
                content = bodies[rel.as_posix()]
                body_score, body_pos = _recall_one(content, query_terms)
                semantic = embedder.similarity(qvec, vec)
                if body_score == 0 and semantic < embedder.min_sim:
                    continue  # no lexical nor semantic signal
                entry_score = _recall_one(_index_entry(index_text, rel.as_posix()), query_terms)[0]
                score = body_score + entry_score + int(round(semantic * 100))
                pos = body_pos if body_pos >= 0 else 0
                hits.append(MemoryHit(name=rel.as_posix(), score=score, snippet=_snippet(content, pos)))
            hits.sort(key=lambda h: (-h.score, h.name))
            return hits[:top_k]
        except Exception:  # noqa: BLE001 - any embedder failure degrades to local
            return self.recall(query, top_k)

    def auto_link(self) -> list[str]:
        """Synchronous cross-linking using the local (offline) embedder.

        For every pair of memory files whose similarity clears the threshold, add
        ``Related: [title](other.md)`` to each file (unless already present).
        Returns the names of files updated. Idempotent and cheap enough to run
        after ``write``. Prefer :meth:`auto_link_async` from async callers.
        """
        files = self.files()
        bodies = {f.as_posix(): self.read(f.as_posix()) for f in files}
        titles = {f.as_posix(): _titles(f.name, bodies[f.as_posix()]) for f in files}
        return self._link_pairs(
            files, bodies, titles,
            lambda a, b: _embed_similarity(bodies[a], bodies[b]) >= _EMBED_MIN_SIM,
        )

    async def auto_link_async(self) -> list[str]:
        """Dense-embedding cross-linking (multilingual when available).

        Falls back to :meth:`auto_link` whenever the embedding backend is
        unavailable or an embed call fails.
        """
        embedder = self._get_embedder()
        try:
            if not await embedder.ensure_ready() or not embedder.available:
                return self.auto_link()
            files = self.files()
            if len(files) < 2:
                return []
            bodies = {f.as_posix(): self.read(f.as_posix()) for f in files}
            vectors = await embedder.embed([bodies[f.as_posix()] for f in files])
            if vectors is None or len(vectors) != len(files):
                return self.auto_link()
            titles = {f.as_posix(): _titles(f.name, bodies[f.as_posix()]) for f in files}
            vs = {f.as_posix(): v for f, v in zip(files, vectors, strict=True)}
            return self._link_pairs(
                files, bodies, titles,
                lambda a, b: embedder.similarity(vs[a], vs[b]) >= embedder.link_min_sim,
            )
        except Exception:  # noqa: BLE001 - any embedder failure degrades to local
            return self.auto_link()

    def _link_pairs(
        self,
        files: list[Path],
        bodies: dict[str, str],
        titles: dict[str, str],
        should_link: Callable[[str, str], bool],
    ) -> list[str]:
        """Shared auto-link body: write ``Related:`` links between pairs that
        satisfy ``should_link(a, b)``. Idempotent; returns updated names."""
        updated: set[str] = set()
        for i in range(len(files)):
            for j in range(i + 1, len(files)):
                a, b = files[i].as_posix(), files[j].as_posix()
                if not should_link(a, b):
                    continue
                if _link_line(a, b, titles[b]) not in bodies[a]:
                    bodies[a] = bodies[a].rstrip() + "\n\n" + _link_line(a, b, titles[b]) + "\n"
                    updated.add(a)
                if _link_line(b, a, titles[a]) not in bodies[b]:
                    bodies[b] = bodies[b].rstrip() + "\n\n" + _link_line(b, a, titles[a]) + "\n"
                    updated.add(b)
        for name in updated:
            self.write(name, bodies[name], overwrite=True)
        return sorted(updated)

    def _load_index(self) -> str:
        try:
            return self.index_text()
        except OSError:
            return ""


def _snippet(content: str, pos: int, width: int = 200) -> str:
    if pos < 0:
        return content[:width].replace("\n", " ")
    start = max(0, pos - width // 3)
    end = min(len(content), pos + width)
    piece = content[start:end].replace("\n", " ")
    prefix = "…" if start > 0 else ""
    return f"{prefix}{piece}"


def _index_entry(index_text: str, rel_name: str) -> str:
    """Return the INDEX.md line describing `rel_name` (title + tags + summary)."""
    for line in index_text.splitlines():
        if f"({rel_name})" in line:
            return line
    return ""


def _link_line(source: str, target: str, target_title: str) -> str:
    """A ``Related: [title](target)`` markdown line for ``source``."""
    return f"Related: [{target_title}]({target})"
