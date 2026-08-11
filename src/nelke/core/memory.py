"""Markdown memory store: durable git-tracked knowledge + auto INDEX + recall.

``memory/*.md`` files are the durable, human-readable self-memory. ``INDEX.md`` is
auto-generated (titles + one-line summaries + tags), kept under a token budget and
injected into the agent system prompt. Session/task logs belong in SQLite, not here.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path

from nelke.core.tools.base import ToolError, resolve_within

_INDEX_NAME = "INDEX.md"
_HEADING_RE = re.compile(r"^#\s+(.+)$")
_TAGS_RE = re.compile(r"^tags\s*:\s*(.+)$", re.IGNORECASE)
_WORD_SPLIT_RE = re.compile(r"[^\w]+")
# Stopwords dropped from queries to avoid noise matches.
_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with",
    "is", "are", "been", "was", "were", "it", "its", "that", "this", "these",
    "those", "from", "as", "by", "at", "be", "have", "has", "do", "does",
}


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


def _tokenize(text: str) -> list[str]:
    return [w for w in _WORD_SPLIT_RE.split(text.lower()) if w and w not in _STOPWORDS]


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
    def __init__(self, memory_dir: Path) -> None:
        self.memory_dir = memory_dir.resolve()

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
        """Relevance search over memory files.

        Scoring combines term-frequency ranking (with log dampening) over the raw
        body, a stemmed fuzzy pass (so plurals/queries still match), and a boost
        when the query terms appear in the file's INDEX entry (title/tags/summary).
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
            if body_score == 0:
                continue
            # boost by matching the INDEX entry (title+tags+summary)
            entry_score = _recall_one(_index_entry(index_text, rel.as_posix()), query_terms)[0]
            score = body_score + entry_score
            pos = body_pos if body_pos >= 0 else 0
            snippet = _snippet(content, pos)
            hits.append(MemoryHit(name=rel.as_posix(), score=score, snippet=snippet))
        hits.sort(key=lambda h: (-h.score, h.name))
        return hits[:top_k]

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
