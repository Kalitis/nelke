"""Markdown memory store: durable git-tracked knowledge + auto INDEX + recall.

``memory/*.md`` files are the durable, human-readable self-memory. ``INDEX.md`` is
auto-generated (titles + one-line summaries + tags), kept under a token budget and
injected into the agent system prompt. Session/task logs belong in SQLite, not here.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from nelke.core.tools.base import ToolError, resolve_within

_INDEX_NAME = "INDEX.md"
_HEADING_RE = re.compile(r"^#\s+(.+)$")
_TAGS_RE = re.compile(r"^tags\s*:\s*(.+)$", re.IGNORECASE)
_WORD_SPLIT_RE = re.compile(r"[^\w]+")


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
        """Simple keyword relevance search over memory files."""
        if not query.strip():
            return []
        query_words = {w for w in _WORD_SPLIT_RE.split(query.lower()) if w and w not in {"the", "a", "an"}}
        if not query_words:
            return []
        scored: list[MemoryHit] = []
        for rel in self.files():
            content = self.read(rel.as_posix())
            lower = content.lower()
            score = 0
            first_hit = -1
            for w in query_words:
                idx = lower.find(w)
                if idx >= 0:
                    score += 1
                    if first_hit < 0 or idx < first_hit:
                        first_hit = idx
            if score == 0:
                continue
            snippet = _snippet(content, first_hit)
            scored.append(MemoryHit(name=rel.as_posix(), score=score, snippet=snippet))
        scored.sort(key=lambda h: (-h.score, h.name))
        return scored[:top_k]


def _snippet(content: str, pos: int, width: int = 200) -> str:
    if pos < 0:
        return content[:width].replace("\n", " ")
    start = max(0, pos - width // 3)
    end = min(len(content), pos + width)
    piece = content[start:end].replace("\n", " ")
    prefix = "…" if start > 0 else ""
    return f"{prefix}{piece}"
