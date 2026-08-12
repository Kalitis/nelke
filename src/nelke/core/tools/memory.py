"""Tools for reading and writing the Nelke markdown memory store."""

from __future__ import annotations

from nelke.core.memory import MemoryStore
from nelke.core.tools.base import BaseTool, ToolError, ToolResult


class RecallTool(BaseTool):
    name = "recall"
    description = "Keyword-search the persistent markdown memory store; returns ranked snippets."
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "top_k": {"type": "integer", "default": 8},
        },
        "required": ["query"],
    }

    def __init__(self, store: MemoryStore, default_top_k: int = 8) -> None:
        self.store = store
        self.default_top_k = default_top_k

    async def execute(self, **kwargs) -> ToolResult:
        query = str(kwargs.get("query", ""))
        top_k = int(kwargs.get("top_k") or self.default_top_k)
        hits = await self.store.arecall(query, top_k)
        if not hits:
            return ToolResult.success("no memory matches")
        lines = []
        for h in hits:
            lines.append(f"[{h.name}] (score {h.score})\n  {h.snippet}")
        return ToolResult.success("\n".join(lines))


class MemoryListTool(BaseTool):
    name = "memory_list"
    description = (
        "List every markdown file in the persistent memory store (short names "
        "like 'skills.md', 'facts/llms.md')."
    )
    parameters = {"type": "object", "properties": {}}

    def __init__(self, store: MemoryStore) -> None:
        self.store = store

    async def execute(self, **kwargs) -> ToolResult:
        files = self.store.files()
        if not files:
            return ToolResult.success("memory store is empty")
        return ToolResult.success("\n".join(f.as_posix() for f in files))


class MemoryShowTool(BaseTool):
    name = "memory_show"
    description = (
        "Return the full text of a file in the persistent memory store by short "
        "name (e.g. 'skills.md' or 'facts/llms.md'). Short names come from `recall` "
        "results and `memory_list`. Use after `recall` to read the whole source."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File under memory/, e.g. 'skills.md' or 'facts/llms.md'"},
        },
        "required": ["path"],
    }

    def __init__(self, store: MemoryStore) -> None:
        self.store = store

    async def execute(self, **kwargs) -> ToolResult:
        path = str(kwargs.get("path", ""))
        try:
            content = self.store.read(path)
        except (ToolError, FileNotFoundError, OSError) as exc:
            return ToolResult.failure(f"memory_show failed: {exc}")
        if len(content) > 60_000:
            content = content[:60_000] + "\n...[truncated]"
        return ToolResult.success(content)


class MemoryWriteTool(BaseTool):
    name = "memory_write"
    description = (
        "Append (default) or overwrite a markdown file in the memory store under "
        "memory/, then rebuild INDEX.md. Use to record skills, lessons and facts."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File under memory/, e.g. 'skills.md' or 'facts/llms.md'"},
            "content": {"type": "string", "description": "Markdown to append/write"},
            "overwrite": {"type": "boolean", "default": False},
        },
        "required": ["path", "content"],
    }

    def __init__(self, store: MemoryStore) -> None:
        self.store = store

    async def execute(self, **kwargs) -> ToolResult:
        path = str(kwargs.get("path", ""))
        content = str(kwargs.get("content", ""))
        overwrite = bool(kwargs.get("overwrite", False))
        try:
            self.store.write(path, content, overwrite=overwrite)
            linked = await self.store.auto_link_async()
            index_len = len(self.store.build_index())
        except (ToolError, FileNotFoundError, OSError) as exc:
            return ToolResult.failure(f"memory_write failed: {exc}")
        link_note = f"; cross-linked {len(linked)} related file(s)" if linked else ""
        return ToolResult.success(
            f"wrote memory/{path}; INDEX.md rebuilt ({index_len} chars){link_note}"
        )
