"""Workspace-scoped filesystem tools (read, write, edit, glob, grep)."""

from __future__ import annotations

import re
from pathlib import Path

from nelke.core.tools.base import BaseTool, ToolResult, resolve_within

MAX_OUTPUT = 60_000
DEFAULT_ENCODING = "utf-8"


class ReadFileTool(BaseTool):
    name = "read"
    description = "Read a text file inside the workspace. Returns its contents (truncated if huge)."
    parameters = {
        "type": "object",
        "properties": {"path": {"type": "string", "description": "Path relative to workspace root"}},
        "required": ["path"],
    }

    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace

    async def execute(self, **kwargs) -> ToolResult:
        path = resolve_within(self.workspace, kwargs.get("path", ""))
        if not path.is_file():
            return ToolResult.failure(f"file not found: {path}")
        try:
            content = path.read_text(encoding=DEFAULT_ENCODING, errors="replace")
        except OSError as exc:
            return ToolResult.failure(f"read failed: {exc}")
        if len(content) > MAX_OUTPUT:
            content = content[:MAX_OUTPUT] + "\n...[truncated]"
        return ToolResult.success(content)


class WriteFileTool(BaseTool):
    name = "write"
    description = "Create or overwrite a text file inside the workspace (parent dirs are created)."
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path relative to workspace root"},
            "content": {"type": "string", "description": "Full file content"},
        },
        "required": ["path", "content"],
    }

    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace

    async def execute(self, **kwargs) -> ToolResult:
        path = resolve_within(self.workspace, kwargs.get("path", ""))
        content = str(kwargs.get("content", ""))
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding=DEFAULT_ENCODING)
        except OSError as exc:
            return ToolResult.failure(f"write failed: {exc}")
        return ToolResult.success(f"wrote {len(content)} bytes to {path.name}")


class EditFileTool(BaseTool):
    name = "edit"
    description = "Replace an exact string inside a workspace file (single occurrence by default)."
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "old_string": {"type": "string"},
            "new_string": {"type": "string"},
            "replace_all": {"type": "boolean", "description": "Replace every occurrence", "default": False},
        },
        "required": ["path", "old_string", "new_string"],
    }

    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace

    async def execute(self, **kwargs) -> ToolResult:
        path = resolve_within(self.workspace, kwargs.get("path", ""))
        if not path.is_file():
            return ToolResult.failure(f"file not found: {path}")
        old = str(kwargs.get("old_string", ""))
        new = str(kwargs.get("new_string", ""))
        replace_all = bool(kwargs.get("replace_all", False))
        try:
            content = path.read_text(encoding=DEFAULT_ENCODING)
        except OSError as exc:
            return ToolResult.failure(f"read failed: {exc}")
        count = content.count(old)
        if count == 0:
            return ToolResult.failure("old_string not found")
        if count > 1 and not replace_all:
            return ToolResult.failure(
                f"old_string found {count} times; pass replace_all=true to replace them all"
            )
        updated = content.replace(old, new) if replace_all else content.replace(old, new, 1)
        try:
            path.write_text(updated, encoding=DEFAULT_ENCODING)
        except OSError as exc:
            return ToolResult.failure(f"write failed: {exc}")
        return ToolResult.success(f"edited {path.name} ({count} replacement(s))")


class GlobTool(BaseTool):
    name = "glob"
    description = "List files matching a glob pattern inside the workspace."
    parameters = {
        "type": "object",
        "properties": {"pattern": {"type": "string", "description": "Glob pattern, e.g. '**/*.py'"}},
        "required": ["pattern"],
    }

    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace

    async def execute(self, **kwargs) -> ToolResult:
        pattern = str(kwargs.get("pattern", ""))
        root = self.workspace.resolve()
        matches = [p.relative_to(root).as_posix() for p in root.glob(pattern)]
        matches.sort()
        return ToolResult.success("\n".join(matches) if matches else "no matches")


class GrepTool(BaseTool):
    name = "grep"
    description = "Search file contents inside the workspace with a regular expression."
    parameters = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Regular expression"},
            "include": {"type": "string", "description": "Glob filter, e.g. '*.py'", "default": None},
        },
        "required": ["pattern"],
    }

    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace

    async def execute(self, **kwargs) -> ToolResult:
        pattern = str(kwargs.get("pattern", ""))
        include = kwargs.get("include") or "**/*"
        try:
            regex = re.compile(pattern)
        except re.error as exc:
            return ToolResult.failure(f"invalid regex: {exc}")
        root = self.workspace.resolve()
        hits: list[str] = []
        for p in root.glob(include):
            if not p.is_file():
                continue
            try:
                for lineno, line in enumerate(p.read_text(encoding=DEFAULT_ENCODING, errors="replace").splitlines(), 1):
                    if regex.search(line):
                        rel = p.relative_to(root).as_posix()
                        hits.append(f"{rel}:{lineno}: {line[:200]}")
                        if len(hits) >= 200:
                            hits.append("...[truncated]")
                            return ToolResult.success("\n".join(hits))
            except OSError:
                continue
        return ToolResult.success("\n".join(hits) if hits else "no matches")
