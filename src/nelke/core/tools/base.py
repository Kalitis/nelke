"""Base abstractions for Nelke tools."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class ToolError(RuntimeError):
    pass


@dataclass
class ToolResult:
    """Result of executing a tool."""

    ok: bool
    output: str = ""
    error: str | None = None

    @classmethod
    def success(cls, output: str) -> "ToolResult":
        return cls(ok=True, output=output)

    @classmethod
    def failure(cls, error: str) -> "ToolResult":
        return cls(ok=False, error=error)

    def render(self) -> str:
        """Human-readable form fed back to the model."""
        if self.ok:
            return self.output
        return f"ERROR: {self.error}"


class BaseTool(ABC):
    """A tool exposed to the model via OpenAI function-calling schema."""

    name: str = ""
    description: str = ""
    parameters: dict[str, Any] = {}

    def to_openai_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    @abstractmethod
    async def execute(self, **kwargs: Any) -> ToolResult: ...


def resolve_within(root: Path, candidate: str | Path) -> Path:
    """Resolve a candidate path, refusing to escape ``root``.

    Paths outside ``root`` raise :class:`ToolError` — the primary workspace
    scoping guard for both normal and self-edit tools.
    """
    root = root.resolve()
    target = (root / candidate).resolve() if not Path(candidate).is_absolute() else Path(candidate).resolve()
    if target != root and root not in target.parents:
        raise ToolError(f"path {candidate!r} is outside allowed root {root}")
    return target
