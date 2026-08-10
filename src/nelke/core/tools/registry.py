"""Tool registry: mapping of tool name -> BaseTool and schema helpers."""

from __future__ import annotations

from typing import Any

from nelke.core.tools.base import BaseTool, ToolError


class ToolRegistry:
    def __init__(self, tools: list[BaseTool] | None = None) -> None:
        self._tools: dict[str, BaseTool] = {}
        if tools:
            self.register_all(*tools)

    def register(self, tool: BaseTool) -> "ToolRegistry":
        if not tool.name:
            raise ToolError("tool must define a non-empty name")
        self._tools[tool.name] = tool
        return self

    def register_all(self, *tools: BaseTool) -> "ToolRegistry":
        for tool in tools:
            self.register(tool)
        return self

    def get(self, name: str) -> BaseTool:
        try:
            return self._tools[name]
        except KeyError:
            raise ToolError(f"unknown tool: {name}") from None

    def has(self, name: str) -> bool:
        return name in self._tools

    def names(self) -> list[str]:
        return sorted(self._tools)

    def schemas(self) -> list[dict[str, Any]]:
        return [tool.to_openai_schema() for tool in self._tools.values()]

    def all(self) -> list[BaseTool]:
        return list(self._tools.values())

    @classmethod
    def from_list(cls, tools: list[BaseTool]) -> "ToolRegistry":
        return cls(tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools
