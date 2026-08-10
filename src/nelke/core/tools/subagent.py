"""Subagent delegation tool: spawn a fresh agent with a clean context."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from nelke.core.tools.base import BaseTool, ToolResult

if TYPE_CHECKING:
    from nelke.core.agent import Agent


class TaskTool(BaseTool):
    name = "task"
    description = (
        "Delegate a bounded subtask to a fresh subagent with a clean context. "
        "Returns the subagent's final answer, truncated."
    )
    parameters = {
        "type": "object",
        "properties": {
            "task_description": {"type": "string", "description": "Self-contained description of the subtask"},
            "tool_names": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional subset of tool names for the subagent",
                "default": None,
            },
        },
        "required": ["task_description"],
    }

    def __init__(
        self,
        agent_factory: Callable[[list[str] | None], "Agent"],
        max_chars: int = 20_000,
    ) -> None:
        self._factory = agent_factory
        self.max_chars = max_chars

    async def execute(self, **kwargs: Any) -> ToolResult:
        task = str(kwargs.get("task_description", "")).strip()
        if not task:
            return ToolResult.failure("task_description is required")
        tool_names = kwargs.get("tool_names")
        agent = self._factory(tool_names)
        try:
            result = await agent.run(task)
        except Exception as exc:  # noqa: BLE001 - report back to caller agent
            return ToolResult.failure(f"subagent failed: {exc}")
        answer = (result.answer or "").strip()
        if len(answer) > self.max_chars:
            answer = answer[: self.max_chars] + "\n...[truncated]"
        return ToolResult.success(answer or "(subagent produced no answer)")
