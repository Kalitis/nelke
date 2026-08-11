"""The agent: system prompt (+ memory index), optional planning, tool-calling loop.

Opens a task by appending a user message, then loops LLM -> tool_calls? -> execute
until the model replies with plain text or the iteration cap is hit. Tokens stream
to a frontend via an opt-in callback.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable

from nelke.core.llm import LLMResponse, ToolCall
from nelke.core.session_analyzer import DegradationReport, analyze_degradation
from nelke.core.tools.base import BaseTool, ToolResult
from nelke.core.tools.registry import ToolRegistry

TokenHandler = Callable[[str], Any] | None
ToolHandler = Callable[[str, dict[str, Any]], Any] | None
ToolResultHandler = Callable[[str, dict[str, Any], str], Any] | None
UsageHandler = Callable[[dict[str, Any]], Any] | None
DegradedHandler = Callable[[DegradationReport], Any] | None

_EMPTY_USAGE = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "calls": 0}


@dataclass
class AgentResult:
    answer: str
    iterations: int
    tool_calls: int
    tool_errors: int = 0
    messages: list[dict[str, Any]] = field(default_factory=list)
    stopped: str = "answer"  # "answer" | "max_iterations"
    usage: dict[str, int] = field(default_factory=lambda: dict(_EMPTY_USAGE))


class Agent:
    def __init__(
        self,
        name: str,
        system_prompt: str,
        tools: list[BaseTool],
        llm: Any,
        *,
        iteration_cap: int = 20,
        stream: bool = False,
        on_token: TokenHandler = None,
        on_tool: ToolHandler = None,
        on_tool_result: ToolResultHandler = None,
        on_usage: UsageHandler = None,
        on_degraded: DegradedHandler = None,
        degrade_error_threshold: int = 3,
        temperature: float | None = None,
        memory_index: str | None = None,
        plan_first: bool = False,
    ) -> None:
        self.name = name
        self.system_prompt = system_prompt
        self.llm = llm
        self.iteration_cap = iteration_cap
        self.stream = stream
        self.on_token = on_token
        self.on_tool = on_tool
        self.on_tool_result = on_tool_result
        self.on_usage = on_usage
        self.on_degraded = on_degraded
        self.degrade_error_threshold = degrade_error_threshold
        self.temperature = temperature
        self.memory_index = memory_index
        self.plan_first = plan_first
        self.registry = ToolRegistry.from_list(tools)
        self._messages: list[dict[str, Any]] = []
        self._tool_errors = 0

    def system_content(self) -> str:
        content = self.system_prompt
        if self.memory_index:
            content += "\n\n# Persistent memory\n" + self.memory_index
        return content

    def reset(self) -> None:
        self._messages = []

    async def plan(self, task: str) -> str:
        """Optional planning step: a single non-tool call that sketches a plan."""
        prompt = (
            "Before executing the task, produce a concise step-by-step plan (3-8 steps). "
            "Answer with the plan only, no tools."
        )
        msgs = [
            {"role": "system", "content": self.system_content()},
            {"role": "user", "content": prompt + "\n\nTask: " + task},
        ]
        resp: LLMResponse = await self.llm.chat(msgs, stream=False)
        return (resp.content or "").strip()

    async def run(
        self, task: str, *, reset: bool = True
    ) -> AgentResult:
        """Run a task with the tool loop. With ``reset=False`` continues the conversation."""
        if reset or not self._messages:
            self._messages = [{"role": "system", "content": self.system_content()}]
        self._messages.append({"role": "user", "content": task})
        msgs = self._messages

        self._usage = dict(_EMPTY_USAGE)
        self._tool_errors = 0
        tool_calls_total = 0
        for i in range(self.iteration_cap):
            resp = await self.llm.chat(
                msgs,
                tools=self.registry.schemas() if self.registry.names() else None,
                stream=self.stream,
                on_token=self.on_token,
                temperature=self.temperature,
            )
            self._merge_usage(resp.usage)
            self._notify_usage(resp.usage)
            if not resp.tool_calls:
                msgs.append({"role": "assistant", "content": resp.content or ""})
                answer = (resp.content or "").strip()
                result = AgentResult(
                    answer=answer, iterations=i + 1, tool_calls=tool_calls_total,
                    tool_errors=self._tool_errors, messages=msgs, usage=self._usage,
                )
                self._maybe_degrade(result, task)
                return result
            msgs.append(self._assistant_tool_message(resp))
            for tc in resp.tool_calls:
                tool_calls_total += 1
                tool_result = await self._execute_tool(tc)
                msgs.append({"role": "tool", "tool_call_id": tc.id, "content": tool_result.render()})
        result = AgentResult(
            answer="", iterations=self.iteration_cap, tool_calls=tool_calls_total,
            tool_errors=self._tool_errors, messages=msgs, stopped="max_iterations",
            usage=self._usage,
        )
        self._maybe_degrade(result, task)
        return result

    def _maybe_degrade(self, result: AgentResult, task: str) -> None:
        if self.on_degraded is None:
            return
        report = analyze_degradation(result, task, error_threshold=self.degrade_error_threshold)
        if report.degraded:
            self.on_degraded(report)

    def _merge_usage(self, usage: dict[str, Any] | None) -> None:
        self._usage["calls"] += 1
        if not usage:
            return
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            self._usage[key] += int(usage.get(key, 0) or 0)

    def _notify_usage(self, usage: dict[str, Any] | None) -> None:
        """Report a single LLM call's usage as soon as it is available."""
        if self.on_usage is None:
            return
        if not usage or not usage.get("total_tokens"):
            return
        self.on_usage(dict(usage))

    async def _execute_tool(self, tc: ToolCall) -> ToolResult:
        if self.on_tool is not None:
            self.on_tool(tc.name, dict(tc.arguments))
        try:
            tool = self.registry.get(tc.name)
        except Exception as exc:  # noqa: BLE001
            result = ToolResult.failure(f"unknown tool {tc.name!r}: {exc}")
            self._notify_tool_result(tc, result)
            self._tool_errors += 1
            return result
        try:
            result = await tool.execute(**tc.arguments)
        except Exception as exc:  # noqa: BLE001 - never let a tool crash the loop
            result = ToolResult.failure(f"tool {tc.name} raised: {exc}")
        self._notify_tool_result(tc, result)
        if not result.ok:
            self._tool_errors += 1
        return result

    def _notify_tool_result(self, tc: ToolCall, result: ToolResult) -> None:
        if self.on_tool_result is not None:
            self.on_tool_result(tc.name, dict(tc.arguments), result.render())

    @staticmethod
    def _assistant_tool_message(resp: LLMResponse) -> dict[str, Any]:
        return {
            "role": "assistant",
            "content": resp.content or None,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
                }
                for tc in resp.tool_calls
            ],
        }


DEFAULT_SYSTEM_PROMPT = (
    "You are Nelke, a general-purpose agent (research, math, programming, web). "
    "Think step by step before acting. Use the available tools to gather information "
    "and do work; write files only inside your workspace. Prefer concise, correct "
    "answers. You have persistent memory available via recall/memory_write."
)


def make_agent(
    workspace: Any,
    llm: Any,
    *,
    name: str = "nelke",
    system_prompt: str | None = None,
    memory: Any = None,
    memory_index: str | None = None,
    task_factory: Callable[[list[str] | None], "Agent"] | None = None,
    on_token: TokenHandler = None,
    on_tool: ToolHandler = None,
    on_tool_result: ToolResultHandler = None,
    on_usage: UsageHandler = None,
    on_degraded: DegradedHandler = None,
    degrade_error_threshold: int = 3,
    stream: bool = False,
    iteration_cap: int = 20,
    code_timeout: int = 120,
    web_timeout: float = 30,
    include_web: bool = True,
    include_shell: bool = True,
    db: Any = None,
    temperature: float | None = None,
) -> Agent:
    """Build a normal-mode agent (workspace-scoped tools).

    ``db`` is optional: when given, the normal-mode agent also gets a
    :class:`CyclesTool` so it can answer questions about in-flight or finished
    self-improvement cycles from an ordinary chat, without interrupting them.
    """
    from pathlib import Path

    from nelke.core.tools.cycles import CyclesTool
    from nelke.core.tools.fs import (
        EditFileTool,
        GlobTool,
        GrepTool,
        ReadFileTool,
        WriteFileTool,
    )
    from nelke.core.tools.memory import MemoryWriteTool, RecallTool
    from nelke.core.tools.shell import BashTool, PythonRunTool
    from nelke.core.tools.subagent import TaskTool
    from nelke.core.tools.web import WebFetchTool, WebSearchTool

    ws = Path(workspace)
    tools: list[BaseTool] = [
        ReadFileTool(ws),
        WriteFileTool(ws),
        EditFileTool(ws),
        GlobTool(ws),
        GrepTool(ws),
    ]
    if include_shell:
        tools += [BashTool(ws, code_timeout), PythonRunTool(ws, code_timeout)]
    if include_web:
        tools += [WebFetchTool(web_timeout), WebSearchTool(web_timeout)]
    if memory is not None:
        tools += [RecallTool(memory), MemoryWriteTool(memory)]
    if task_factory is not None:
        tools += [TaskTool(task_factory)]
    if db is not None:
        tools += [CyclesTool(db)]
    return Agent(
        name=name,
        system_prompt=system_prompt or DEFAULT_SYSTEM_PROMPT,
        tools=tools,
        llm=llm,
        iteration_cap=iteration_cap,
        stream=stream,
        on_token=on_token,
        on_tool=on_tool,
        on_tool_result=on_tool_result,
        on_usage=on_usage,
        on_degraded=on_degraded,
        degrade_error_threshold=degrade_error_threshold,
        memory_index=memory_index,
        temperature=temperature,
    )
