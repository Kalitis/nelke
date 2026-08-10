"""Multi-provider OpenAI-compatible LLM client.

Wraps ``openai.AsyncOpenAI`` with per-profile ``base_url``/``api_key``/``model``.
Supports streaming, native function calling, retries with exponential backoff, and
a ReAct-style fallback parser so local models without native function calling work.
"""

from __future__ import annotations

import ast
import asyncio
import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable

ToolCallback = Callable[[str], Any] | None


class LLMError(RuntimeError):
    """Raised when the LLM could not be reached after retries."""


@dataclass
class ToolCall:
    """A single tool call requested by the model."""

    id: str
    name: str
    arguments: dict[str, Any]
    raw_arguments: str = ""


@dataclass
class LLMResponse:
    """Normalized model reply (independent of transport)."""

    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str = "stop"
    usage: dict[str, Any] | None = None


def parse_react_actions(text: str) -> list[tuple[str, str]]:
    """Split a ReAct-style block into (tool name, raw input) pairs.

    Accepted shape (repeated per tool call)::

        Thought: ...
        Action: tool_name
        Action Input: {"key": "value"}

    The input may span lines (e.g. a pretty-printed JSON object).
    """
    results: list[tuple[str, str]] = []
    current_name: str | None = None
    input_lines: list[str] = []
    name_re = re.compile(r"^Action\s*:\s*(.*)$")
    input_re = re.compile(r"^Action\s*Input\s*:\s*(.*)$")
    thought_re = re.compile(r"^Thought\s*:")
    observation_re = re.compile(r"^Observation\s*:")

    def flush() -> None:
        nonlocal current_name, input_lines
        if current_name is not None:
            results.append((current_name.strip(), "\n".join(input_lines).strip()))
        current_name = None
        input_lines = []

    for line in text.splitlines():
        if not line.strip():
            if current_name is not None:
                input_lines.append(line)
            continue
        m = name_re.match(line)
        if m:
            flush()
            current_name = m.group(1)
            continue
        m = input_re.match(line)
        if m:
            input_lines.append(m.group(1))
            continue
        if thought_re.match(line) or observation_re.match(line):
            flush()
            continue
        if current_name is not None:
            input_lines.append(line)
    flush()
    return results


def _extract_json_object(raw: str) -> dict[str, Any] | None:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-zA-Z]*", "", raw).strip()
        raw = re.sub(r"```$", "", raw).strip()
    for candidate in (raw,):
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
    brace = re.search(r"\{.*\}", raw, re.DOTALL)
    if brace:
        try:
            obj = json.loads(brace.group(0))
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
    return None


def _first_param_name(schemas: list[dict[str, Any]], tool_name: str) -> str:
    for schema in schemas:
        func = schema.get("function", {})
        if func.get("name") != tool_name:
            continue
        params = func.get("parameters") or {}
        required = params.get("required") or []
        if required:
            return str(required[0])
        props = params.get("properties") or {}
        if props:
            return str(next(iter(props)))
    return "value"


def parse_action_input(
    raw: str, schemas: list[dict[str, Any]], tool_name: str
) -> dict[str, Any]:
    """Coerce a ReAct raw action input into keyword arguments for the tool."""
    obj = _extract_json_object(raw)
    if obj is not None:
        return obj
    literal: Any = raw.strip()
    if not literal:
        return {}
    try:
        literal = ast.literal_eval(literal)
    except (ValueError, SyntaxError):
        literal = literal
    if isinstance(literal, dict):
        return literal
    return {_first_param_name(schemas, tool_name): literal}


def _fallback_tool_calls(
    content: str, schemas: list[dict[str, Any]]
) -> list[ToolCall]:
    actions = parse_react_actions(content)
    calls: list[ToolCall] = []
    for i, (name, raw) in enumerate(actions):
        if not name:
            continue
        calls.append(
            ToolCall(
                id=f"call_react_{i}",
                name=name,
                arguments=parse_action_input(raw, schemas, name),
                raw_arguments=raw,
            )
        )
    return calls


class LLMClient:
    """OpenAI-compatible chat client with retries and optional ReAct fallback."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        *,
        max_retries: int = 3,
        timeout: float = 120,
        extra: dict[str, Any] | None = None,
        allow_fallback_parse: bool = True,
    ) -> None:
        import openai  # lazy import keeps boot path light

        self._client = openai.AsyncOpenAI(
            base_url=base_url, api_key=api_key, timeout=timeout, **(extra or {})
        )
        self.model = model
        self.max_retries = max_retries
        self.timeout = timeout
        self.allow_fallback_parse = allow_fallback_parse

    async def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stream: bool = False,
        on_token: ToolCallback = None,
    ) -> LLMResponse:
        kwargs: dict[str, Any] = {"model": model or self.model, "messages": messages}
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        if temperature is not None:
            kwargs["temperature"] = temperature
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        kwargs["stream"] = stream

        resp = await self._request_with_retry(kwargs)
        if not stream:
            return self._parse_non_stream(resp, tools)

        content, calls, finish, usage = await self._consume_stream(resp, on_token)
        return self._finalize(content, calls, finish, usage, tools)

    async def _request_with_retry(self, kwargs: dict[str, Any]) -> Any:
        last: BaseException | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                return await self._client.chat.completions.create(**kwargs)
            except Exception as exc:  # noqa: BLE001 - surfaced as LLMError
                last = exc
                if attempt >= self.max_retries:
                    break
                await asyncio.sleep(min(2**attempt, 8))
        raise LLMError(f"LLM request failed after {self.max_retries} attempts: {last}")

    def _parse_non_stream(
        self, resp: Any, tools: list[dict[str, Any]] | None
    ) -> LLMResponse:
        choice = resp.choices[0]
        message = choice.message
        content = message.content or ""
        calls: list[ToolCall] = []
        for tc in message.tool_calls or []:
            raw = tc.function.arguments or ""
            try:
                args: dict[str, Any] = json.loads(raw) if raw else {}
            except json.JSONDecodeError:
                args = {}
            calls.append(
                ToolCall(id=tc.id, name=tc.function.name, arguments=args, raw_arguments=raw)
            )
        finish = getattr(choice, "finish_reason", "stop") or "stop"
        usage = _usage_to_dict(getattr(resp, "usage", None))
        return self._finalize(content, calls, finish, usage, tools)

    async def _consume_stream(
        self, resp: Any, on_token: ToolCallback
    ) -> tuple[str, list[ToolCall], str, dict[str, Any] | None]:
        content = ""
        finish = "stop"
        usage: dict[str, Any] | None = None
        acc: dict[int, dict[str, Any]] = {}
        async for chunk in resp:
            if chunk.usage is not None:
                usage = _usage_to_dict(chunk.usage)
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            if choice.finish_reason:
                finish = choice.finish_reason
            delta = choice.delta
            if delta is None:
                continue
            if delta.content:
                content += delta.content
                if on_token is not None:
                    on_token(delta.content)
            for tcd in delta.tool_calls or []:
                idx = tcd.index or 0
                entry = acc.setdefault(idx, {"id": "", "name": "", "args": ""})
                if tcd.id and tcd.type == "function":
                    entry["id"] = tcd.id
                if tcd.function:
                    if tcd.function.name:
                        entry["name"] = tcd.function.name
                    if tcd.function.arguments:
                        entry["args"] += tcd.function.arguments
        calls: list[ToolCall] = []
        for i in sorted(acc):
            e = acc[i]
            raw = e["args"]
            try:
                args: dict[str, Any] = json.loads(raw) if raw else {}
            except json.JSONDecodeError:
                args = {}
            calls.append(ToolCall(id=e["id"] or f"call_{i}", name=e["name"], arguments=args, raw_arguments=raw))
        return content, calls, finish, usage

    def _finalize(
        self,
        content: str,
        calls: list[ToolCall],
        finish: str,
        usage: dict[str, Any] | None,
        schemas: list[dict[str, Any]] | None,
    ) -> LLMResponse:
        if not calls and self.allow_fallback_parse and content:
            calls = _fallback_tool_calls(content, schemas or [])
        return LLMResponse(
            content=content if not calls else "", tool_calls=calls, finish_reason=finish, usage=usage
        )


def _usage_to_dict(usage: Any) -> dict[str, Any]:
    try:
        return {
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens,
        }
    except AttributeError:
        return {}


class StubLLM:
    """No-op LLM used by ``boot_check()`` — never touches the network."""

    def __init__(self, answer: str = "OK") -> None:
        self._answer = answer

    async def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stream: bool = False,
        on_token: ToolCallback = None,
    ) -> LLMResponse:
        if stream and on_token is not None:
            on_token(self._answer)
        return LLMResponse(content=self._answer, tool_calls=[], finish_reason="stop")


def build_llm(profile_name: str | None = None) -> LLMClient:
    """Construct an :class:`LLMClient` from the active config/profile."""
    from nelke.config import get_profile

    profile = get_profile(profile_name)
    key = profile.resolved_api_key()
    return LLMClient(
        base_url=profile.base_url,
        api_key=key or "not-needed",
        model=profile.model,
        extra=profile.extra or {},
    )
