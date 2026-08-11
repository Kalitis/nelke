"""LLM client internals: ReAct fallback parser, response normalization, streaming."""

from __future__ import annotations

from types import SimpleNamespace as NS

from nelke.core.llm import (
    LLMClient,
    _fallback_tool_calls,
    _usage_to_dict,
    parse_action_input,
    parse_react_actions,
    usage_cache_pct,
)


def _client() -> LLMClient:
    cli = object.__new__(LLMClient)
    cli.allow_fallback_parse = True
    return cli


# --------------------------------------------------------------------------- #
# ReAct parser
# --------------------------------------------------------------------------- #
def test_parse_react_single_action():
    text = (
        "Thought: I need the file.\n"
        "Action: read\n"
        "Action Input: {\"path\": \"notes.md\"}\n"
        "Observation: ...\n"
    )
    actions = parse_react_actions(text)
    assert actions == [("read", '{"path": "notes.md"}')]


def test_parse_react_multiple_actions():
    text = (
        "Action: web_search\n"
        "Action Input: {\"query\": \"pytest fixtures\"}\n"
        "Thought: searching\n"
        "Action: recall\n"
        "Action Input: \"fixtures\"\n"
    )
    actions = parse_react_actions(text)
    assert [a[0] for a in actions] == ["web_search", "recall"]


def test_parse_action_input_json_object():
    schemas = [{"function": {"name": "read", "parameters": {"properties": {"path": {}}, "required": ["path"]}}}]
    assert parse_action_input('{"path": "a"}', schemas, "read") == {"path": "a"}


def test_parse_action_input_code_fenced_json():
    schemas: list[dict] = []
    raw = '```json\n{"query": "x"}\n```'
    assert parse_action_input(raw, schemas, "any") == {"query": "x"}


def test_parse_action_input_primitive_uses_first_param():
    schemas = [{"function": {"name": "recall", "parameters": {"properties": {"query": {}}, "required": ["query"]}}}]
    assert parse_action_input("fixtures", schemas, "recall") == {"query": "fixtures"}


def test_fallback_tool_calls_produces_toolcalls():
    schemas = [{"function": {"name": "read", "parameters": {"properties": {"path": {}}, "required": ["path"]}}}]
    calls = _fallback_tool_calls(
        'Action: read\nAction Input: {"path": "x.txt"}', schemas
    )
    assert len(calls) == 1
    assert calls[0].name == "read"
    assert calls[0].arguments == {"path": "x.txt"}
    assert calls[0].id.startswith("call_react")


# --------------------------------------------------------------------------- #
# Response normalization (no network)
# --------------------------------------------------------------------------- #
def _tool_call_delta(name=None, arguments='{"path": "x"}', tool_id="t1", index=0):
    return NS(
        id=tool_id,
        index=index,
        type="function",
        function=NS(name=name, arguments=arguments),
    )


def _message(content=None, tool_calls=None):
    return NS(content=content, tool_calls=tool_calls)


def test_parse_native_tool_calls():
    cli = _client()
    resp = NS(
        choices=[NS(message=_message(None, [_tool_call_delta(name="read")]), finish_reason="tool_calls")],
        usage=NS(prompt_tokens=1, completion_tokens=2, total_tokens=3),
    )
    out = cli._parse_non_stream(resp, tools=None)
    assert out.tool_calls[0].name == "read"
    assert out.tool_calls[0].arguments == {"path": "x"}
    assert out.usage is not None
    assert out.usage["total_tokens"] == 3


def test_react_fallback_from_plain_content():
    cli = _client()
    schemas = [{"function": {"name": "read", "parameters": {"properties": {"path": {}}, "required": ["path"]}}}]
    resp = NS(
        choices=[NS(message=_message('Action: read\nAction Input: {"path": "x.txt"}'), finish_reason="stop")],
        usage=None,
    )
    out = cli._parse_non_stream(resp, schemas)
    assert out.content == ""
    assert out.tool_calls[0].name == "read"


def test_no_tool_activity_stays_answer():
    cli = _client()
    resp = NS(
        choices=[NS(message=_message("plain answer"), finish_reason="stop")], usage=None
    )
    out = cli._parse_non_stream(resp, tools=None)
    assert out.content == "plain answer"
    assert out.tool_calls == []


# --------------------------------------------------------------------------- #
# Streaming accumulation
# --------------------------------------------------------------------------- #
class _FakeStream:
    def __init__(self, chunks):
        self._chunks = chunks

    def __aiter__(self):
        return self._iter()

    async def _iter(self):
        for c in self._chunks:
            yield c


def _chunk(content=None, tool_delta=None, finish=None):
    return NS(
        choices=[] if finish is None and content is None and tool_delta is None else [NS(
            delta=NS(content=content, tool_calls=tool_delta) if not (content is None and tool_delta is None) else None,
            finish_reason=finish,
        )],
        usage=None,
    )


async def test_stream_content_tokens():
    cli = _client()
    chunks = [
        _chunk(content="Hel"),
        _chunk(content="lo"),
        _chunk(content=None, finish="stop"),
    ]
    received: list[str] = []
    content, calls, finish, usage = await cli._consume_stream(_FakeStream(chunks), received.append)
    assert content == "Hello"
    assert received == ["Hel", "lo"]
    assert calls == []
    assert finish == "stop"


async def test_stream_tool_call_accumulation():
    cli = _client()
    stream = _FakeStream([
        _chunk(tool_delta=[_tool_call_delta(name="re", arguments='{"pat')]),
        _chunk(tool_delta=[_tool_call_delta(index=0, arguments='h": "x"}')]),
        _chunk(content=None, finish="tool_calls"),
    ])
    content, calls, finish, usage = await cli._consume_stream(stream, None)
    assert content == ""
    assert calls[0].name == "re"
    assert calls[0].arguments == {"path": "x"}
    assert finish == "tool_calls"


# --------------------------------------------------------------------------- #
# Usage normalization: prompt-cache metrics
# --------------------------------------------------------------------------- #
def test_usage_dict_captures_openai_cached_tokens():
    usage = NS(
        prompt_tokens=1500,
        completion_tokens=50,
        total_tokens=1550,
        prompt_tokens_details=NS(cached_tokens=1000),
    )
    d = _usage_to_dict(usage)
    assert d["prompt_tokens"] == 1500
    assert d["total_tokens"] == 1550
    assert d["cache_read_tokens"] == 1000
    assert d["cache_write_tokens"] == 0


def test_usage_dict_captures_deepseek_cache_tokens():
    usage = NS(
        prompt_tokens=1500,
        completion_tokens=60,
        total_tokens=1560,
        prompt_cache_hit_tokens=1210,
        prompt_cache_miss_tokens=290,
    )
    d = _usage_to_dict(usage)
    assert d["cache_read_tokens"] == 1210
    assert d["cache_write_tokens"] == 290


def test_usage_dict_no_cache_fields_default_to_zero():
    usage = NS(prompt_tokens=10, completion_tokens=5, total_tokens=15)
    d = _usage_to_dict(usage)
    assert d == {
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "total_tokens": 15,
        "cache_read_tokens": 0,
        "cache_write_tokens": 0,
    }


def test_usage_dict_openai_details_win_over_deepseek_fields():
    usage = NS(
        prompt_tokens=100,
        completion_tokens=1,
        total_tokens=101,
        prompt_tokens_details=NS(cached_tokens=99),
        prompt_cache_hit_tokens=50,
    )
    d = _usage_to_dict(usage)
    assert d["cache_read_tokens"] == 99


def test_usage_dict_empty_on_missing_attrs():
    assert _usage_to_dict(NS()) == {}


# --------------------------------------------------------------------------- #
# Cache-read percentage helper
# --------------------------------------------------------------------------- #
def test_usage_cache_pct_rounds_to_integer():
    assert usage_cache_pct({"prompt_tokens": 3616, "cache_read_tokens": 3328}) == 92
    assert usage_cache_pct({"prompt_tokens": 100, "cache_read_tokens": 50}) == 50


def test_usage_cache_pct_zero_without_prompt():
    assert usage_cache_pct({}) == 0
    assert usage_cache_pct({"prompt_tokens": 0, "cache_read_tokens": 10}) == 0
    assert usage_cache_pct({"prompt_tokens": 10, "cache_read_tokens": 0}) == 0


def test_usage_cache_pct_full_and_partial():
    assert usage_cache_pct({"prompt_tokens": 100, "cache_read_tokens": 100}) == 100
    assert usage_cache_pct({"prompt_tokens": 3, "cache_read_tokens": 1}) == 33
