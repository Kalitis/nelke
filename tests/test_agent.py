"""Agent loop tests: tool calling, iteration cap, graceful unknown tools, subagents."""

from __future__ import annotations

from conftest import final_response, llm_with_script, tool_response

from nelke.core.agent import Agent
from nelke.core.llm import LLMResponse, ToolCall
from nelke.core.tools.fs import ReadFileTool


def _agent(tmp_path, llm, tools=None, **kw):
    return Agent(
        name="test",
        system_prompt="You are a test agent.",
        tools=tools or [ReadFileTool(tmp_path)],
        llm=llm,
        iteration_cap=kw.pop("iteration_cap", 10),
        **kw,
    )


async def test_plan_first_prepends_plan(tmp_path):
    """With plan_first, the agent runs a plan call before the tool loop and the
    plan text lands in the conversation context."""
    plan_llm = llm_with_script([
        final_response("1. read x.txt\n2. summarize"),
    ])
    agent = _agent(tmp_path, plan_llm, plan_first=True)
    result = await agent.run("summarize x.txt")
    assert result.answer == "1. read x.txt\n2. summarize"
    # The plan (from the first call) is present as an assistant message before
    # the final answer.
    assert result.iterations == 1
    assistant_contents = [m["content"] for m in result.messages if m["role"] == "assistant"]
    assert any(c and c.startswith("Plan:") for c in assistant_contents)


async def test_plan_first_counts_plan_llm_call(tmp_path):
    """The plan-first planning call's usage is merged into the run totals."""

    class PlanLLM:
        def __init__(self) -> None:
            self.n = 0

        async def chat(self, messages, **kw):
            self.n += 1
            if self.n == 1:
                return final_response("plan step")
            return final_response("done")

    agent = _agent(tmp_path, PlanLLM(), plan_first=True)
    result = await agent.run("go")
    assert result.answer == "done"
    assert result.usage["calls"] == 2


async def test_plan_first_does_not_replan_later_turns(tmp_path):
    """With reset=False, plan-first runs only once per conversation, not per turn."""
    calls: list[int] = []

    class PlanLLM:
        def __init__(self) -> None:
            self.plan_made = False

        async def chat(self, messages, **kw):
            calls.append(len(messages))
            if not self.plan_made:
                self.plan_made = True
                return final_response("1. do the thing")
            return final_response("final")

    agent = _agent(tmp_path, PlanLLM(), plan_first=True)
    r1 = await agent.run("first", reset=True)
    r2 = await agent.run("second", reset=False)
    # Turn 1: the first call produces the plan, the second call the final answer
    # (r1.answer == "final"). Turn 2 must NOT re-plan — its only call is the
    # answer itself. 3 calls total = plan + answer + answer(no re-plan).
    assert r1.answer == "final"
    assert r2.answer == "final"
    assert len(calls) == 3


async def test_plan_first_survives_plan_error(tmp_path):
    """A planning exception must degrade to a normal tool loop, not crash."""

    class BrokenPlanLLM:
        def __init__(self) -> None:
            self.n = 0

        async def chat(self, messages, **kw):
            self.n += 1
            if self.n == 1:
                raise RuntimeError("plan backend down")
            return final_response("recovered without plan")

    agent = _agent(tmp_path, BrokenPlanLLM(), plan_first=True)
    result = await agent.run("go")
    assert result.answer == "recovered without plan"
    # the plan failure is counted as a tool error but the run completes
    assert result.tool_errors == 1


async def test_tool_loop_read_then_answer(tmp_path):
    (tmp_path / "x.txt").write_text("hello world", encoding="utf-8")
    llm = llm_with_script([
        tool_response("read", {"path": "x.txt"}),
        final_response("I read it."),
    ])
    agent = _agent(tmp_path, llm)
    result = await agent.run("read x.txt")
    assert result.answer == "I read it."
    assert result.tool_calls == 1
    assert result.iterations == 2
    # the tool result reached the conversation
    tool_msgs = [m for m in result.messages if m["role"] == "tool"]
    assert len(tool_msgs) == 1
    assert "hello world" in tool_msgs[0]["content"]


async def test_missing_file_returns_error_but_loop_continues(tmp_path):
    llm = llm_with_script([
        tool_response("read", {"path": "missing.txt"}),
        final_response("still responded"),
    ])
    agent = _agent(tmp_path, llm)
    result = await agent.run("do it")
    assert result.answer == "still responded"
    tool_msgs = [m for m in result.messages if m["role"] == "tool"]
    assert "ERROR" in tool_msgs[0]["content"]


async def test_unknown_tool_fails_gracefully(tmp_path):
    llm = llm_with_script([
        tool_response("no_such_tool", {}),
        final_response("recovered"),
    ])
    agent = _agent(tmp_path, llm)
    result = await agent.run("try")
    assert result.answer == "recovered"
    assert result.tool_calls == 1


async def test_iteration_cap_stops_loop(tmp_path):
    loop = [tool_response("read", {"path": "x.txt"})] * 10
    llm = llm_with_script(loop)
    agent = _agent(tmp_path, llm, iteration_cap=3)
    result = await agent.run("spin")
    assert result.stopped == "max_iterations"
    assert result.iterations == 3
    assert result.answer == ""


async def test_tool_errors_are_tracked(tmp_path):
    llm = llm_with_script([
        tool_response("read", {"path": "missing.txt"}),
        tool_response("read", {"path": "missing.txt"}),
        tool_response("read", {"path": "missing.txt"}),
        final_response("recovered"),
    ])
    agent = _agent(tmp_path, llm)
    result = await agent.run("read missing files")
    assert result.tool_errors == 3
    assert result.tool_calls == 3


async def test_on_degraded_hook_fires_on_iteration_cap(tmp_path):
    reports = []
    loop = [tool_response("read", {"path": "x.txt"})] * 10
    llm = llm_with_script(loop)
    agent = _agent(
        tmp_path, llm, iteration_cap=4,
        on_degraded=reports.append,
    )
    await agent.run("fetch the widget")
    assert len(reports) == 1
    assert reports[0].degraded
    assert reports[0].suggested_objective  # non-empty proposed objective


async def test_on_degraded_not_called_on_success(tmp_path):
    reports = []
    (tmp_path / "x.txt").write_text("hello world", encoding="utf-8")
    llm = llm_with_script([tool_response("read", {"path": "x.txt"}), final_response("read it")])
    agent = _agent(tmp_path, llm, on_degraded=reports.append)
    result = await agent.run("read x.txt")
    assert result.tool_errors == 0
    assert reports == []


async def test_stream_tokens_and_tool_notify(tmp_path):
    tokens: list[str] = []
    tools_seen: list[tuple[str, dict]] = []

    class StreamingLLM:
        async def chat(self, messages, *, tools=None, model=None, temperature=None,
                       max_tokens=None, stream=False, on_token=None):
            if tools and messages[-1]["role"] == "user" and "read" in str(tools):
                return LLMResponse(content="", tool_calls=[ToolCall("call_1", "read", {"path": "x.txt"})])
            resp = final_response("done streaming")
            if stream and on_token is not None:
                for token in ("done ", "streaming"):
                    on_token(token)
            return resp

    (tmp_path / "x.txt").write_text("hi", encoding="utf-8")
    agent = _agent(tmp_path, StreamingLLM(),
                   on_token=tokens.append, on_tool=lambda n, a: tools_seen.append((n, a)), stream=True)
    result = await agent.run("go")
    assert result.answer == "done streaming"
    assert tokens == ["done ", "streaming"]
    assert tools_seen == [("read", {"path": "x.txt"})]


async def test_subagent_task_tool(tmp_path):
    from nelke.core.tools.subagent import TaskTool

    sub_llm = llm_with_script([final_response("subresult")])

    def factory(tool_names=None):
        return Agent(
            name="sub",
            system_prompt="subagent prompt",
            tools=[],
            llm=sub_llm,
        )

    parent = Agent(
        name="parent",
        system_prompt="parent prompt",
        tools=[TaskTool(factory)],
        llm=llm_with_script([
            tool_response("task", {"task_description": "count items"}),
            final_response("parent done"),
        ]),
        iteration_cap=5,
    )
    result = await parent.run("delegate")
    assert "subresult" in [m["content"] for m in result.messages if m["role"] == "tool"][0]
    assert result.answer == "parent done"


async def test_usage_is_accumulated(tmp_path):
    class UsageLLM:
        def __init__(self) -> None:
            self.n = 0

        async def chat(self, messages, *, tools=None, model=None, temperature=None,
                       max_tokens=None, stream=False, on_token=None):
            self.n += 1
            if self.n == 1:
                return LLMResponse(content="", tool_calls=[ToolCall("call_1", "read", {"path": "x.txt"})],
                                   usage={"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7})
            return LLMResponse(content="done", usage={"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4})

    (tmp_path / "x.txt").write_text("hi", encoding="utf-8")
    agent = _agent(tmp_path, UsageLLM())
    result = await agent.run("go")
    assert result.usage == {
        "prompt_tokens": 8, "completion_tokens": 3, "total_tokens": 11,
        "calls": 2, "cache_read_tokens": 0, "cache_read_pct": 0,
    }


async def test_usage_cache_read_accumulated_as_percent(tmp_path):
    """cache_read_tokens are summed and surfaced as a % of billed prompt tokens."""

    class UsageLLM:
        def __init__(self) -> None:
            self.n = 0

        async def chat(self, messages, *, tools=None, model=None, temperature=None,
                       max_tokens=None, stream=False, on_token=None):
            self.n += 1
            if self.n == 1:
                return LLMResponse(content="", tool_calls=[ToolCall("call_1", "read", {"path": "x.txt"})],
                                   usage={"prompt_tokens": 500, "completion_tokens": 5, "total_tokens": 505,
                                          "cache_read_tokens": 400})
            return LLMResponse(content="done", usage={"prompt_tokens": 500, "completion_tokens": 5, "total_tokens": 505,
                                                      "cache_read_tokens": 500})

    (tmp_path / "x.txt").write_text("hi", encoding="utf-8")
    agent = _agent(tmp_path, UsageLLM())
    result = await agent.run("go")
    assert result.usage["cache_read_tokens"] == 900
    assert result.usage["cache_read_pct"] == 90  # 900 of 1000 prompt tokens
    assert result.usage["prompt_tokens"] == 1000


async def test_on_usage_reports_each_call_in_real_time(tmp_path):
    """The on_usage hook fires once per LLM call, as soon as usage is known."""
    usages: list[dict] = []
    (tmp_path / "x.txt").write_text("hi", encoding="utf-8")

    class UsageLLM:
        def __init__(self) -> None:
            self.n = 0

        async def chat(self, messages, *, tools=None, model=None, temperature=None,
                       max_tokens=None, stream=False, on_token=None):
            self.n += 1
            if self.n == 1:
                return LLMResponse(content="", tool_calls=[ToolCall("call_1", "read", {"path": "x.txt"})],
                                   usage={"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7})
            return LLMResponse(content="done", usage={"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4})

    agent = _agent(tmp_path, UsageLLM(), on_usage=usages.append)
    await agent.run("go")
    assert usages == [
        {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
        {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
    ]


async def test_on_tool_result_notified(tmp_path):
    results_seen: list[tuple[str, str]] = []
    (tmp_path / "x.txt").write_text("hello world", encoding="utf-8")
    llm = llm_with_script([tool_response("read", {"path": "x.txt"}), final_response("done")])
    agent = _agent(tmp_path, llm, on_tool_result=lambda n, a, r: results_seen.append((n, r)))
    await agent.run("read")
    assert results_seen and results_seen[0][0] == "read"
    assert "hello world" in results_seen[0][1]
