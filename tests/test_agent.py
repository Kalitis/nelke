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
    assert result.usage == {"prompt_tokens": 8, "completion_tokens": 3, "total_tokens": 11, "calls": 2}


async def test_on_tool_result_notified(tmp_path):
    results_seen: list[tuple[str, str]] = []
    (tmp_path / "x.txt").write_text("hello world", encoding="utf-8")
    llm = llm_with_script([tool_response("read", {"path": "x.txt"}), final_response("done")])
    agent = _agent(tmp_path, llm, on_tool_result=lambda n, a, r: results_seen.append((n, r)))
    await agent.run("read")
    assert results_seen and results_seen[0][0] == "read"
    assert "hello world" in results_seen[0][1]
