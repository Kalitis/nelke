"""TUI frontend tests.

Two layers (per plan acceptance — "frontend-structural tests only"):
  1. A pure unit test of the ``StreamSink`` callback helper with a fake LLM —
     no Textual loop needed.
  2. A structural test that the app composes all three tabs without raising,
     driven through Textual's ``run_test`` pilot.
"""

from __future__ import annotations

import pytest
from conftest import final_response, tool_response

from nelke.core import services
from nelke.core.llm import LLMResponse


# --------------------------------------------------------------------------- #
# StreamSink / build_tui_callbacks (pure unit, no Textual loop)
# --------------------------------------------------------------------------- #
def test_stream_sink_collects_tokens_and_tools():
    from nelke.frontends.tui import StreamSink, build_tui_callbacks

    sink = StreamSink()
    callbacks = build_tui_callbacks(sink)
    assert callbacks.stream is True
    callbacks.on_token("hel")
    callbacks.on_token("lo")
    callbacks.on_tool("read", {"path": "x.txt"})
    callbacks.on_tool_result("read", {"path": "x.txt"}, "the answer is 42")
    assert sink.answer == "hello"
    assert sink.tools == ["read(path=x.txt)"]
    assert "the answer is 42" in sink.results[0]


async def test_callbacks_propagate_from_agent_run(tmp_path):
    """An agent driven by a FakeLLM feeds the sink through the callbacks."""
    from nelke.core.agent import Agent
    from nelke.core.tools.fs import ReadFileTool
    from nelke.frontends.tui import StreamSink

    (tmp_path / "x.txt").write_text("hello world", encoding="utf-8")

    state = {"i": 0}
    responses = [
        tool_response("read", {"path": "x.txt"}),
        final_response("I read it."),
    ]

    class _LLM:
        async def chat(self, messages, *, tools=None, stream=False, on_token=None, **_kw):
            resp = responses[min(state["i"], len(responses) - 1)]
            state["i"] += 1
            if stream and on_token and resp.content:
                on_token(resp.content)
            return resp

    sink = StreamSink()
    agent = Agent(
        name="t", system_prompt="test", tools=[ReadFileTool(tmp_path)],
        llm=_LLM(), stream=True,
        on_token=sink.on_token, on_tool=sink.on_tool, on_tool_result=sink.on_tool_result,
    )
    result = await agent.run("read x.txt")
    assert result.answer == "I read it."
    assert sink.answer == "I read it."
    assert sink.tools == ["read(path=x.txt)"]


# --------------------------------------------------------------------------- #
# Structural test: the app composes without raising
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_app_composes_all_tabs(settings, tmp_repo):
    """run_test drives the Textual pilot; compose must not raise."""
    from nelke.frontends.tui import AppStateData, NelkeTUI

    app = NelkeTUI(state=AppStateData(settings=settings, repo_path=tmp_repo.repo))
    async with app.run_test() as pilot:
        await pilot.pause()
        # all three tabs exist
        assert app.query_one("#chat-tab")
        assert app.query_one("#improve-tab")
        assert app.query_one("#memory-tab")
        # input widgets are present
        assert app.query_one("#chat-input")
        assert app.query_one("#improve-input")
        # chat log is a RichLog
        from textual.widgets import RichLog

        assert isinstance(app.query_one("#chat-log"), RichLog)


@pytest.mark.asyncio
async def test_chat_submit_writes_to_log(settings, tmp_repo):
    """Typing into the chat input and pressing Enter dispatches a worker.

    Uses a fake LLM factory so no real model is contacted.
    """
    from nelke.frontends.tui import AppStateData, NelkeTUI

    def factory(_profile):
        class _LLM:
            async def chat(self, messages, *, tools=None, stream=False, on_token=None, **_kw):
                if stream and on_token:
                    on_token("hi back")
                return LLMResponse(
                    content="hi back",
                    usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                )

        return _LLM()

    app = NelkeTUI(state=AppStateData(
        settings=settings, llm_factory=factory, repo_path=tmp_repo.repo,
    ))
    async with app.run_test() as pilot:
        chat_input = app.query_one("#chat-input")
        chat_input.value = "hello"
        from textual.widgets import Input

        chat_input.post_message(Input.Submitted(chat_input, "hello"))
        await pilot.pause(delay=0.3)
        await pilot.pause(delay=0.3)
        log_text = app.query_one("#chat-log").lines
        joined = "".join(str(line) for line in log_text)
        assert "hello" in joined  # user message echoed


# --------------------------------------------------------------------------- #
# Memory tab wiring
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_memory_tab_lists_files(settings, tmp_repo):
    from nelke.frontends.tui import AppStateData, NelkeTUI

    services.open_memory(tmp_repo.repo).write("notes/alpha.md", "# Alpha\n\nbody")
    app = NelkeTUI(state=AppStateData(settings=settings, repo_path=tmp_repo.repo))
    async with app.run_test() as pilot:
        await pilot.pause()
        lv = app.query_one("#memory-list")
        # one item rendered
        items = list(lv.children)
        assert items, "expected the alpha.md item in the memory list"
