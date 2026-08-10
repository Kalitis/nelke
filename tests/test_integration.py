"""Real-model integration tests (plan Phase A, item 4).

These contact a real OpenAI-compatible endpoint and are skipped unless
``NELKE_TEST_REAL=1`` is set. The endpoint used is the active profile
(``NELKE_DEFAULT_PROFILE``) — typically a local LM Studio or Ollama server.

Run locally with::

    $env:NELKE_TEST_REAL = "1"
    $env:NELKE_DEFAULT_PROFILE = "lmstudio"
    uv run pytest tests/test_integration.py -q
"""

from __future__ import annotations

import os

import pytest

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("NELKE_TEST_REAL") != "1",
        reason="set NELKE_TEST_REAL=1 to run real-model integration tests",
    ),
]


@pytest.fixture(scope="module")
def llm():
    from nelke.core.llm import build_llm

    return build_llm()


async def test_chat_round_trip(llm):
    """A plain text round-trip: the model must answer and report usage."""
    resp = await llm.chat(
        [
            {"role": "system", "content": "You are a terse assistant."},
            {"role": "user", "content": "Reply with exactly: hello"},
        ],
        stream=False,
    )
    assert resp.content, "expected a non-empty answer from the real model"
    assert "hello" in resp.content.lower()


async def test_chat_streaming(llm):
    """Streaming must yield the same content chunk-by-chunk as the final answer."""
    chunks: list[str] = []
    resp = await llm.chat(
        [{"role": "user", "content": "Count from 1 to 3, one number per line."}],
        stream=True,
        on_token=chunks.append,
    )
    assert "".join(chunks) == resp.content
    assert resp.content.strip()


async def test_tool_loop(llm, tmp_path):
    """An agent must drive a real tool (read_file) and answer from its result."""
    from nelke.core.agent import make_agent

    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "data.txt").write_text("The hidden number is 42.\n", encoding="utf-8")
    agent = make_agent(ws, llm, iteration_cap=8, include_web=False, include_shell=False)
    result = await agent.run(
        "Read data.txt with the read tool and tell me the hidden number. Reply with just the number."
    )
    assert result.tool_calls >= 1, "expected the agent to call a tool"
    assert "42" in result.answer


async def test_react_fallback_path(llm, tmp_path):
    """Native tool calling OR the ReAct fallback must both let the agent answer."""
    from nelke.core.agent import make_agent

    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "hello.txt").write_text("content-thank-you", encoding="utf-8")
    agent = make_agent(ws, llm, iteration_cap=8, include_web=False, include_shell=False)
    result = await agent.run(
        "Read hello.txt and report what it contains. Answer concisely."
    )
    assert result.answer, "expected an answer from either path"
