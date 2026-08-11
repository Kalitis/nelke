"""Telegram bot frontend tests with a mocked aiogram Bot (no network).

Handlers are driven directly on :class:`NelkeBot` with a fake ``Bot`` that
records ``send_message``/``edit_message_text``/``answer_callback_query`` calls.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace as NS
from typing import Any

from conftest import FakeGovernance, final_response, tool_response

from nelke.core.llm import LLMResponse
from nelke.frontends.telegram_bot import BotState, NelkeBot


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #
class FakeBot:
    """Records bot API calls; methods return fake message objects.

    ``make_message`` returns a Message-like object whose ``answer`` records
    into ``self.sends``, so handler-level replies are captured here too.
    """

    def __init__(self) -> None:
        self.sends: list[dict[str, Any]] = []
        self.edits: list[dict[str, Any]] = []
        self.callback_answers: list[str] = []
        self._next_msg_id = 100

    async def send_message(self, chat_id: int, text: str, **kwargs: Any) -> NS:
        self.sends.append({"chat_id": chat_id, "text": text, **kwargs})
        self._next_msg_id += 1
        return NS(message_id=self._next_msg_id, chat=NS(id=chat_id))

    async def edit_message_text(self, text: str, *, chat_id: int, message_id: int, **kwargs: Any) -> NS:
        self.edits.append({"chat_id": chat_id, "message_id": message_id, "text": text, **kwargs})
        return NS()

    async def answer_callback_query(self, text: str = "", **kwargs: Any) -> NS:
        self.callback_answers.append(text)
        return NS()

    def make_message(self, text: str, chat_id: int = 1, message_id: int = 10) -> NS:
        bot = self

        async def answer(body: str, **kwargs: Any) -> NS:
            await bot.send_message(chat_id, body, **kwargs)
            return NS(message_id=99, chat=NS(id=chat_id))

        return NS(chat=NS(id=chat_id), text=text, message_id=message_id, answer=answer)


def _cmd(args: str) -> NS:
    return NS(args=args or None)


def _scripted_llm_factory(responses: list[LLMResponse], cycle_reviewer=None):
    state = {"i": 0, "r": 0}

    def _factory(_profile: str | None):
        class _LLM:
            async def chat(self, messages, *, tools=None, stream=False, on_token=None, **_kw):
                system = next((m["content"] for m in messages if m.get("role") == "system"), "")
                if "review gate" in system or "AI review gate" in system:
                    state["r"] += 1
                    return cycle_reviewer(messages, tools)
                resp = responses[min(state["i"], len(responses) - 1)]
                state["i"] += 1
                if stream and on_token is not None and resp.content:
                    on_token(resp.content)
                return resp

        return _LLM()

    return _factory


def _good_fix_plan():
    return [
        tool_response("self_write", {"path": "memory/facts/tg.md", "content": "# TG\n\ndone"}),
        final_response("step done"),
        tool_response("propose_cycle_complete", {}),
        final_response("complete"),
    ]


def _approved_reviewer():
    return lambda m, t: final_response("VERDICT: APPROVE\nSUMMARY: ok\nCOMMENTS: none")


def _make_bot(settings, tmp_repo, factory, governance=None) -> tuple[NelkeBot, FakeBot, BotState]:
    fake = FakeBot()
    state = BotState(
        settings=settings, llm_factory=factory,
        governance=governance, repo_path=tmp_repo.repo,
    )
    nelke = NelkeBot(fake, state)  # type: ignore[arg-type]
    return nelke, fake, state


# --------------------------------------------------------------------------- #
# /start
# --------------------------------------------------------------------------- #
async def test_start_answers_help(settings, tmp_repo):
    nelke, fake, _ = _make_bot(settings, tmp_repo, _scripted_llm_factory([final_response("x")]))
    msg = fake.make_message("/start")
    await nelke.on_start(msg)
    assert fake.sends and "Nelke bot" in fake.sends[0]["text"]


# --------------------------------------------------------------------------- #
# /chat
# --------------------------------------------------------------------------- #
async def _wait_for(state: BotState, chat_id: int = 1, timeout: float = 5.0) -> None:
    """Wait for the in-flight task for ``chat_id`` to finish."""
    task = state.tasks.get(chat_id)
    if task is None:
        return
    try:
        await asyncio.wait_for(asyncio.shield(task), timeout=timeout)
    except (asyncio.TimeoutError, Exception):
        pass


async def _wait_for_gate(state: BotState, timeout: float = 5.0) -> str:
    """Poll until the cycle parks on the human gate; return its cycle id."""
    import time

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if state.gate_futures:
            return next(iter(state.gate_futures))
        await asyncio.sleep(0.05)
    if state.gate_futures:
        return next(iter(state.gate_futures))
    raise AssertionError("cycle did not reach the human gate in time")


async def test_chat_acks_and_edits_answer(settings, tmp_repo):
    factory = _scripted_llm_factory([
        LLMResponse(content="hello world", usage={"prompt_tokens": 2, "completion_tokens": 2, "total_tokens": 4}),
    ])
    nelke, fake, state = _make_bot(settings, tmp_repo, factory)
    msg = fake.make_message("/chat hello")
    await nelke.on_chat(msg, _cmd("hello"))
    await _wait_for(state)
    assert any("hello world" in e["text"] for e in fake.edits)
    assert any("tokens: 4" in e["text"] for e in fake.edits)


async def test_chat_empty_args_usage_hint(settings, tmp_repo):
    factory = _scripted_llm_factory([final_response("x")])
    nelke, fake, _ = _make_bot(settings, tmp_repo, factory)
    msg = fake.make_message("/chat")
    await nelke.on_chat(msg, _cmd(""))
    assert fake.sends and "Usage" in fake.sends[0]["text"]


async def test_chat_tags_session_as_telegram(settings, tmp_repo):
    from nelke.core import services

    factory = _scripted_llm_factory([final_response("ok")])
    nelke, fake, state = _make_bot(settings, tmp_repo, factory)
    msg = fake.make_message("/chat hi")
    await nelke.on_chat(msg, _cmd("hi"))
    await _wait_for(state)
    db = services.open_db(settings)
    rows = db.connect().execute("SELECT frontend FROM sessions").fetchall()
    assert any(r["frontend"] == "telegram" for r in rows)


# --------------------------------------------------------------------------- #
# /improve + inline review
# --------------------------------------------------------------------------- #
def _callback(fake: FakeBot, data: str) -> NS:
    async def answer(text: str = "", **kwargs: Any) -> NS:
        await fake.answer_callback_query(text, **kwargs)
        return NS()

    return NS(data=data, answer=answer)


async def test_improve_approve_merges(settings, tmp_repo):
    factory = _scripted_llm_factory(_good_fix_plan(), cycle_reviewer=_approved_reviewer())
    nelke, fake, state = _make_bot(settings, tmp_repo, factory, governance=FakeGovernance())
    msg = fake.make_message("/improve add a memory lesson")
    await nelke.on_improve(msg, _cmd("add a memory lesson"))
    cycle_id = await _wait_for_gate(state)
    await nelke.on_review_callback(_callback(fake, f"review:approve:{cycle_id}"))
    await _wait_for(state)
    assert (tmp_repo.repo / "memory" / "facts" / "tg.md").exists()
    assert tmp_repo.current_branch() == "main"


async def test_improve_reject_keeps_branch(settings, tmp_repo):
    factory = _scripted_llm_factory(_good_fix_plan(), cycle_reviewer=_approved_reviewer())
    nelke, fake, state = _make_bot(settings, tmp_repo, factory, governance=FakeGovernance())
    msg = fake.make_message("/improve add a memory lesson")
    await nelke.on_improve(msg, _cmd("add a memory lesson"))
    cycle_id = await _wait_for_gate(state)
    await nelke.on_review_callback(_callback(fake, f"review:reject:{cycle_id}"))
    await _wait_for(state)
    assert tmp_repo.current_branch().startswith("improve/")


async def test_review_callback_invalid_data(settings, tmp_repo):
    factory = _scripted_llm_factory([final_response("x")])
    nelke, fake, _ = _make_bot(settings, tmp_repo, factory)
    await nelke.on_review_callback(_callback(fake, "bogus"))
    assert fake.callback_answers and "invalid" in fake.callback_answers[0]


async def test_review_callback_already_resolved(settings, tmp_repo):
    factory = _scripted_llm_factory([final_response("x")])
    nelke, fake, state = _make_bot(settings, tmp_repo, factory)
    fut: asyncio.Future[bool] = asyncio.get_event_loop().create_future()
    fut.set_result(True)
    state.gate_futures["c1"] = fut
    await nelke.on_review_callback(_callback(fake, "review:approve:c1"))
    assert "already" in fake.callback_answers[0]


# --------------------------------------------------------------------------- #
# /cancel
# --------------------------------------------------------------------------- #
async def test_cancel_with_nothing_running(settings, tmp_repo):
    factory = _scripted_llm_factory([final_response("x")])
    nelke, fake, _ = _make_bot(settings, tmp_repo, factory)
    msg = fake.make_message("/cancel")
    await nelke.on_cancel(msg)
    assert any("nothing to cancel" in s["text"] for s in fake.sends)


# --------------------------------------------------------------------------- #
# /memory
# --------------------------------------------------------------------------- #
async def test_memory_list(settings, tmp_repo):
    from nelke.core import services

    services.open_memory(tmp_repo.repo).write("facts/m.md", "# M\n\nbody")
    factory = _scripted_llm_factory([final_response("x")])
    nelke, fake, _ = _make_bot(settings, tmp_repo, factory)
    msg = fake.make_message("/memory")
    await nelke.on_memory(msg, _cmd(""))
    assert any("facts/m.md" in s["text"] for s in fake.sends)


async def test_memory_recall(settings, tmp_repo):
    from nelke.core import services

    services.open_memory(tmp_repo.repo).write("facts/k.md", "# K\n\ntags: math\n\nThe answer is 42.\n")
    factory = _scripted_llm_factory([final_response("x")])
    nelke, fake, _ = _make_bot(settings, tmp_repo, factory)
    msg = fake.make_message("/memory answer math")
    await nelke.on_memory(msg, _cmd("answer math"))
    assert any("facts/k.md" in s["text"] for s in fake.sends)


async def test_memory_empty(settings, tmp_repo):
    factory = _scripted_llm_factory([final_response("x")])
    nelke, fake, _ = _make_bot(settings, tmp_repo, factory)
    msg = fake.make_message("/memory")
    await nelke.on_memory(msg, _cmd(""))
    assert any("no memory files" in s["text"] for s in fake.sends)
