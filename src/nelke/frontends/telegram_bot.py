"""Telegram bot frontend (aiogram 3) — a thin I/O adapter over the Nelke core.

Commands:
  /start            greeting + help
  /chat <text>      ack immediately, run the agent, edit the message as the
                    answer streams in (throttled), then post usage.
  /improve <obj>    ack, run a self-improvement cycle; the human gate sends an
                    inline ✅/❌ keyboard and awaits the button press.
  /cancel           cancel the running task/cycle for this chat.
  /memory [query]   list memory files, or recall hits for a query.

The bot token is read from ``NELKE_TELEGRAM_TOKEN`` via the shared
``load_env_files()`` path (same as ``OPENAI_API_KEY``). Handlers are methods
on :class:`NelkeBot` so tests can drive them with a fake bot — no polling.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field
from typing import Any

from aiogram import Bot, Dispatcher, F, Router
from aiogram.filters import Command, CommandObject
from aiogram.types import (
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
)

from nelke.config import Settings, load_env_files
from nelke.core import services
from nelke.core.services import Callbacks

load_env_files()

_ACK_CHAT = "🤔 working…"
_ACK_IMPROVE = "🔧 cycle starting…"
_EDIT_INTERVAL = 0.5  # throttle message edits while streaming
_PROGRESS_REPORT_EVENTS = 12  # one TG progress report per this many cycle events


def _cycle_progress_report(settings: Settings, cycle_id: str, *, repo_path: Any = None) -> str:
    """Build a human-readable progress report from the persisted cycle trace."""
    db = services.open_db(settings)
    cycle = db.get_cycle(cycle_id)
    if cycle is None:
        return f"cycle {cycle_id}: unknown"
    events = db.list_cycle_events(cycle_id, limit=60)
    lines = [
        f"🔧 cycle: {cycle_id[:16]}…  status={cycle['status']}",
        f"objective: {cycle['objective'][:80]}",
        f"branch: {cycle['branch']}",
    ]
    for ev in events:
        kind = ev["kind"]
        if kind == "agent_token":
            continue
        if kind in {"agent_tool", "agent_tool_result"}:
            import json

            payload = {}
            try:
                payload = json.loads(ev["payload"] or "{}")
            except (ValueError, TypeError):
                pass
            tool = payload.get("tool", "")
            if kind == "agent_tool":
                args = payload.get("args") or {}
                args_txt = ", ".join(
                    f"{k}={str(v)[:25]}" for k, v in list(args.items())[:2]
                )
                lines.append(f"  · {tool}({args_txt})")
            else:
                lines.append(f"  · → {tool}: {payload.get('snippet', '')[:120]}")
            continue
        lines.append(f"  · {ev['kind']}: {(ev['message'] or '')[:160]}")
    return "\n".join(lines[:60])


# --------------------------------------------------------------------------- #
# Per-bot state
# --------------------------------------------------------------------------- #
@dataclass
class BotState:
    """In-flight tasks + human-gate futures, keyed by chat id."""

    settings: Settings
    profile: str | None = None
    llm_factory: Any = None
    governance: Any = None
    repo_path: Any = None
    # chat_id -> running asyncio.Task (for /cancel)
    tasks: dict[int, asyncio.Task[Any]] = field(default_factory=dict)
    # cycle_id -> Future[bool] parked on the human gate
    gate_futures: dict[str, asyncio.Future[bool]] = field(default_factory=dict)
    # cycle_id -> (chat_id, message_id) to edit when the gate resolves
    gate_messages: dict[str, tuple[int, int]] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Bot adapter
# --------------------------------------------------------------------------- #
class NelkeBot:
    """A Telegram adapter over :mod:`nelke.core.services`.

    Handlers are async methods taking the aiogram ``Message``/``CallbackQuery``
    plus the shared :class:`BotState`. Tests construct this with a fake ``Bot``
    that records calls.
    """

    def __init__(self, bot: Bot, state: BotState) -> None:
        self.bot = bot
        self.state = state
        self.router = Router(name="nelke")
        self._register()

    def _register(self) -> None:
        self.router.message.register(self.on_start, Command("start"))
        self.router.message.register(self.on_chat, Command("chat"))
        self.router.message.register(self.on_improve, Command("improve"))
        self.router.message.register(self.on_cancel, Command("cancel"))
        self.router.message.register(self.on_memory, Command("memory"))
        self.router.callback_query.register(self.on_review_callback, F.data.startswith("review:"))

    # ---- /start -------------------------------------------------------------
    async def on_start(self, message: Message) -> None:
        await message.answer(
            "Nelke bot.\n"
            "/chat <text> — ask Nelke\n"
            "/improve <objective> — self-improvement cycle\n"
            "/cancel — stop the running task\n"
            "/memory [query] — list memory or recall hits",
        )

    # ---- /chat --------------------------------------------------------------
    async def on_chat(self, message: Message, command: CommandObject) -> None:
        text = (command.args or "").strip()
        if not text:
            await message.answer("Usage: /chat <text>")
            return
        ack = await message.answer(_ACK_CHAT)
        chat_id = message.chat.id
        task = asyncio.create_task(self._run_chat(chat_id, ack.message_id, text))
        self.state.tasks[chat_id] = task

    async def _run_chat(self, chat_id: int, message_id: int, text: str) -> None:
        last_edit = 0.0

        async def edit(answer: str, usage: dict[str, Any] | None = None) -> None:
            nonlocal last_edit
            now = asyncio.get_event_loop().time()
            if now - last_edit < _EDIT_INTERVAL:
                return
            last_edit = now
            body = answer or "…"
            if usage and usage.get("total_tokens"):
                body += f"\n\ntokens: {usage['total_tokens']}"
            try:
                await self.bot.edit_message_text(body[:4000], chat_id=chat_id, message_id=message_id)
            except Exception:  # noqa: BLE001 - "message is not modified" etc. are harmless
                pass

        async def token_runner(sink_tokens: list[str], done: asyncio.Event) -> Any:
            while not done.is_set():
                await asyncio.sleep(_EDIT_INTERVAL)
                await edit("".join(sink_tokens))
            return None

        sink_tokens: list[str] = []
        done = asyncio.Event()
        editor = asyncio.create_task(token_runner(sink_tokens, done))

        def on_token(tok: str) -> None:
            sink_tokens.append(tok)

        try:
            result, _session_id = await services.run_task(
                text, self.state.settings, self.state.profile,
                frontend_name="telegram",
                callbacks=Callbacks(on_token=on_token, stream=True),
                repo=self.state.repo_path,
                llm_factory=self.state.llm_factory or services._llm_factory_default,
            )
            done.set()
            await asyncio.gather(editor, return_exceptions=True)
            usage = result.usage or {}
            await edit(result.answer or "(no answer)", usage)
            if not result.answer:
                await self.bot.send_message(chat_id, "(no answer)")
        except asyncio.CancelledError:
            done.set()
            await asyncio.gather(editor, return_exceptions=True)
            await self.bot.edit_message_text("cancelled", chat_id=chat_id, message_id=message_id)
            raise
        except Exception as exc:  # noqa: BLE001
            done.set()
            await asyncio.gather(editor, return_exceptions=True)
            await self.bot.edit_message_text(f"error: {exc}"[:4000], chat_id=chat_id, message_id=message_id)
        finally:
            self.state.tasks.pop(chat_id, None)

    # ---- /improve -----------------------------------------------------------
    async def on_improve(self, message: Message, command: CommandObject) -> None:
        objective = (command.args or "").strip()
        if not objective:
            await message.answer("Usage: /improve <objective>")
            return
        ack = await message.answer(_ACK_IMPROVE)
        chat_id = message.chat.id
        task = asyncio.create_task(self._run_improve(chat_id, ack.message_id, objective))
        self.state.tasks[chat_id] = task

    async def _run_improve(self, chat_id: int, message_id: int, objective: str) -> None:
        # Issue periodic progress reports so the user can see what the cycle did
        # without polling the message (TL;DR: one report per N cycle events).
        last_report_events = 0

        async def maybe_report(cycle_id: str) -> None:
            nonlocal last_report_events
            db = services.open_db(self.state.settings)
            try:
                n_events = len(db.list_cycle_events(cycle_id))
            except Exception:  # noqa: BLE001 - progress must never crash the cycle
                n_events = 0
            if n_events - last_report_events >= _PROGRESS_REPORT_EVENTS:
                last_report_events = n_events
                try:
                    report = _cycle_progress_report(
                        self.state.settings, cycle_id, repo_path=self.state.repo_path,
                    )
                    await self.bot.send_message(chat_id, report[:3500])
                except Exception:  # noqa: BLE001 - best-effort
                    pass

        async def human_gate(req: Any) -> bool:
            keyboard = InlineKeyboardMarkup(inline_keyboard=[[
                InlineKeyboardButton(text="✅ Approve", callback_data=f"review:approve:{req.cycle_id}"),
                InlineKeyboardButton(text="❌ Reject", callback_data=f"review:reject:{req.cycle_id}"),
            ]])
            diff_preview = (req.diff or "")[:1500]
            body = (
                f"Human review required.\nObjective: {req.objective}\n"
                f"Branch: {req.branch}\n\n```\n{diff_preview}\n```"
            )[:3500]
            await self.bot.edit_message_text(
                body, chat_id=chat_id, message_id=message_id, reply_markup=keyboard,
            )
            future: asyncio.Future[bool] = asyncio.get_event_loop().create_future()
            self.state.gate_futures[req.cycle_id] = future
            self.state.gate_messages[req.cycle_id] = (chat_id, message_id)
            return await future

        try:
            result = await services.run_cycle(
                objective, self.state.settings, self.state.profile,
                human_approve=human_gate,
                repo_path=self.state.repo_path,
                llm_factory=self.state.llm_factory or services._llm_factory_default,
                governance=self.state.governance,
                on_event=lambda ev: maybe_report(ev.cycle_id),
            )
            await self.bot.edit_message_text(
                f"cycle {result.status}\nbranch: {result.branch}\nsteps: {result.steps}",
                chat_id=chat_id, message_id=message_id,
            )
        except asyncio.CancelledError:
            await self.bot.edit_message_text("cancelled", chat_id=chat_id, message_id=message_id)
            raise
        except Exception as exc:  # noqa: BLE001
            await self.bot.edit_message_text(f"error: {exc}"[:4000], chat_id=chat_id, message_id=message_id)
        finally:
            self.state.tasks.pop(chat_id, None)

    async def on_review_callback(self, callback: CallbackQuery) -> None:
        # data shape: review:<approve|reject>:<cycle_id>
        parts = (callback.data or "").split(":")
        if len(parts) != 3:
            await callback.answer("invalid callback")
            return
        _prefix, decision_raw, cycle_id = parts
        approved = decision_raw == "approve"
        future = self.state.gate_futures.pop(cycle_id, None)
        if future is None or future.done():
            await callback.answer("already resolved")
            return
        future.set_result(approved)
        await callback.answer("approved" if approved else "rejected")
        chat_id, message_id = self.state.gate_messages.get(cycle_id, (None, None))
        label = "✅ approved — merging" if approved else "❌ rejected — keeping branch"
        if chat_id is not None:
            await self.bot.edit_message_text(label, chat_id=chat_id, message_id=message_id)

    # ---- /cancel ------------------------------------------------------------
    async def on_cancel(self, message: Message) -> None:
        chat_id = message.chat.id
        task = self.state.tasks.pop(chat_id, None)
        if task is not None and not task.done():
            task.cancel()
            await message.answer("cancelling…")
        else:
            await message.answer("nothing to cancel")

    # ---- /memory ------------------------------------------------------------
    async def on_memory(self, message: Message, command: CommandObject) -> None:
        repo = self.state.repo_path or services.find_repo(self.state.settings)
        query = (command.args or "").strip()
        if query:
            hits = services.recall_memory(repo, query)
            if not hits:
                await message.answer("no memory matches")
                return
            lines = [f"*{h.name}* (score {h.score})\n{h.snippet}" for h in hits]
            await message.answer("\n\n".join(lines)[:4000])
            return
        files = services.memory_overview(repo)
        if not files:
            await message.answer("no memory files yet")
            return
        lines = [f"{f['name']} ({f['size']} B)" for f in files]
        await message.answer("\n".join(lines)[:4000])


# --------------------------------------------------------------------------- #
# Launcher
# --------------------------------------------------------------------------- #
def launch() -> None:
    """Run the Telegram bot (long polling).

    Requires ``NELKE_TELEGRAM_TOKEN``. Optional ``NELKE_TELEGRAM_PROXY`` routes
    bot traffic through a local proxy (e.g. ``socks5h://127.0.0.1:12334`` or
    ``http://127.0.0.1:12334``) — needed where api.telegram.org is blocked.
    """
    from aiogram.client.session.aiohttp import AiohttpSession

    token = os.environ.get("NELKE_TELEGRAM_TOKEN")
    if not token:
        raise RuntimeError(
            "NELKE_TELEGRAM_TOKEN is not set. Add it to ~/.nelke/.env "
            "(see config.example.toml)."
        )
    proxy = os.environ.get("NELKE_TELEGRAM_PROXY")
    session = AiohttpSession(proxy=proxy) if proxy else None
    bot = Bot(token, session=session)
    state = BotState(settings=Settings())
    nelke = NelkeBot(bot, state)
    dp = Dispatcher()
    dp.include_router(nelke.router)

    async def main() -> None:
        await dp.start_polling(bot)

    asyncio.run(main())
