"""Telegram bot frontend (aiogram 3) — a thin I/O adapter over the Nelke core.

Commands:
  /start            greeting + help
  /chat <text>      same as just typing the text — ack immediately, run the
                    agent, edit the message as the answer streams in
                    (throttled), then post usage. Each Telegram chat keeps a
                    persistent session, so conversations continue across
                    messages (same as web/TUI).
  /new              start a brand-new chat (fresh session; memory is global).
  /history          show the current chat's persisted transcript.
  /chats            list this Telegram chat's conversations.
  /open <id>        resume an old conversation (/chats shows the ids).
  /improve <obj>    ack, run a self-improvement cycle; the human gate sends an
                    inline ✅/❌ keyboard and awaits the button press.
  /review           list open human reviews.
  /review approve <id> | /review reject <id>
                    approve/reject a pending review from text — works even if
                    the inline keyboard is gone (e.g. the bot restarted while a
                    cycle was parked on the human gate).
  /cancel           cancel the running task/cycle for this chat.
  /memory [query]   list memory files, or recall hits for a query.

Plain text messages are treated as `/chat <text>` (no prefix needed). In
group/supergroup chats the bot only answers messages that mention it or reply
to one of its own messages.

The bot token is read from ``NELKE_TELEGRAM_TOKEN`` via the shared
``load_env_files()`` path (same as ``OPENAI_API_KEY``). Handlers are methods
on :class:`NelkeBot` so tests can drive them with a fake bot — no polling.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
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
from nelke.core.llm import usage_cache_pct
from nelke.core.services import Callbacks

load_env_files()

_ACK_CHAT = "🤔 working…"
_ACK_IMPROVE = "🔧 cycle starting…"
_EDIT_INTERVAL = 0.5  # throttle message edits while streaming
_PROGRESS_REPORT_EVENTS = 12  # one TG progress report per this many cycle events
_REPORT_KINDS = {
    "cycle_start", "step_start", "gate", "commit", "boot_check_failed",
    "step_ok", "propose_complete", "ai_review", "review_feedback",
    "awaiting_human", "human_pending", "human_rejected", "merged",
    "cycle_error", "deps_synced", "deps_failed", "idle",
}


def _short_frontend(frontend: str | None) -> str:
    """Compact origin label for a chat (which frontend created it)."""
    return {"telegram": "tg", "tui": "tui", "web": "web"}.get(
        str(frontend or ""), str(frontend or "?")
    )


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
    usage_total = 0
    usage_calls = 0
    for ev in events:
        kind = ev["kind"]
        if kind == "usage":
            import json

            payload = {}
            try:
                payload = json.loads(ev["payload"] or "{}")
            except (ValueError, TypeError):
                pass
            usage_total += int(payload.get("total_tokens", 0) or 0)
            usage_calls += 1
            continue
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
    if usage_total:
        lines.append(f"tokens: {usage_total} ({usage_calls} calls)")
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
    # chat_id -> current persistent session id (chat history)
    current_sessions: dict[int, str] = field(default_factory=dict)
    # cycle_id -> Future[bool] parked on the human gate
    gate_futures: dict[str, asyncio.Future[bool]] = field(default_factory=dict)
    # cycle_id -> (chat_id, message_id) to edit when the gate resolves
    gate_messages: dict[str, tuple[int, int]] = field(default_factory=dict)
    # bot's @username, fetched lazily (used to detect mentions in group chats)
    bot_username: str | None = None


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
        self.router.message.register(self.on_new_chat, Command("new"))
        self.router.message.register(self.on_history, Command("history"))
        self.router.message.register(self.on_chats, Command("chats"))
        self.router.message.register(self.on_open, Command("open"))
        self.router.message.register(self.on_improve, Command("improve"))
        self.router.message.register(self.on_review, Command("review"))
        self.router.message.register(self.on_cancel, Command("cancel"))
        self.router.message.register(self.on_memory, Command("memory"))
        self.router.message.register(self.on_project, Command("project"))
        self.router.callback_query.register(self.on_review_callback, F.data.startswith("review:"))
        # Catch-all: plain text = /chat. Registered last so command handlers
        # win; messages that begin with "/" (e.g. unknown commands) are ignored.
        self.router.message.register(self.on_user_message, F.text)

    # ---- /start -------------------------------------------------------------
    async def on_start(self, message: Message) -> None:
        await message.answer(
            "Nelke bot.\n"
            "just type a message and Nelke answers (no /chat prefix needed)\n"
            "/new — start a new chat\n"
            "/history — show this chat's transcript\n"
            "/chats — list your chats\n"
            "/open <id> — continue an old chat\n"
            "/improve <objective> — self-improvement cycle\n"
            "/review — list / approve / reject pending reviews\n"
            "/cancel — stop the running task\n"
            "/memory [query] — list memory or recall hits\n"
            "/project create <name> — make a project\n"
            "/project list — show all projects\n"
            "/project show <id> — project card with chats/memory\n"
            "/project set_stage <id> <stage> — set project stage (idea/active/done…)\n"
            "/project set_memory <id> <note> <text> — write a memory note\n"
            "/project link_chat <id> — attach this chat to a project",
        )

    # ---- chat session management -------------------------------------------
    @staticmethod
    def _session_meta(row: Any) -> dict[str, Any]:
        try:
            meta = json.loads(row["meta"] or "{}")
            return meta if isinstance(meta, dict) else {}
        except (ValueError, TypeError):
            return {}

    def _resolve_session(self, chat_id: int) -> str | None:
        """The current persisted session id for a Telegram chat, or ``None``."""
        db = services.open_db(self.state.settings)
        sid = self.state.current_sessions.get(chat_id)
        if sid and db.get_session(sid) is not None:
            return sid
        for row in db.list_sessions(frontend="telegram", limit=300):
            if str(self._session_meta(row).get("tg_chat_id")) == str(chat_id):
                self.state.current_sessions[chat_id] = str(row["id"])
                return str(row["id"])
        return None

    def _ensure_session(self, chat_id: int) -> str:
        sid = self._resolve_session(chat_id)
        if sid is not None:
            return sid
        return self._new_session(chat_id)

    def _new_session(self, chat_id: int) -> str:
        db = services.open_db(self.state.settings)
        sid = db.create_session("telegram", meta={"tg_chat_id": str(chat_id)})
        self.state.current_sessions[chat_id] = sid
        return sid

    @staticmethod
    def _title_from_first_user(db: Any, session_id: str) -> str:
        row = db.first_user_message(session_id)
        if row is None or not (row["content"] or "").strip():
            return "New chat"
        return " ".join(row["content"].split())[:60] or "New chat"

    def _ensure_title(self, session_id: str) -> None:
        """Title a chat from its first user message once, when unset."""
        db = services.open_db(self.state.settings)
        row = db.get_session(session_id)
        if row is None or self._session_meta(row).get("title"):
            return
        row2 = db.first_user_message(session_id)
        if row2 is None:
            return
        title = " ".join((row2["content"] or "").split())[:60].strip()
        if title:
            db.update_session_meta(session_id, title=title)

    # ---- /chat --------------------------------------------------------------
    async def on_chat(self, message: Message, command: CommandObject) -> None:
        text = (command.args or "").strip()
        if not text:
            await message.answer("Usage: /chat <text>")
            return
        await self._dispatch_chat(message, text)

    async def on_user_message(self, message: Message) -> None:
        """Plain text = /chat (no prefix needed). See module docstring."""
        text = (message.text or "").strip()
        if not text or text.startswith("/"):
            return  # commands are routed to their own handlers
        # In groups, only answer messages addressed to the bot, so we never
        # reply to every message in a shared chat.
        if getattr(message.chat, "type", "private") != "private" and not await self._addressed_to_bot(message):
            return
        await self._dispatch_chat(message, text)

    async def _dispatch_chat(self, message: Message, text: str) -> None:
        chat_id = message.chat.id
        running = self.state.tasks.get(chat_id)
        if running is not None and not running.done():
            await message.answer("busy — wait for the current task to finish or /cancel")
            return
        session_id = self._ensure_session(chat_id)
        ack = await message.answer(_ACK_CHAT)
        task = asyncio.create_task(self._run_chat(chat_id, ack.message_id, text, session_id))
        self.state.tasks[chat_id] = task

    async def _ensure_bot_username(self) -> str | None:
        """The bot's @username, cached after the first ``get_me()`` call."""
        if self.state.bot_username is None:
            me_call = getattr(self.bot, "get_me", None)
            if me_call is None:
                self.state.bot_username = ""
            else:
                try:
                    me = await me_call()
                    self.state.bot_username = (getattr(me, "username", None) or "").strip()
                except Exception:  # noqa: BLE001 - mention detection is best-effort
                    self.state.bot_username = ""
        return self.state.bot_username or None

    async def _addressed_to_bot(self, message: Message) -> bool:
        """True when a group message is aimed at the bot (reply or @mention)."""
        bot_id = getattr(self.bot, "id", None)
        rto = getattr(message, "reply_to_message", None)
        if bot_id is not None and rto is not None:
            if getattr(getattr(rto, "from_user", None), "id", None) == bot_id:
                return True
        uname = await self._ensure_bot_username()
        if uname and message.text and f"@{uname}".lower() in message.text.lower():
            return True
        return False

    async def _run_chat(self, chat_id: int, message_id: int, text: str, session_id: str) -> None:
        last_edit = 0.0
        live_usage: dict[str, int] = {
            "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
            "cache_read_tokens": 0, "calls": 0,
        }

        async def edit(answer: str, usage: dict[str, Any] | None = None) -> None:
            nonlocal last_edit
            now = asyncio.get_event_loop().time()
            if now - last_edit < _EDIT_INTERVAL:
                return
            last_edit = now
            body = answer or "…"
            usage = usage or live_usage
            if usage.get("total_tokens"):
                body += f"\n\ntokens: {usage['total_tokens']} (cache {usage_cache_pct(usage)}% of prompt)"
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

        def on_usage(usage: dict[str, Any]) -> None:
            for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                live_usage[key] += int(usage.get(key, 0) or 0)
            live_usage["cache_read_tokens"] += int(usage.get("cache_read_tokens", 0) or 0)
            live_usage["calls"] += 1

        try:
            result, _sid, _msg_id = await services.run_chat_turn(
                text, self.state.settings, self.state.profile, session_id,
                frontend_name="telegram",
                callbacks=Callbacks(on_token=on_token, on_usage=on_usage, stream=True),
                repo=self.state.repo_path,
                llm_factory=self.state.llm_factory or services._llm_factory_default,
            )
            done.set()
            await asyncio.gather(editor, return_exceptions=True)
            usage = result.usage or live_usage
            await edit(result.answer or "(no answer)", usage)
            if not result.answer:
                await self.bot.send_message(chat_id, "(no answer)")
            self._ensure_title(session_id)
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

    # ---- /new ---------------------------------------------------------------
    async def on_new_chat(self, message: Message) -> None:
        chat_id = message.chat.id
        running = self.state.tasks.get(chat_id)
        if running is not None and not running.done():
            await message.answer("busy — wait for the current task to finish or /cancel")
            return
        sid = self._new_session(chat_id)
        await message.answer(f"New chat started ({sid[-8:]}).\n/chat <text> to ask.")

    # ---- /history -----------------------------------------------------------
    async def on_history(self, message: Message, command: CommandObject) -> None:
        chat_id = message.chat.id
        sid = self._resolve_session(chat_id)
        if sid is None:
            await message.answer("No chat yet — /chat <text> to start one.")
            return
        messages = services.get_chat_messages(self.state.settings, sid)
        lines: list[str] = []
        for m in messages:
            role = str(m.get("role") or "")
            content = str(m.get("content") or "").strip()
            if role == "tool" or not content:
                continue
            prefix = "🧑 You:" if role == "user" else "🤖 Nelke:"
            lines.append(f"{prefix}\n{content}")
        if not lines:
            await message.answer("No messages yet in this chat.")
            return
        await message.answer("\n\n".join(lines[-40:])[:4000])

    # ---- /chats -------------------------------------------------------------
    async def on_chats(self, message: Message) -> None:
        chat_id = message.chat.id
        db = services.open_db(self.state.settings)
        current = self.state.current_sessions.get(chat_id)
        lines: list[str] = []
        # Chats are shared across web/TUI/Telegram, so this lists every
        # conversation (with its origin) — a chat started on the web can be
        # continued here with /open, and vice versa.
        for row in db.list_sessions(limit=200):
            title = self._session_meta(row).get("title") or self._title_from_first_user(db, row["id"])
            origin = _short_frontend(row["frontend"])
            marker = " ◀ current" if current and str(row["id"]) == str(current) else ""
            lines.append(f"· {row['id'][-8:]} [{origin}] {title} ({row['message_count'] or 0} msgs){marker}")
        if not lines:
            await message.answer("No chats yet — /chat <text> to start one.")
            return
        await message.answer(
            "All chats (web/TUI/Telegram; /open <id> to continue anywhere):\n"
            + "\n".join(lines)[:4000]
        )

    # ---- /open --------------------------------------------------------------
    async def on_open(self, message: Message, command: CommandObject) -> None:
        arg = (command.args or "").strip()
        chat_id = message.chat.id
        if not arg:
            await message.answer("Usage: /open <id> — the id's last 8 chars from /chats")
            return
        db = services.open_db(self.state.settings)
        target: str | None = None
        # Any chat, from any frontend, is resumable — not just this Telegram
        # chat's own sessions.
        for row in db.list_sessions(limit=2000):
            rid = str(row["id"])
            if rid == arg or rid.endswith(arg) or rid.startswith(arg):
                target = rid
                break
        if target is None:
            await message.answer(f"chat '{arg}' not found — /chats to list your chats")
            return
        self.state.current_sessions[chat_id] = target
        sess_row = db.get_session(target)
        if sess_row is not None:
            origin = _short_frontend(sess_row["frontend"])
            title = self._session_meta(sess_row).get("title") or self._title_from_first_user(db, target)
        else:
            origin = ""
            title = "New chat"
        await message.answer(
            f"Opened [{origin}] chat {target[-8:]} {title}.\n/history to see it, /chat <text> to continue."
        )

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

        report_tasks: set[asyncio.Task[Any]] = set()
        cycle_usage: dict[str, int] = {
            "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
            "cache_read_tokens": 0, "calls": 0,
        }

        def on_event(ev: Any) -> None:
            # schedule the progress report as a task: CycleEngine._emit calls
            # on_event synchronously, so the async maybe_report needs a task.
            if ev.kind not in _REPORT_KINDS:
                return
            task = asyncio.create_task(maybe_report(ev.cycle_id))
            report_tasks.add(task)
            task.add_done_callback(report_tasks.discard)

        def on_usage(usage: dict[str, Any]) -> None:
            for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                cycle_usage[key] += int(usage.get(key, 0) or 0)
            cycle_usage["cache_read_tokens"] += int(usage.get("cache_read_tokens", 0) or 0)
            cycle_usage["calls"] += 1

        try:
            result = await services.run_cycle(
                objective, self.state.settings, self.state.profile,
                human_approve=human_gate,
                repo_path=self.state.repo_path,
                llm_factory=self.state.llm_factory or services._llm_factory_default,
                governance=self.state.governance,
                on_event=on_event,
                on_usage=on_usage,
            )
            if report_tasks:
                await asyncio.gather(*report_tasks, return_exceptions=True)
            final = f"cycle {result.status}\nbranch: {result.branch}\nsteps: {result.steps}"
            if cycle_usage["total_tokens"]:
                pct = usage_cache_pct(cycle_usage)
                final += f"\ntokens: {cycle_usage['total_tokens']} ({cycle_usage['calls']} calls, cache {pct}%)"
            await self.bot.edit_message_text(
                final,
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

    # ---- /review (textual approval path) ------------------------------------
    async def on_review(self, message: Message, command: CommandObject) -> None:
        args = (command.args or "").strip().split()
        if not args or args[0].lower() == "list":
            await self._review_list(message)
            return
        action = args[0].lower()
        if action in ("approve", "reject") and len(args) >= 2:
            await self._review_resolve(message, action, args[1])
            return
        await message.answer("Usage: /review | /review list | /review approve <id> | /review reject <id>")

    async def _review_list(self, message: Message) -> None:
        db = services.open_db(self.state.settings)
        open_reqs = [r for r in db.list_review_requests(open_only=True) if r["kind"] == "human"]
        if not open_reqs:
            await message.answer("no open human reviews")
            return
        lines = []
        for req in open_reqs[-20:]:
            cycle = db.get_cycle(req["cycle_id"])
            obj = (cycle["objective"][:60] if cycle else req["cycle_id"])
            branch = cycle["branch"] if cycle else ""
            lines.append(
                f"· {branch} {obj}\n  /review approve {req['id'][-12:]} · reject"
            )
        await message.answer("Open human reviews:\n" + "\n".join(lines)[:4000])

    async def _review_resolve(self, message: Message, action: str, prefix: str) -> None:
        approved = action == "approve"
        db = services.open_db(self.state.settings)
        matches = [
            r for r in db.list_review_requests(open_only=True)
            if r["kind"] == "human"
            and (r["id"] == prefix or r["id"].startswith(prefix) or r["id"].endswith(prefix))
        ]
        if not matches:
            await message.answer(f"no open review matches '{prefix}' — /review to list")
            return
        req = matches[0]
        cycle_id: str = req["cycle_id"] or ""
        # In-process: the cycle may be parked on this bot's in-memory gate —
        # resolve the future and let the running engine finish (merge + DB
        # update). Don't touch the DB here or the engine would merge twice.
        future = self.state.gate_futures.get(cycle_id) if cycle_id else None
        if future is not None and not future.done():
            self.state.gate_futures.pop(cycle_id, None)
            future.set_result(approved)
            cid, mid = self.state.gate_messages.pop(cycle_id, (None, None))
            if cid is not None:
                label = "✅ approved — merging" if approved else "❌ rejected — keeping branch"
                try:
                    await self.bot.edit_message_text(label, chat_id=cid, message_id=mid)
                except Exception:  # noqa: BLE001 - best-effort label edit
                    pass
            await message.answer("approved — merging…" if approved else "rejected — branch kept")
            return
        # Recovery path (bot restarted or no running gate): resolve straight
        # from the DB. Skip cycles that already finished to avoid double-merges.
        if cycle_id:
            cycle = db.get_cycle(cycle_id)
            if cycle is not None and cycle["status"] in ("merged", "rejected", "merge-conflict"):
                await message.answer(f"already resolved: {cycle['status']}")
                return
        try:
            result = services.resolve_review(
                self.state.settings, req["id"], "approved" if approved else "rejected",
                repo_path=self.state.repo_path,
            )
        except Exception as exc:  # noqa: BLE001 - surface resolution failures cleanly
            await message.answer(f"error: {exc}"[:4000])
            return
        if result.error:
            await message.answer(f"{'approved' if approved else 'rejected'} — {result.error}"[:4000])
            return
        await message.answer(
            f"{'✅ approved' if approved else '❌ rejected'} — {result.status}"
        )

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

    # ---- /project ----------------------------------------------------------
    _PROJECT_USAGE = (
        "Usage:\n"
        "/project create <name>\n"
        "/project list\n"
        "/project show <id>\n"
        "/project set_stage <id> <stage>\n"
        "/project set_memory <id> <note.md> <text>\n"
        "/project link_chat <id>"
    )

    def _resolve_project_id(self, prefix: str) -> str | None:
        """Find a project id by exact / prefix / suffix match (like /open)."""
        db = services.open_db(self.state.settings)
        for row in db.list_projects(limit=2000):
            pid = str(row["id"])
            if pid == prefix or pid.endswith(prefix) or pid.startswith(prefix):
                return pid
        return None

    async def on_project(self, message: Message, command: CommandObject) -> None:
        args = (command.args or "").strip().split()
        if not args:
            await message.answer(self._PROJECT_USAGE)
            return
        sub = args[0].lower()
        rest = args[1:]
        try:
            if sub == "create":
                await self._project_create(message, rest)
            elif sub == "list":
                await self._project_list(message)
            elif sub == "show":
                await self._project_show(message, rest)
            elif sub == "set_stage":
                await self._project_set_stage(message, rest)
            elif sub == "set_memory":
                await self._project_set_memory(message, rest)
            elif sub == "link_chat":
                await self._project_link_chat(message, rest)
            else:
                await message.answer(self._PROJECT_USAGE)
        except ValueError as exc:  # noqa: BLE001 - service-level input validation
            await message.answer(str(exc)[:4000])
        except Exception as exc:  # noqa: BLE001 - surface other failures cleanly
            await message.answer(f"error: {exc}"[:4000])

    async def _project_create(self, message: Message, rest: list[str]) -> None:
        if not rest:
            await message.answer("Usage: /project create <name>")
            return
        name = " ".join(rest).strip()
        repo = self.state.repo_path or services.find_repo(self.state.settings)
        pid = services.create_project(self.state.settings, name=name, repo=repo)
        await message.answer(
            f"✅ created project *{name}*\n"
            f"id: `{pid}`\n"
            f"/project set_stage {pid[-8:]} <stage> — set its stage"
        )

    async def _project_list(self, message: Message) -> None:
        repo = self.state.repo_path or services.find_repo(self.state.settings)
        projects = services.list_projects(self.state.settings, repo=repo)
        if not projects:
            await message.answer("no projects yet — /project create <name>")
            return
        lines: list[str] = ["Projects:"]
        for p in projects:
            stage = p.get("stage") or "—"
            mem = services.project_memory_files(self.state.settings, p["id"], repo=repo)
            lines.append(
                f"· *{p['name']}*  [{stage}]  ({p['chat_count']} chats · {len(mem)} memory)\n"
                f"  id: {p['id'][-8:]} · /project show {p['id'][-8:]}"
            )
        await message.answer("\n".join(lines)[:4000])

    async def _project_show(self, message: Message, rest: list[str]) -> None:
        if not rest:
            await message.answer("Usage: /project show <id>")
            return
        pid = self._resolve_project_id(rest[0])
        if pid is None:
            await message.answer(f"project '{rest[0]}' not found — /project list")
            return
        repo = self.state.repo_path or services.find_repo(self.state.settings)
        project = services.get_project(self.state.settings, pid, repo=repo)
        if project is None:
            await message.answer("project not found")
            return
        stage = project.get("stage") or "—"
        lines = [
            f"📂 *{project['name']}*",
            f"stage: {stage}   ·   id: `{project['id']}`",
            f"created {project.get('created_at')} · updated {project.get('updated_at')}",
        ]
        desc = project.get("description") or ""
        lines.append(f"description: {desc}" if desc else "(no description)")
        chats = project.get("chats") or []
        lines.append(f"💬 {len(chats)} chat{'s' if len(chats) != 1 else ''}:")
        for c in chats[:15]:
            origin = _short_frontend(c.get("frontend"))
            count = c.get("message_count", 0)
            lines.append(f"  · [{origin}] {str(c['id'])[-8:]} ({count} msgs)")
        if len(chats) > 15:
            lines.append(f"  … and {len(chats) - 15} more")
        mem = project.get("memory_files") or []
        lines.append(f"🧠 {len(mem)} memory note{'s' if len(mem) != 1 else ''}:")
        for m in mem[:15]:
            lines.append(f"  · {m['name']} ({m['size']} B)")
        if len(mem) > 15:
            lines.append(f"  … and {len(mem) - 15} more")
        if not mem:
            lines.append("  (no memory yet — /project set_memory)")
        await message.answer("\n".join(lines)[:4000])

    async def _project_set_stage(self, message: Message, rest: list[str]) -> None:
        if len(rest) < 2:
            await message.answer("Usage: /project set_stage <id> <stage>")
            return
        prefix, stage_parts = rest[0], rest[1:]
        stage = " ".join(stage_parts).strip()
        pid = self._resolve_project_id(prefix)
        if pid is None:
            await message.answer(f"project '{prefix}' not found — /project list")
            return
        services.update_project(self.state.settings, pid, stage=stage)
        await message.answer(f"✅ stage set to *{stage}*")

    async def _project_set_memory(self, message: Message, rest: list[str]) -> None:
        # /project set_memory <id> <note.md> <text...>
        if len(rest) < 3:
            await message.answer("Usage: /project set_memory <id> <note.md> <text>")
            return
        prefix, note = rest[0], rest[1]
        text = " ".join(rest[2:]).strip()
        if not text:
            await message.answer("memory text is required")
            return
        pid = self._resolve_project_id(prefix)
        if pid is None:
            await message.answer(f"project '{prefix}' not found — /project list")
            return
        repo = self.state.repo_path or services.find_repo(self.state.settings)
        ok = services.set_project_memory(
            self.state.settings, pid, note, text, append=False, repo=repo,
        )
        if not ok:
            await message.answer("could not write project memory")
            return
        await message.answer(f"✅ wrote *{note}* ({len(text)} chars)")

    async def _project_link_chat(self, message: Message, rest: list[str]) -> None:
        if not rest:
            await message.answer("Usage: /project link_chat <id>")
            return
        chat_id = message.chat.id
        session_id = self._ensure_session(chat_id)
        pid = self._resolve_project_id(rest[0])
        if pid is None:
            await message.answer(f"project '{rest[0]}' not found — /project list")
            return
        services.attach_chat_to_project(
            self.state.settings, chat_id=session_id, project_id=pid,
        )
        await message.answer(
            f"✅ this chat is now linked to the project\nid: {pid[-8:]}"
        )


# --------------------------------------------------------------------------- #
# Launcher
# --------------------------------------------------------------------------- #
_companion_started = False


def start_companion() -> None:
    """Start the Telegram bot in a daemon thread, alongside web/tui/cli.

    Lets the user keep talking to Nelke over Telegram while away from the
    keyboard (e.g. at lunch). No-op when ``NELKE_TELEGRAM_TOKEN`` is unset or
    the companion is already running; failures only warn on stderr so the
    host frontend never crashes because of the bot.
    """
    global _companion_started
    if _companion_started:
        return
    if not os.environ.get("NELKE_TELEGRAM_TOKEN"):
        return
    import threading

    def _run() -> None:
        try:
            launch()
        except Exception as exc:  # noqa: BLE001 - best-effort companion
            import sys

            print(f"[nelke] telegram companion stopped: {exc}", file=sys.stderr)

    _companion_started = True
    thread = threading.Thread(target=_run, name="nelke-telegram-companion", daemon=True)
    thread.start()


def launch() -> None:
    """Run the Telegram bot (long polling).

    Requires ``NELKE_TELEGRAM_TOKEN``. Optional ``NELKE_TELEGRAM_PROXY`` routes
    bot traffic through a local proxy (e.g. ``socks5h://127.0.0.1:12334`` or
    ``http://127.0.0.1:12334``) — needed where api.telegram.org is blocked.
    """
    from aiogram.client.session.aiohttp import AiohttpSession

    from nelke import __version__

    token = os.environ.get("NELKE_TELEGRAM_TOKEN")
    if not token:
        raise RuntimeError(
            "NELKE_TELEGRAM_TOKEN is not set. Add it to ~/.nelke/.env "
            "(see config.example.toml)."
        )
    proxy = os.environ.get("NELKE_TELEGRAM_PROXY")
    session = AiohttpSession(proxy=proxy) if proxy else None
    bot = Bot(token, session=session)
    print(f"nelke v{__version__} — Telegram bot (long polling)", file=sys.stderr)
    state = BotState(settings=Settings())
    nelke = NelkeBot(bot, state)
    dp = Dispatcher()
    dp.include_router(nelke.router)

    async def main() -> None:
        await dp.start_polling(bot)

    asyncio.run(main())
