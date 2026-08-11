"""Textual TUI frontend — a thin I/O adapter over the Nelke core.

Tabs: Chat (streaming + multi-turn), Improve (live cycle events + review
modal), Memory (file list + recall). Agent/cycle work runs inside Textual
workers; streaming callbacks and cycle events are marshalled back onto the UI
loop via ``call_from_thread`` so the app never blocks. The review modal is an
async ``ModalScreen``: ``CycleEngine``'s awaitable ``human_approve`` gate
``await``s it directly.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Any

from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import (
    Button,
    Footer,
    Header,
    Input,
    ListItem,
    ListView,
    ProgressBar,
    RichLog,
    Static,
    TabbedContent,
    TabPane,
)

from nelke.config import Settings, load_env_files
from nelke.core import services
from nelke.core.llm import usage_cache_pct
from nelke.core.services import Callbacks

load_env_files()


# --------------------------------------------------------------------------- #
# Review modal
# --------------------------------------------------------------------------- #
class ReviewModal(ModalScreen[bool]):
    """The human-gate screen: shows the diff + AI verdict, returns approve/reject."""

    BINDINGS = [
        Binding("y", "decide(True)", "Approve", show=False),
        Binding("n", "decide(False)", "Reject", show=False),
        Binding("escape", "decide(False)", "Reject", show=False),
    ]

    def __init__(self, review: dict[str, Any]) -> None:
        super().__init__()
        self.review = review

    def compose(self) -> ComposeResult:
        r = self.review
        diff = (r.get("diff") or "")[:4000]
        yield Vertical(
            Static(f"[bold]Review request[/]: {r.get('id', '')}", classes="modal-title"),
            Static(f"Objective: {r.get('objective', '')}"),
            Static(f"Branch: {r.get('branch', '')}   Status: {r.get('status', '')}"),
            Static(f"AI verdict: {r.get('verdict', '')}"),
            Static(r.get("comments") or "", classes="modal-comments"),
            Static(diff, classes="modal-diff"),
            Horizontal(
                Button("Approve (y)", id="approve", variant="success"),
                Button("Reject (n)", id="reject", variant="error"),
                classes="modal-actions",
            ),
            classes="review-modal",
        )

    @on(Button.Pressed, "#approve")
    def _approve(self) -> None:
        self.dismiss(True)

    @on(Button.Pressed, "#reject")
    def _reject(self) -> None:
        self.dismiss(False)

    def action_decide(self, approved: bool) -> None:
        self.dismiss(approved)


# --------------------------------------------------------------------------- #
# Callbacks helper (unit-testable without a Textual loop)
# --------------------------------------------------------------------------- #
@dataclass
class StreamSink:
    """Collects streaming output so the TUI loop can render it in one place.

    Designed to be driven by ``Agent`` callbacks: ``on_token`` accumulates the
    answer, ``on_tool``/``on_tool_result`` record tool activity. The TUI
    workers drain these lists onto a ``RichLog`` via ``call_from_thread``.
    Separating the sink from the widget keeps it unit-testable without a
    running Textual app.
    """

    tokens: list[str] = field(default_factory=list)
    tools: list[str] = field(default_factory=list)
    results: list[str] = field(default_factory=list)
    usages: list[dict[str, Any]] = field(default_factory=list)

    def on_token(self, token: str) -> None:
        self.tokens.append(token)

    def on_tool(self, name: str, args: dict[str, Any]) -> None:
        self.tools.append(f"{name}({_fmt_args(args)})")

    def on_tool_result(self, name: str, args: dict[str, Any], result: str) -> None:
        snippet = " ".join(result.split())[:160]
        self.results.append(f"=> {name}: {snippet}")

    def on_usage(self, usage: dict[str, Any]) -> None:
        self.usages.append(dict(usage))

    @property
    def usage_total(self) -> dict[str, int]:
        totals: dict[str, int] = {
            "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "cache_read_tokens": 0,
            "calls": len(self.usages),
        }
        for u in self.usages:
            totals["prompt_tokens"] += int(u.get("prompt_tokens", 0) or 0)
            totals["completion_tokens"] += int(u.get("completion_tokens", 0) or 0)
            totals["total_tokens"] += int(u.get("total_tokens", 0) or 0)
            totals["cache_read_tokens"] += int(u.get("cache_read_tokens", 0) or 0)
        return totals

    @property
    def answer(self) -> str:
        return "".join(self.tokens)


def _usage_text(total: dict[str, int]) -> str:
    calls = int(total.get("calls", 0))
    return (
        f"tokens: {total.get('total_tokens', 0)} "
        f"(prompt {total.get('prompt_tokens', 0)} + completion {total.get('completion_tokens', 0)}"
        f", cache {usage_cache_pct(total)}%)"
        f" · {calls} call{'s' if calls != 1 else ''}"
    )


def _fmt_args(args: dict[str, Any]) -> str:
    items = []
    for k, v in list(args.items())[:2]:
        text = str(v)
        if len(text) > 60:
            text = text[:60] + "..."
        items.append(f"{k}={text}")
    return ", ".join(items)


def build_tui_callbacks(sink: StreamSink) -> Callbacks:
    """Wire a :class:`StreamSink` into a :class:`Callbacks` for ``run_task``."""
    return Callbacks(
        on_token=sink.on_token, on_tool=sink.on_tool,
        on_tool_result=sink.on_tool_result, on_usage=sink.on_usage,
        stream=True,
    )


# --------------------------------------------------------------------------- #
# App state
# --------------------------------------------------------------------------- #
@dataclass
class AppStateData:
    settings: Settings
    profile: str | None = None
    llm_factory: Any = None
    governance: Any = None
    repo_path: Any = None


# --------------------------------------------------------------------------- #
# Main app
# --------------------------------------------------------------------------- #
class NelkeTUI(App):
    """Nelke terminal UI."""

    CSS = """
    .review-modal { background: $panel; border: $border; padding: 1 2; width: 80; height: auto; max-height: 80%; }
    .modal-title { margin-bottom: 1; }
    .modal-diff { background: $surface; color: $text-muted; padding: 1; max-height: 20; overflow: auto; }
    .modal-comments { color: $text; margin: 1 0; }
    .modal-actions { height: auto; align-horizontal: right; padding: 1 0 0 0; }
    TabbedContent { height: 1fr; }
    RichLog { border: $border; height: 1fr; }
    ProgressBar { margin: 0 0 1 0; }
    #improve-status { margin-bottom: 1; color: $text-muted; }
    #chat-usage { margin-bottom: 1; color: $text-muted; }
    #improve-usage { margin-bottom: 1; color: $text-muted; }
    .tab-actions { height: auto; margin-bottom: 1; }
    .chat-split { height: 1fr; }
    .chat-split > ListView { width: 30; margin-right: 1; border: $border; }
    .chat-main { width: 1fr; height: 1fr; }
    .chat-sidemeta { color: $text-muted; }
    #chat-new { margin-right: 1; }
    #cycle-list { height: 8; border: $border; margin-bottom: 1; }
    #cycle-detail { height: 1fr; }
    """

    BINDINGS = [
        Binding("ctrl+c", "cancel_work", "Cancel", show=False),
        Binding("q", "quit", "Quit", show=True),
    ]

    def __init__(self, state: AppStateData | None = None) -> None:
        super().__init__()
        self.state = state or AppStateData(settings=Settings())
        # active chat session + parallel id lists for the ListView widgets
        self._active_chat: str | None = None
        self._chat_ids: list[str] = []
        self._cycle_ids: list[str] = []

    def compose(self) -> ComposeResult:
        yield Header()
        with TabbedContent():
            with TabPane("Chat", id="chat-tab"):
                yield Horizontal(
                    Button("+ New chat", id="chat-new", variant="primary"),
                    Button("Refresh", id="chat-refresh"),
                    classes="tab-actions",
                )
                with Horizontal(classes="chat-split"):
                    yield ListView(id="chat-list")
                    with Vertical(classes="chat-main"):
                        yield RichLog(id="chat-log", markup=True)
                        yield Static("tokens: 0", id="chat-usage")
                        yield Input(id="chat-input", placeholder="Ask Nelke… (Enter to send)")
            with TabPane("Improve", id="improve-tab"):
                yield Input(id="improve-input", placeholder="Objective to improve… (Enter to run)")
                yield ProgressBar(total=100, show_eta=False, id="improve-progress")
                yield Static("Idle — no cycle running", id="improve-status")
                yield Static("tokens: 0", id="improve-usage")
                yield RichLog(id="improve-log", markup=True)
                yield Static("Cycle history — select a cycle to inspect", classes="chat-sidemeta")
                yield ListView(id="cycle-list")
                yield RichLog(id="cycle-detail", markup=True, classes="chat-main")
            with TabPane("Memory", id="memory-tab"):
                yield ListView(id="memory-list")
                yield Input(id="recall-input", placeholder="Recall query… (Enter)")
                yield RichLog(id="memory-detail", markup=True)
        yield Footer()

    def on_mount(self) -> None:
        self.title = "Nelke"
        self._refresh_memory()
        self._refresh_chats()
        self._refresh_cycles()

    # ---- helpers ------------------------------------------------------------
    def _ui(self, fn: Any, *args: Any, **kwargs: Any) -> None:
        """Run a UI mutation from inside a worker, thread-safely.

        Async workers run on the app's own event loop, so the call can be
        direct; thread workers (if any are added later) must marshal through
        ``call_from_thread``. Detecting the app thread makes both paths work.
        """
        import threading

        if getattr(self, "_thread_id", None) == threading.get_ident():
            fn(*args, **kwargs)
        else:
            self.call_from_thread(fn, *args, **kwargs)

    # ---- Chat ---------------------------------------------------------------
    @on(Button.Pressed, "#chat-new")
    def _on_chat_new(self) -> None:
        chat_id = services.create_chat(self.state.settings, frontend="tui")
        self._active_chat = chat_id
        self._refresh_chats(select_id=chat_id)
        log = self._query_one_log("chat-log")
        log.clear()
        log.write("[dim]new chat — say hi.[/]")
        self.query_one("#chat-input", Input).focus()

    @on(Button.Pressed, "#chat-refresh")
    def _on_chat_refresh(self) -> None:
        self._refresh_chats()
        if self._active_chat:
            chat = services.get_chat(self.state.settings, self._active_chat, repo=self.state.repo_path)
            if chat:
                self._render_chat_history(chat["messages"], clear=True)

    @on(ListView.Selected, "#chat-list")
    def _on_chat_select(self, event: ListView.Selected) -> None:
        idx = event.list_view.index
        if idx is not None and 0 <= idx < len(self._chat_ids):
            self._active_chat = self._chat_ids[idx]
            self._load_chat_into_log(self._active_chat)

    def _refresh_chats(self, select_id: str | None = None) -> None:
        try:
            chats = services.list_chats(self.state.settings, frontend="tui")
        except Exception:  # noqa: BLE001 - chat list is best-effort
            return
        lv = self.query_one("#chat-list", ListView)
        lv.clear()
        self._chat_ids = []
        for c in chats:
            self._chat_ids.append(c["id"])
            lv.append(ListItem(Static(f"{c['title']}  ({c['message_count']})")))
        if select_id and select_id in self._chat_ids:
            lv.index = self._chat_ids.index(select_id)

    def _load_chat_into_log(self, chat_id: str) -> None:
        log = self._query_one_log("chat-log")
        log.clear()
        chat = services.get_chat(self.state.settings, chat_id, repo=self.state.repo_path)
        if chat is None:
            log.write("[bold red]chat not found[/]")
            return
        log.write(f"[bold]chat:[/] {chat['title']}  [dim]{chat['id']}[/]")
        if chat.get("memory"):
            log.write(f"[dim]memory: {', '.join(m['name'] for m in chat['memory'][:8])}[/]")
        self._render_chat_history(chat["messages"], clear=False)

    def _render_chat_history(self, messages: list[dict[str, Any]], clear: bool = True) -> None:
        log = self._query_one_log("chat-log")
        if clear:
            log.clear()
        for m in messages:
            role = m.get("role")
            content = m.get("content") or ""
            if role == "user":
                log.write(f"[bold cyan]you:[/] {content}")
            elif role == "assistant":
                tool_calls = m.get("tool_calls") or []
                if content:
                    log.write(f"[green]nelke:[/] {content}")
                elif tool_calls:
                    names = ", ".join(
                        (t.get("function") or {}).get("name", "?") for t in tool_calls
                    )
                    log.write(f"[dim]nelke: → tools: {names}[/]")
                else:
                    log.write(f"[green]nelke:[/] {content or '(no answer)'}")
            elif role == "tool":
                log.write(f"[dim]  ↳ tool: {content[:120]}[/]")
        if not messages:
            log.write("[dim]no messages yet — say hi.[/]")

    @on(Input.Submitted, "#chat-input")
    def _on_chat_submit(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        if not text:
            return
        event.input.value = ""
        if not self._active_chat:
            self._active_chat = services.create_chat(self.state.settings, frontend="tui")
            self._refresh_chats(select_id=self._active_chat)
        self._query_one_log("chat-log").write(f"[bold cyan]you:[/] {text}")
        self._run_chat(text)

    @work(exclusive=True, name="chat")
    async def _run_chat(self, text: str) -> None:
        sink = StreamSink()
        log = self._query_one_log("chat-log")
        usage_widget = self.query_one("#chat-usage", Static)
        chat_id = self._active_chat
        if chat_id is None:
            chat_id = services.create_chat(self.state.settings, frontend="tui")
            self._active_chat = chat_id
        callbacks = build_tui_callbacks(sink)

        def on_usage(usage: dict[str, Any]) -> None:
            sink.on_usage(usage)
            self._ui(usage_widget.update, _usage_text(sink.usage_total))

        callbacks.on_usage = on_usage
        try:
            result, _chat_id, _msg_id = await services.run_chat_turn(
                text, self.state.settings, self.state.profile, chat_id,
                frontend_name="tui",
                callbacks=callbacks,
                repo=self.state.repo_path,
                llm_factory=self.state.llm_factory or services._llm_factory_default,
            )
        except Exception as exc:  # noqa: BLE001
            self._ui(log.write, f"[bold red]error:[/] {exc}")
            return
        # drain the sink onto the log
        self._ui(self._render_sink, log, sink)
        self._ui(usage_widget.update, _usage_text(sink.usage_total))
        self._ui(self._refresh_chats)

    def _render_sink(self, log: RichLog, sink: StreamSink) -> None:
        for tool in sink.tools:
            log.write(f"[dim cyan] -> {tool}[/]")
        for res in sink.results[-4:]:
            log.write(f"[dim]{res}[/]")
        if sink.answer:
            log.write(f"[green]nelke:[/] {sink.answer}")

    # ---- Improve ------------------------------------------------------------
    @on(Input.Submitted, "#improve-input")
    def _on_improve_submit(self, event: Input.Submitted) -> None:
        objective = event.value.strip()
        if not objective:
            return
        event.input.value = ""
        self._query_one_log("improve-log").write(f"[bold cyan]objective:[/] {objective}")
        self._run_improve(objective)

    @work(exclusive=True, name="improve")
    async def _run_improve(self, objective: str) -> None:
        log = self._query_one_log("improve-log")
        progress = self.query_one("#improve-progress", ProgressBar)
        status = self.query_one("#improve-status", Static)
        usage_widget = self.query_one("#improve-usage", Static)
        cycle_usage: dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "calls": 0}
        self._ui(progress.update, progress=0)
        self._ui(usage_widget.update, "tokens: 0")

        def on_event(event: Any) -> None:
            if event.kind == "usage":
                data = event.data or {}
                cycle_usage["total_tokens"] += int(data.get("total_tokens", 0) or 0)
                cycle_usage["calls"] += 1
                self._ui(usage_widget.update, _usage_text(cycle_usage))
                return
            self._ui(log.write, f"[dim]{event.kind}:[/] {event.message}")
            if event.kind in {"commit", "gate"} and event.step is not None and event.data.get("total_steps"):
                pct = min(100, int(event.step / event.data["total_steps"] * 100))
                self._ui(progress.update, progress=pct)
                self._ui(status.update, f"step {event.step}/{event.data['total_steps']}")
            if event.kind == "agent_tool":
                tool = event.data.get("tool", "")
                args = event.data.get("args") or {}
                args_txt = ", ".join(f"{k}={str(v)[:30]}" for k, v in list(args.items())[:2])
                self._ui(log.write, f"[bold cyan]  🔧 {tool}({args_txt})[/]")
                if "path" in args:
                    self._ui(log.write, f"[dim]  📄 {args['path']}[/]")
            elif event.kind == "agent_tool_result":
                self._ui(log.write, f"[dim]  ✔ {event.data.get('snippet', '')[:120]}[/]")
            elif event.kind == "cycle_start":
                self._ui(status.update, "running…")

        async def human_gate(req: Any) -> bool:
            # Look up the review row so the modal can show the diff.
            db = services.open_db(self.state.settings)
            human_reqs = [r for r in db.list_review_requests(cycle_id=req.cycle_id, open_only=True)
                          if r["kind"] == "human"]
            review: dict[str, Any] | None = None
            if human_reqs:
                review = services.get_review(self.state.settings, human_reqs[0]["id"],
                                             repo_path=self.state.repo_path)
            if review is None:
                review = {"objective": req.objective, "branch": req.branch,
                          "diff": req.diff, "verdict": "", "comments": "",
                          "status": "awaiting-human", "id": ""}
            self._ui(status.update, "awaiting human review")
            decision: bool = await self.push_screen_wait(ReviewModal(review))
            return decision

        try:
            result = await services.run_cycle(
                objective, self.state.settings, self.state.profile,
                on_event=on_event, human_approve=human_gate,
                repo_path=self.state.repo_path,
                llm_factory=self.state.llm_factory or services._llm_factory_default,
                governance=self.state.governance,
            )
        except Exception as exc:  # noqa: BLE001
            self._ui(log.write, f"[bold red]error:[/] {exc}")
            return
        self._ui(progress.update, 100 if result.merged else progress.progress)
        self._ui(status.update, result.status)
        self._ui(
            log.write,
            f"[bold]cycle {result.status}[/] branch={result.branch} steps={result.steps}",
        )
        self._ui(self._refresh_cycles)

    # ---- Cycle history ------------------------------------------------------
    def _refresh_cycles(self) -> None:
        try:
            cycles = services.list_cycles(self.state.settings)
        except Exception:  # noqa: BLE001 - history is best-effort
            return
        lv = self.query_one("#cycle-list", ListView)
        lv.clear()
        self._cycle_ids = []
        for c in cycles:
            self._cycle_ids.append(c["id"])
            objective = (c.get("objective") or "")[:40] or c["id"]
            lv.append(ListItem(Static(f"[{c['status']}] {objective}")))

    @on(ListView.Selected, "#cycle-list")
    def _on_cycle_select(self, event: ListView.Selected) -> None:
        idx = event.list_view.index
        if idx is None or not (0 <= idx < len(self._cycle_ids)):
            return
        detail = services.get_cycle_detail(self.state.settings, self._cycle_ids[idx])
        detail_log = self.query_one("#cycle-detail", RichLog)
        detail_log.clear()
        if detail is None:
            detail_log.write("[dim]cycle not found[/]")
            return
        detail_log.write(f"[bold]{detail['objective'] or '(no objective)'}[/]  [dim]{detail['id']}[/]")
        detail_log.write(
            f"status: {detail['status']}  ai={detail['ai_verdict']}  "
            f"human={detail['human_verdict']}  branch={detail['branch']}"
        )
        for step in detail["steps"]:
            detail_log.write(
                f"  [dim]step {step['step']}[/] {step['status']}  {str(step['summary'] or '')[:80]}"
            )
        for ev in detail["events"]:
            detail_log.write(f"[dim]{ev['kind']}:[/] {ev['message'] or ''}")

    # ---- Memory -------------------------------------------------------------
    def _refresh_memory(self) -> None:
        try:
            repo = self.state.repo_path or services.find_repo(self.state.settings)
            files = services.memory_overview(repo)
        except Exception:  # noqa: BLE001 - memory is best-effort
            return
        lv = self.query_one("#memory-list", ListView)
        lv.clear()
        for f in files:
            lv.append(ListItem(Static(f"{f['name']}  ({f['size']} B)")))

    @on(Input.Submitted, "#recall-input")
    def _on_recall(self, event: Input.Submitted) -> None:
        query = event.value.strip()
        if not query:
            return
        event.input.value = ""
        log = self._query_one_log("memory-detail")
        repo = self.state.repo_path or services.find_repo(self.state.settings)
        hits = services.recall_memory(repo, query)
        if not hits:
            log.write("[dim]no memory matches[/]")
            return
        for h in hits:
            log.write(f"[bold]{h.name}[/] (score {h.score})\n{h.snippet}\n")

    @on(ListView.Selected, "#memory-list")
    def _on_memory_select(self, event: ListView.Selected) -> None:
        text = str(event.item.children[0].render()) if event.item.children else ""
        name = text.split("  (")[0].strip()
        log = self._query_one_log("memory-detail")
        try:
            repo = self.state.repo_path or services.find_repo(self.state.settings)
            content = services.open_memory(repo).read(name)
        except Exception as exc:  # noqa: BLE001
            log.write(f"[red]{exc}[/]")
            return
        log.write(content)

    # ---- helpers ------------------------------------------------------------
    def _query_one_log(self, widget_id: str) -> RichLog:
        return self.query_one(f"#{widget_id}", RichLog)

    def action_cancel_work(self) -> None:
        workers = list(self.workers)
        for w in workers:
            w.cancel()
        log = self._query_one_log("chat-log")
        try:
            log.write("[bold yellow]cancelled[/]")
        except Exception:  # noqa: BLE001
            pass


def launch() -> None:
    """Run the TUI."""
    from nelke import __version__

    print(f"nelke v{__version__} — TUI", file=sys.stderr)
    NelkeTUI().run()
