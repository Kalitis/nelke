"""CLI frontend (Typer + Rich): thin adapter over the Nelke core.

Implements chat, task, improve, review, memory, config, db and doctor handlers.
Web/TUI/Telegram are separate phases (3-5) and their launchers report that here.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import typer
from rich import box
from rich.console import Console, Group
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TaskID, TextColumn
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from nelke.config import ProfileError, Settings, get_profile, load_env_files, load_profiles
from nelke.core.llm import usage_cache_pct

for _stream in (sys.stdout, sys.stderr):
    _reconfigure = getattr(_stream, "reconfigure", None)
    if callable(_reconfigure):
        try:
            _reconfigure(encoding="utf-8", errors="replace")
        except (TypeError, ValueError):
            pass

# Load ~/.nelke/.env (and ./.) into os.environ before anything reads it, so
# secrets referenced by profiles via api_key_ref (e.g. OPENAI_API_KEY) are
# visible — pydantic-settings only loads NELKE_-prefixed fields on its own.
load_env_files()

console = Console()


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #
def find_repo() -> Path:
    override = os.environ.get("NELKE_REPO")
    if override:
        return Path(override).expanduser()
    here = Path.cwd()
    if (here / "src" / "nelke").exists() and (here / ".git").exists():
        return here
    default = Path.home() / "source" / "repos" / "nelke"
    if default.exists():
        return default
    return here


def open_settings() -> Settings:
    return Settings()


def open_db(settings: Settings | None = None):
    from nelke.core.db import Database

    settings = settings or open_settings()
    db = Database(settings.db_path)
    db.migrate()
    return db


def open_memory(repo: Path):
    from nelke.core.memory import MemoryStore

    return MemoryStore(repo / "memory")


def get_llm(profile: str | None = None):
    from nelke.core.llm import build_llm

    return build_llm(profile)


def _fatal(message: str) -> None:
    console.print(f"[bold red]{message}[/]")
    raise typer.Exit(1)


# --------------------------------------------------------------------------- #
# Stream helpers
# --------------------------------------------------------------------------- #
class AnswerStream:
    """Accumulates tokens and renders them live; also reports tool calls + results."""

    def __init__(self) -> None:
        self.buffer: list[str] = []
        self.tools: list[str] = []
        self.results: list[str] = []
        self.usage_total: dict[str, int] = {
            "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
            "cache_read_tokens": 0, "calls": 0,
        }
        self._live: Live | None = None

    def start(self) -> None:
        self._live = Live(console=console, refresh_per_second=12, transient=False)
        self._live.start()

    def _update(self) -> None:
        if self._live is not None:
            try:
                self._live.update(self._render())
            except Exception:  # noqa: BLE001 - a glyph/rendering issue must not kill the run
                pass

    def on_token(self, token: str) -> None:
        self.buffer.append(token)
        self._update()

    def on_tool(self, name: str, args: dict[str, Any]) -> None:
        self.tools.append(f"{name}({_fmt_args(args)})")
        self._update()

    def on_tool_result(self, name: str, args: dict[str, Any], result: str) -> None:
        snippet = " ".join(result.split())
        if len(snippet) > 160:
            snippet = snippet[:160] + "..."
        self.results.append(f"[dim]=> {name}: {snippet}[/]")
        self._update()

    def on_usage(self, usage: dict[str, Any]) -> None:
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            self.usage_total[key] += int(usage.get(key, 0) or 0)
        self.usage_total["cache_read_tokens"] += int(usage.get("cache_read_tokens", 0) or 0)
        self.usage_total["calls"] += 1
        self._update()

    def _render(self) -> Group:
        parts: list[Any] = []
        if self.usage_total.get("total_tokens"):
            usage = self.usage_total
            parts.append(
                Text(
                    f"tokens: {usage['total_tokens']}"
                    f" (cache {usage_cache_pct(usage)}% of prompt)"
                    f" ({usage['calls']} call{'s' if usage['calls'] != 1 else ''})",
                    style="dim",
                )
            )
        if self.tools:
            parts.append(Text(" -> " + " ".join(f"[cyan]{t}[/]" for t in self.tools), style="dim"))
        if self.results:
            parts.append(Group(*[Text(r) for r in self.results[-4:]]))
        current = "".join(self.buffer)
        if current:
            parts.append(Markdown(current))
        if not parts:
            parts.append(Text("thinking...", style="dim italic"))
        return Group(*parts)

    def finish(self) -> str:
        if self._live is not None:
            self._live.stop()
            self._live = None
        return "".join(self.buffer)


def _fmt_args(args: dict[str, Any]) -> str:
    items = []
    for k, v in list(args.items())[:2]:
        text = str(v)
        if len(text) > 60:
            text = text[:60] + "..."
        items.append(f"{k}={text}")
    return ", ".join(items)


# --------------------------------------------------------------------------- #
# Agent session (chat / task)
# --------------------------------------------------------------------------- #
def _workspace_for(settings: Settings, session_id: str) -> Path:
    ws = settings.workspaces_dir / session_id
    ws.mkdir(parents=True, exist_ok=True)
    return ws


def _build_session_agent(settings: Settings, profile: str | None, stream: AnswerStream | None,
                         on_degraded=None):
    from nelke.core.agent import make_agent

    llm = get_llm(profile)
    db = open_db(settings)
    session_id = db.create_session("cli")
    repo = find_repo()
    memory = open_memory(repo)
    memory_index = memory.build_index(max_tokens=settings.index_max_tokens)
    workspace = _workspace_for(settings, session_id)

    def task_factory(tool_names: list[str] | None = None):
        return make_agent(
            workspace=workspace,
            llm=get_llm(profile),
            name="subagent",
            system_prompt="You are a Nelke subagent solving a bounded subtask. Return a concise final answer.",
            memory=memory,
            memory_index=memory_index,
            include_web=True,
            include_shell=True,
            iteration_cap=settings.max_agent_iterations,
            code_timeout=settings.code_timeout,
            web_timeout=settings.web_timeout,
            temperature=settings.agent_temperature,
        )

    # Persist each LLM call's usage as it happens, then forward to the stream so
    # the live panel and the DB both stay in sync in real time.
    user_on_usage = stream.on_usage if stream else None

    def on_usage(usage: dict[str, Any]) -> None:
        try:
            if usage.get("total_tokens"):
                db.add_usage(usage, session_id=session_id)
        except Exception:  # noqa: BLE001 - persistence must never break the run
            pass
        if user_on_usage is not None:
            try:
                user_on_usage(usage)
            except Exception:  # noqa: BLE001 - rendering must never break the run
                pass

    agent = make_agent(
        workspace=workspace,
        llm=llm,
        name="nelke",
        system_prompt="You are Nelke, a general-purpose agent. Work inside your workspace.",
        memory=memory,
        memory_index=memory_index,
        task_factory=task_factory,
        on_token=stream.on_token if stream else None,
        on_tool=stream.on_tool if stream else None,
        on_tool_result=stream.on_tool_result if stream else None,
        on_usage=on_usage,
        on_degraded=on_degraded,
        stream=bool(stream),
        iteration_cap=settings.max_agent_iterations,
        code_timeout=settings.code_timeout,
        web_timeout=settings.web_timeout,
        db=db,
        temperature=settings.agent_temperature,
        plan_first=settings.plan_first,
    )
    return agent, db, session_id, memory


def run_task(text: str, *, profile: str | None = None, interactive: bool = False) -> None:
    settings = open_settings()
    use_live = bool(interactive and not os.environ.get("NELKE_NO_STREAM"))
    stream: AnswerStream | None = AnswerStream() if use_live else None
    if stream is not None:
        stream.start()

    def on_degraded(report) -> None:
        console.print()
        console.print(Panel(
            "[yellow]Nelke didn't fully complete this task:[/]\n"
            f"{report.describe()}\n\n"
            f'[dim]suggested: [bold cyan]nelke improve "{report.suggested_objective}"[/][/]',
            title="Self-improvement opportunity", border_style="yellow",
        ))

    agent, db, session_id, memory = _build_session_agent(
        settings, profile, stream, on_degraded=on_degraded
    )
    try:
        result = asyncio.run(agent.run(text))
    except Exception as exc:  # noqa: BLE001
        db.end_session(session_id)
        _fatal(f"agent failed: {exc}")
    db.end_session(session_id)

    if interactive:
        if stream is not None:
            stream.finish()
            console.print()
            console.print(_usage_line(result.usage))
        else:
            console.print()
            console.print(Panel(Markdown(result.answer or "*(no answer)*"), title="Nelke", border_style="green"))
    else:
        console.print(result.answer or "(no answer)")
        if result.usage.get("total_tokens"):
            console.print(_usage_line(result.usage))


def _usage_line(usage: dict[str, int]) -> str:
    calls = int(usage.get("calls", 0))
    pct = int(usage.get("cache_read_pct") or usage_cache_pct(usage))
    return (
        f"[dim]tokens: {usage.get('total_tokens', 0)} "
        f"(prompt {usage.get('prompt_tokens', 0)} + completion {usage.get('completion_tokens', 0)}"
        f", cache {pct}%) - "
        f"{calls} LLM call{'s' if calls != 1 else ''}[/]"
    )


# --------------------------------------------------------------------------- #
# Improve (self-improvement cycle)
# --------------------------------------------------------------------------- #
class ImproveStream:
    """Streams cycle events into a live Rich panel with a collapsible gate block.

    Also renders a live progress bar (steps vs total) and streams the cycle
    worker's tool calls so the user sees exactly what is being edited.
    """

    _LABELS = {
        "cycle_start": "[bold]Cycle started[/]",
        "step_start": "[cyan]step[/]",
        "gate": "[yellow]gate[/]",
        "commit": "[green]commit[/]",
        "boot_check_failed": "[bold red]boot-check failed[/]",
        "step_ok": "[green]step ok[/]",
        "propose_complete": "[bold magenta]proposing completion[/]",
        "ai_review": "[bold blue]AI review[/]",
        "review_feedback": "[magenta]reviewer feedback[/]",
        "awaiting_human": "[bold yellow]awaiting human review[/]",
        "human_pending": "[dim]no human gate — branch left for review[/]",
        "human_rejected": "[bold red]human rejected[/]",
        "merged": "[bold green]merged[/]",
        "cycle_error": "[bold red]error[/]",
        "deps_synced": "[cyan]deps synced[/]",
        "deps_failed": "[bold red]deps sync failed[/]",
        "idle": "[dim]idle[/]",
    }
    _STOP_KINDS = {"awaiting_human", "human_pending", "merged", "human_rejected", "cycle_error"}

    def __init__(self, objective: str, total_steps: int = 0) -> None:
        self.objective = objective
        self.rows: list[tuple[str, str]] = []
        self.gate_block: str = ""
        self.tool_lines: list[str] = []
        self.usage_total: dict[str, int] = {
            "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
            "cache_read_tokens": 0, "calls": 0,
        }
        self._live: Live | None = None
        self._progress = Progress(TextColumn("[progress.description]{task.description}"),
                                  BarColumn(), TextColumn("[progress.percentage]{task.percentage:>3.0f}%"))
        self._task_id: TaskID | None = None
        self._total_steps = total_steps

    def start(self) -> None:
        self._live = Live(console=console, refresh_per_second=10, transient=False)
        self._live.start()
        if self._total_steps:
            self._task_id = self._progress.add_task("improve", total=self._total_steps)
        self._update()

    def __call__(self, event: Any) -> None:
        if event.kind == "usage":
            payload = event.data or {}
            for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                self.usage_total[key] += int(payload.get(key, 0) or 0)
            self.usage_total["cache_read_tokens"] += int(payload.get("cache_read_tokens", 0) or 0)
            self.usage_total["calls"] += 1
            self._update()
            return
        label = self._LABELS.get(event.kind, event.kind)
        self.rows.append((label, str(event.message or "")))
        if event.kind == "gate":
            self.gate_block = event.message
        if event.kind in {"agent_tool", "agent_tool_result"}:
            payload = event.data or {}
            tool = payload.get("tool", "")
            if event.kind == "agent_tool":
                args = payload.get("args") or {}
                args_txt = ", ".join(f"{k}={str(v)[:30]}" for k, v in list(args.items())[:2])
                self.tool_lines.append(f"🔧 {tool}({args_txt})")
                if "path" in args:
                    self.tool_lines.append(f"   📄 {args['path']}")
            else:
                self.tool_lines.append(f"   ✔ {payload.get('snippet', '')[:120]}")
            if len(self.tool_lines) > 8:
                self.tool_lines = self.tool_lines[-8:]
        prog = event.progress
        if prog and self._task_id is not None:
            self._progress.update(self._task_id, completed=prog[0], total=prog[1])
        if event.kind in self._STOP_KINDS:
            self.stop()
            return
        self._update()

    def stop(self) -> None:
        if self._live is not None:
            try:
                self._live.stop()
            except Exception:  # noqa: BLE001 - rendering must never crash the cycle
                pass
            self._live = None

    def _update(self) -> None:
        if self._live is None:
            return
        try:
            self._live.update(self._render())
        except Exception:  # noqa: BLE001
            pass

    def _render(self) -> Group:
        parts: list[Any] = [Text(f"objective: {self.objective[:80]}", style="bold")]
        if self._total_steps:
            parts.append(self._progress)
        if self.usage_total.get("total_tokens"):
            usage = self.usage_total
            parts.append(
                Text(
                    f"tokens: {usage['total_tokens']} (prompt {usage['prompt_tokens']} "
                    f"+ completion {usage['completion_tokens']}, cache {usage_cache_pct(usage)}%) · "
                    f"{usage['calls']} call{'s' if usage['calls'] != 1 else ''}",
                    style="dim",
                )
            )
        if self.tool_lines:
            parts.append(Group(*[Text(line, overflow="ellipsis") for line in self.tool_lines]))
        recent = self.rows[-14:]
        if recent:
            parts.append(Group(*[Text(f"{label} {msg}", overflow="ellipsis") for label, msg in recent]))
        else:
            parts.append(Text("preparing cycle...", style="dim italic"))
        if self.gate_block:
            gate_lines = self.gate_block.splitlines()
            shown = gate_lines[:24]
            if len(gate_lines) > 24:
                shown.append(f"... {len(gate_lines) - 24} more lines")
            parts.append(
                Panel("\n".join(shown) or "(no gate output)", title="gate",
                      border_style="yellow")
            )
        return Group(*parts)


def improve(objective: str, *, yes: bool = False, profile: str | None = None) -> None:
    from nelke.core.cycle import CycleEngine, HumanReviewRequest
    from nelke.core.gitops import GitRepo
    from nelke.core.governance import Governance

    settings = open_settings()
    repo_path = find_repo()
    if not (repo_path / ".git").exists():
        _fatal(f"{repo_path} is not a git repository; cannot run a cycle")
    git = GitRepo(repo_path)
    db = open_db(settings)
    gov = Governance(git)

    stream = ImproveStream(objective, total_steps=settings.max_cycle_steps)
    stream.start()
    drain = stream

    def human_gate(human: HumanReviewRequest) -> bool:
        if yes:
            console.print("[dim]--yes provided; auto-approving human gate[/]")
            return True
        console.print(Rule("[bold]Proposed changes for merge to main[/]"))
        console.print(human.diff[:4000])
        console.print(Rule())
        return bool(typer.confirm("Approve and merge into main?", default=False))

    engine = CycleEngine(
        git, db, gov, get_llm(profile),
        on_event=drain,
        human_approve=human_gate,
        max_steps=settings.max_cycle_steps,
        max_step_attempts=settings.max_step_attempts,
        max_review_rounds=settings.max_review_rounds,
        agent_temperature=settings.agent_temperature,
    )
    try:
        result = asyncio.run(engine.run(objective))
    except Exception as exc:  # noqa: BLE001
        stream.stop()
        _fatal(f"cycle failed: {exc}")
    stream.stop()
    console.print()
    usage = db.usage_totals(cycle_id=result.cycle_id)
    usage_text = (
        f"tokens: {usage['total_tokens']} (prompt {usage['prompt_tokens']} + "
        f"completion {usage['completion_tokens']}, cache {usage.get('cache_read_pct', 0)}%)  -  "
        f"{usage['calls']} LLM calls"
    )
    console.print(Panel(
        f"cycle: [bold]{result.cycle_id}[/]\nbranch: {result.branch}\nstatus: [bold]{result.status}[/]\n"
        f"steps: {result.steps}\nAI: {result.ai_verdict} / human: {result.human_verdict}\n[dim]{usage_text}[/]",
        title="Cycle result", border_style="blue" if result.merged else "yellow",
    ))
    if result.message:
        console.print(result.message[:1000])


# --------------------------------------------------------------------------- #
# Review requests
# --------------------------------------------------------------------------- #
def _resolve_review(settings: Settings, request_id: str, decision: str, repo_path: Path) -> None:
    from nelke.core.cycle import merge_cycle_branch
    from nelke.core.gitops import GitRepo

    db = open_db(settings)
    rows = [
        r for r in db.list_review_requests(open_only=False)
        if r["id"].startswith(request_id) or r["id"] == request_id
    ]
    if not rows:
        _fatal(f"review request not found: {request_id}")
    req = rows[0]
    cycle = db.get_cycle(req["cycle_id"])
    if cycle is None:
        _fatal(f"cycle not found for request {request_id}")
    db.resolve_review_request(req["id"], decision)
    if decision == "approved":
        git = GitRepo(repo_path)
        try:
            merge_cycle_branch(git, cycle["branch"], cycle_id=cycle["id"])
            db.update_cycle(cycle["id"], status="merged", human_verdict="approved", ended_at=_now())
            console.print(f"[green]Approved and merged {cycle['branch']} into main[/]")
        except Exception as exc:  # noqa: BLE001
            db.update_cycle(cycle["id"], status="merge-conflict", human_verdict="approved")
            _fatal(f"merge failed: {exc}")
    else:
        db.update_cycle(cycle["id"], status="rejected", human_verdict="rejected", ended_at=_now())
        console.print(f"[yellow]Rejected; branch {cycle['branch']} kept[/]")


def review_list(settings: Settings | None = None) -> None:
    settings = settings or open_settings()
    db = open_db(settings)
    open_reqs = db.list_review_requests(open_only=True)
    table = Table(title="Open review requests", box=box.SIMPLE)
    table.add_column("id")
    table.add_column("cycle id")
    table.add_column("kind")
    table.add_column("verdict")
    for r in open_reqs:
        table.add_row(r["id"], r["cycle_id"], r["kind"], r["verdict"])
    console.print(table)
    if not open_reqs:
        console.print("[dim]no open review requests[/]")
    all_reqs = db.list_review_requests(open_only=False)
    recent = [r for r in all_reqs if r["verdict"] != "pending"]
    for r in recent[-8:]:
        cycle = db.get_cycle(r["cycle_id"]) if r["cycle_id"] else None
        obj = (cycle["objective"][:60] if cycle else r["cycle_id"])
        console.print(f"[dim]{r['id']} {r['kind']}:{r['verdict']} — {obj}[/]")


def review_approve(request_id: str) -> None:
    _resolve_review(open_settings(), request_id, "approved", find_repo())


def review_reject(request_id: str) -> None:
    _resolve_review(open_settings(), request_id, "rejected", find_repo())


def _now():
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# --------------------------------------------------------------------------- #
# Memory
# --------------------------------------------------------------------------- #
def memory_list() -> None:
    store = open_memory(find_repo())
    files = store.files()
    table = Table(title="Memory files", box=box.SIMPLE)
    table.add_column("file")
    table.add_column("size")
    for f in files:
        size = (find_repo() / "memory" / f).stat().st_size
        table.add_row(f.as_posix(), str(size))
    console.print(table)
    if not files:
        console.print("[dim]no memory files yet — use `nelke memory edit <name>`[/]")


def memory_show(name: str) -> None:
    from nelke.core.tools.base import ToolError

    store = open_memory(find_repo())
    try:
        content = store.read(name)
    except (FileNotFoundError, ToolError) as exc:
        _fatal(str(exc))
    console.print(Markdown(content))


def memory_edit(name: str) -> None:
    from nelke.core.tools.base import ToolError

    store = open_memory(find_repo())
    target = (find_repo() / "memory" / name).resolve()
    try:
        _ = store.read(name)
    except FileNotFoundError:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(f"# {name}\n\ntags: \n\n", encoding="utf-8")
    except ToolError as exc:
        _fatal(str(exc))
    editor = os.environ.get("NELKE_EDITOR") or os.environ.get("EDITOR") or "code"
    if editor == "code":
        subprocess.run(["code", "--wait", str(target)], check=False)
    else:
        subprocess.run([editor, str(target)], check=False)
    store.build_index()
    console.print(f"[green]saved {target.relative_to(find_repo())}; INDEX.md rebuilt[/]")


def memory_recall(query: str, top_k: int = 8) -> None:
    store = open_memory(find_repo())
    hits = store.recall(query, top_k)
    if not hits:
        console.print("[dim]no memory matches[/]")
        return
    for h in hits:
        console.print(Panel(h.snippet, title=f"[bold]{h.name}[/] (score {h.score})", border_style="cyan"))


# --------------------------------------------------------------------------- #
# Config / db / doctor
# --------------------------------------------------------------------------- #
def config_show() -> None:
    settings = open_settings()
    table = Table(title="Nelke settings", box=box.SIMPLE)
    table.add_column("key")
    table.add_column("value")
    rows = [
        ("nelke_home", str(settings.nelke_home)),
        ("db_path", str(settings.db_path)),
        ("default_profile", settings.default_profile),
        ("max_agent_iterations", str(settings.max_agent_iterations)),
        ("max_cycle_steps", str(settings.max_cycle_steps)),
        ("code_timeout", str(settings.code_timeout)),
        ("plan_first", str(settings.plan_first)),
    ]
    for k, v in rows:
        table.add_row(k, v)
    console.print(table)
    profiles = load_profiles()
    if profiles:
        ptable = Table(title="Profiles", box=box.SIMPLE)
        ptable.add_column("name")
        ptable.add_column("base_url")
        ptable.add_column("model")
        ptable.add_column("key")
        for name, p in profiles.items():
            api_key = p.resolved_api_key()
            shown = "***" if api_key and api_key not in ("not-needed", "") else str(api_key)
            ptable.add_row(name, p.base_url, p.model, shown)
        console.print(ptable)
    else:
        console.print("[yellow]no profiles — run `nelke config init`[/]")


def config_init() -> None:
    settings = open_settings()
    settings.nelke_home.mkdir(parents=True, exist_ok=True)
    examples_dir = find_repo()
    src_toml = examples_dir / "config.example.toml"
    src_env = examples_dir / ".env.example"
    dst_toml = settings.config_file
    dst_env = settings.nelke_home / ".env"
    for src, dst, _label in ((src_toml, dst_toml, "config.toml"), (src_env, dst_env, ".env")):
        if dst.exists():
            console.print(f"[yellow]already exists: {dst} (skipped)[/]")
        elif src.exists():
            shutil.copyfile(src, dst)
            console.print(f"[green]wrote {dst} from {src.name}[/]")
        else:
            console.print(f"[yellow]template not found: {src} (skipped)[/]")
    Console().print("Edit profiles and secrets, then run `nelke doctor`.")


def db_status() -> None:
    db = open_db()
    counts = db.status()
    table = Table(title="Database", box=box.SIMPLE)
    table.add_column("table")
    table.add_column("rows")
    for name, count in counts.items():
        table.add_row(name, str(count))
    console.print(table)


def db_cleanup() -> None:
    """Mark abandoned ``running`` cycles as ``stuck``; print what changed."""
    from nelke.core.services import reconcile_stale_cycles

    settings = open_settings()
    marked = reconcile_stale_cycles(settings)
    table = Table(title="Reconciled stale cycles", box=box.SIMPLE)
    table.add_column("cycle")
    table.add_column("branch")
    table.add_column("reason")
    if not marked:
        console.print("[green]No stale running cycles found.[/]")
        return
    for m in marked:
        table.add_row(m["id"], m["branch"], m["reason"])
    console.print(table)
    console.print(f"[yellow]marked {len(marked)} cycle(s) as stuck[/]")


def doctor() -> None:
    settings = open_settings()
    console.print(Rule("[bold]Nelke doctor[/]"))
    from nelke import __version__

    console.print(f"nelke: {__version__}")
    console.print(f"python: {sys.version.split()[0]}")
    for tool in ("git", "uv"):
        console.print(f"{tool}: {shutil.which(tool) or 'NOT FOUND'}")
    console.print(f"repo: {find_repo()}")
    console.print(f"home: {settings.nelke_home}")
    console.print(f"db: {settings.db_path} (exists: {settings.db_path.exists()})")
    settings.nelke_home.mkdir(parents=True, exist_ok=True)
    try:
        open_db(settings)
        console.print("[green]database OK (migrated)[/]")
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]database error: {exc}[/]")
    try:
        p = get_profile()
        key = p.resolved_api_key()
        console.print(f"profile: {p.name} @ {p.base_url} model={p.model}")
        console.print("[green]api key configured[/]" if key else "[yellow]api key not set (local models ok)[/]")
    except ProfileError as exc:
        console.print(f"[yellow]{exc}[/]")
    console.print("boot_check...")
    try:
        import nelke

        nelke.boot_check()
        console.print("[green]boot_check passed[/]")
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]boot_check FAILED: {exc}[/]")


def launch(frontend: str) -> None:
    console.print(
        f"[yellow]`nelke {frontend}` is planned for phase "
        f"{'3 (web)' if frontend == 'web' else '4 (tui)' if frontend == 'tui' else '5 (telegram)'} "
        f"of the v0 plan and is not implemented yet.[/]"
    )
    raise typer.Exit(0)
