"""Frontend-agnostic service layer: the wiring every adapter shares.

Each frontend (CLI, web, TUI, telegram) is a thin I/O adapter. The agent
session setup, the self-improvement cycle plumbing and the review-resolution
flow are identical across all of them, so they live here and take the
transport-specific bits (streaming callbacks, the human-gate callable, event
sink) as parameters. Nothing in this module prints or knows about a UI
library.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nelke.config import Settings
from nelke.core.agent import Agent, AgentResult
from nelke.core.db import Database, _now
from nelke.core.gitops import GitRepo
from nelke.core.governance import Governance
from nelke.core.llm import LLMClient
from nelke.core.memory import MemoryHit, MemoryStore

TokenHandler = Callable[[str], Any] | None
ToolHandler = Callable[[str, dict[str, Any]], Any] | None
ToolResultHandler = Callable[[str, dict[str, Any], str], Any] | None
EventHandler = Callable[[Any], Any] | None
HumanGate = Callable[[Any], bool | Awaitable[bool]] | None
LLMFactory = Callable[[str | None], LLMClient | Any]


@dataclass
class Callbacks:
    """Streaming callbacks handed verbatim to :class:`Agent`."""

    on_token: TokenHandler = None
    on_tool: ToolHandler = None
    on_tool_result: ToolResultHandler = None
    stream: bool = False


@dataclass
class ChatSession:
    """A wired-up agent session bound to a db row and a memory store."""

    agent: Agent
    db: Database
    session_id: str
    memory: MemoryStore


@dataclass
class ResolveResult:
    """Outcome of resolving a review request (shared by all frontends)."""

    request_id: str
    cycle_id: str
    branch: str
    status: str
    human_verdict: str
    error: str | None = None


def _llm_factory_default(profile: str | None) -> Any:
    from nelke.core.llm import build_llm

    return build_llm(profile)


def find_repo(settings: Settings | None = None) -> Path:
    """Locate the Nelke repo (``NELKE_REPO`` override, cwd, or the home checkout)."""
    import os

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


def open_db(settings: Settings | None = None) -> Database:
    settings = settings or Settings()
    db = Database(settings.db_path)
    db.migrate()
    return db


def open_memory(repo: Path) -> MemoryStore:
    return MemoryStore(repo / "memory")


def build_chat_session(
    settings: Settings,
    profile: str | None,
    *,
    frontend_name: str,
    callbacks: Callbacks | None = None,
    repo: Path | None = None,
    llm_factory: LLMFactory = _llm_factory_default,
) -> ChatSession:
    """Wire a normal-mode agent for a one-off or multi-turn conversation.

    Mirrors the CLI setup: a db session tagged ``frontend_name``, a memory
    index, a workspace under ``settings.workspaces_dir``, and an agent with
    the supplied streaming callbacks. The returned agent keeps its own
    message history, so call ``agent.run(text, reset=False)`` for continuity.
    """
    from nelke.core.agent import make_agent

    callbacks = callbacks or Callbacks()
    repo = repo or find_repo(settings)
    llm = llm_factory(profile)
    db = open_db(settings)
    session_id = db.create_session(frontend_name)
    memory = open_memory(repo)
    memory_index = memory.build_index(max_tokens=settings.index_max_tokens)
    workspace = settings.workspaces_dir / session_id
    workspace.mkdir(parents=True, exist_ok=True)

    def task_factory(tool_names: list[str] | None = None) -> Agent:
        return make_agent(
            workspace=workspace,
            llm=llm_factory(profile),
            name="subagent",
            system_prompt="You are a Nelke subagent solving a bounded subtask. Return a concise final answer.",
            memory=memory,
            memory_index=memory_index,
            include_web=True,
            include_shell=True,
            iteration_cap=settings.max_agent_iterations,
            code_timeout=settings.code_timeout,
            web_timeout=settings.web_timeout,
        )

    agent = make_agent(
        workspace=workspace,
        llm=llm,
        name="nelke",
        system_prompt="You are Nelke, a general-purpose agent. Work inside your workspace.",
        memory=memory,
        memory_index=memory_index,
        task_factory=task_factory,
        on_token=callbacks.on_token,
        on_tool=callbacks.on_tool,
        on_tool_result=callbacks.on_tool_result,
        stream=callbacks.stream,
        iteration_cap=settings.max_agent_iterations,
        code_timeout=settings.code_timeout,
        web_timeout=settings.web_timeout,
    )
    return ChatSession(agent=agent, db=db, session_id=session_id, memory=memory)


async def run_task(
    text: str,
    settings: Settings,
    profile: str | None,
    *,
    frontend_name: str,
    callbacks: Callbacks | None = None,
    reset: bool = True,
    repo: Path | None = None,
    llm_factory: LLMFactory = _llm_factory_default,
) -> tuple[AgentResult, str]:
    """Run a single user turn and persist usage; returns ``(result, session_id)``.

    Use ``reset=False`` to continue a multi-turn conversation within the same
    session — callers keep the :class:`ChatSession` for that; this helper is
    for the stateless one-shot case.
    """
    session = build_chat_session(
        settings, profile, frontend_name=frontend_name, callbacks=callbacks,
        repo=repo, llm_factory=llm_factory,
    )
    try:
        result = await session.agent.run(text, reset=reset)
    finally:
        session.db.end_session(session.session_id)
    if result.usage.get("total_tokens"):
        session.db.add_usage(result.usage, session_id=session.session_id)
    return result, session.session_id


async def run_cycle(
    objective: str,
    settings: Settings,
    profile: str | None,
    *,
    on_event: EventHandler = None,
    human_approve: HumanGate = None,
    repo_path: Path | None = None,
    llm_factory: LLMFactory = _llm_factory_default,
    governance: Any = None,
) -> Any:
    """Run a self-improvement cycle. Returns the :class:`CycleResult`.

    Raises ``RuntimeError`` if the repo path is not a git checkout. The
    ``human_approve`` callable is handed straight to :class:`CycleEngine`
    and may be sync or async (the engine awaits awaitables). ``governance``
    defaults to a real :class:`Governance`; tests pass a fake to avoid
    running lint/typecheck/tests subprocesses.
    """
    from nelke.core.cycle import CycleEngine

    repo_path = repo_path or find_repo(settings)
    if not (repo_path / ".git").exists():
        raise RuntimeError(f"{repo_path} is not a git repository; cannot run a cycle")
    git = GitRepo(repo_path)
    db = open_db(settings)
    gov = governance if governance is not None else Governance(git)
    engine = CycleEngine(
        git, db, gov, llm_factory(profile),
        on_event=on_event,
        human_approve=human_approve,
        max_steps=settings.max_cycle_steps,
        max_step_attempts=settings.max_step_attempts,
        max_review_rounds=settings.max_review_rounds,
    )
    return await engine.run(objective)


def list_open_reviews(settings: Settings | None = None) -> list[dict[str, Any]]:
    """Open human review requests, each joined with its cycle objective."""
    db = open_db(settings)
    out: list[dict[str, Any]] = []
    for row in db.list_review_requests(open_only=True):
        if row["kind"] != "human":
            continue
        cycle = db.get_cycle(row["cycle_id"])
        out.append(
            {
                "id": row["id"],
                "cycle_id": row["cycle_id"],
                "kind": row["kind"],
                "verdict": row["verdict"],
                "objective": cycle["objective"] if cycle else "",
                "branch": cycle["branch"] if cycle else "",
                "comments": row["comments"] or "",
            }
        )
    return out


def get_review(
    settings: Settings | None = None,
    request_id: str | None = None,
    *,
    repo_path: Path | None = None,
) -> dict[str, Any] | None:
    """A single review request (open or closed) by id-prefix match."""
    db = open_db(settings)
    request_id = (request_id or "").strip()
    if not request_id:
        return None
    rows = [r for r in db.list_review_requests(open_only=False)
            if r["id"] == request_id or r["id"].startswith(request_id)]
    if not rows:
        return None
    row = rows[0]
    cycle = db.get_cycle(row["cycle_id"]) if row["cycle_id"] else None
    git = GitRepo(repo_path or find_repo(settings))
    diff = ""
    if cycle:
        try:
            diff = git.diff("main", cycle["branch"])
        except Exception:  # noqa: BLE001 - branch may be gone; diff is best-effort UI
            diff = ""
    return {
        "id": row["id"],
        "cycle_id": row["cycle_id"],
        "kind": row["kind"],
        "verdict": row["verdict"],
        "objective": cycle["objective"] if cycle else "",
        "branch": cycle["branch"] if cycle else "",
        "status": cycle["status"] if cycle else "",
        "comments": row["comments"] or "",
        "diff": diff,
    }


def resolve_review(
    settings: Settings | None,
    request_id: str,
    decision: str,
    repo_path: Path | None = None,
) -> ResolveResult:
    """Resolve a human review request and apply the merge/reject effect.

    ``decision`` is ``"approved"`` or ``"rejected"``. On approval the cycle
    branch is merged into ``main`` ``--no-ff``; on rejection the branch is
    kept. The matching review_requests row is always marked resolved, so the
    first frontend to resolve wins. Returns a :class:`ResolveResult`
    describing the outcome; ``error`` is set if the merge failed.
    """
    settings = settings or Settings()
    repo_path = repo_path or find_repo(settings)
    db = open_db(settings)
    rows = [
        r for r in db.list_review_requests(open_only=False)
        if r["id"].startswith(request_id) or r["id"] == request_id
    ]
    if not rows:
        raise RuntimeError(f"review request not found: {request_id}")
    req = rows[0]
    cycle = db.get_cycle(req["cycle_id"])
    if cycle is None:
        raise RuntimeError(f"cycle not found for request {request_id}")
    db.resolve_review_request(req["id"], decision)

    if decision == "approved":
        git = GitRepo(repo_path)
        try:
            git.checkout("main")
            git.merge_no_ff(
                cycle["branch"],
                f"Merge branch {cycle['branch']!r} (cycle {cycle['id']})",
                "Co-authored-by: Nelke <nelke@local>",
            )
            db.update_cycle(cycle["id"], status="merged", human_verdict="approved", ended_at=_now())
            return ResolveResult(
                request_id=req["id"], cycle_id=cycle["id"], branch=cycle["branch"],
                status="merged", human_verdict="approved",
            )
        except Exception as exc:  # noqa: BLE001
            db.update_cycle(cycle["id"], status="merge-conflict", human_verdict="approved")
            return ResolveResult(
                request_id=req["id"], cycle_id=cycle["id"], branch=cycle["branch"],
                status="merge-conflict", human_verdict="approved", error=str(exc),
            )

    db.update_cycle(cycle["id"], status="rejected", human_verdict="rejected", ended_at=_now())
    return ResolveResult(
        request_id=req["id"], cycle_id=cycle["id"], branch=cycle["branch"],
        status="rejected", human_verdict="rejected",
    )


def memory_overview(repo: Path) -> list[dict[str, Any]]:
    """Memory files (name + size) for a browser view."""
    store = open_memory(repo)
    out: list[dict[str, Any]] = []
    for rel in store.files():
        path = store.memory_dir / rel
        size = path.stat().st_size if path.exists() else 0
        out.append({"name": rel.as_posix(), "size": size})
    return out


def recall_memory(repo: Path, query: str, top_k: int = 8) -> list[MemoryHit]:
    return open_memory(repo).recall(query, top_k)
