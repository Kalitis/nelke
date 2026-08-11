"""Frontend-agnostic service layer: the wiring every adapter shares.

Each frontend (CLI, web, TUI, telegram) is a thin I/O adapter. The agent
session setup, the self-improvement cycle plumbing and the review-resolution
flow are identical across all of them, so they live here and take the
transport-specific bits (streaming callbacks, the human-gate callable, event
sink) as parameters. Nothing in this module prints or knows about a UI
library.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace
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
UsageHandler = Callable[[dict[str, Any]], Any] | None
EventHandler = Callable[[Any], Any] | None
HumanGate = Callable[[Any], bool | Awaitable[bool]] | None
LLMFactory = Callable[[str | None], LLMClient | Any]


@dataclass
class Callbacks:
    """Streaming callbacks handed verbatim to :class:`Agent`."""

    on_token: TokenHandler = None
    on_tool: ToolHandler = None
    on_tool_result: ToolResultHandler = None
    on_usage: UsageHandler = None
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


def _persist_usage(
    db: Database,
    key: str,
    user_handler: UsageHandler,
    *,
    cycle_id: str | None = None,
) -> UsageHandler:
    """Return an ``on_usage`` handler that persists each LLM call's usage as
    soon as it is available, then forwards it to the frontend handler.

    ``key`` is the ``session_id`` (chat/task) or ``cycle_id`` (improve) used
    to tag the ``usage_events`` row. Persistence is best-effort: a database
    hiccup must never break the conversation or the cycle.
    """

    def _handler(usage: dict[str, Any]) -> None:
        try:
            if usage.get("total_tokens"):
                if cycle_id:
                    db.add_usage(usage, cycle_id=cycle_id)
                else:
                    db.add_usage(usage, session_id=key)
        except Exception:  # noqa: BLE001 - persistence must never break a run
            pass
        if user_handler is not None:
            try:
                user_handler(usage)
            except Exception:  # noqa: BLE001 - UI updates must never break a run
                pass

    return _handler


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
    session_id: str | None = None,
    title: str | None = None,
    chat_memory: bool = False,
) -> ChatSession:
    """Wire a normal-mode agent for a one-off or multi-turn conversation.

    Mirrors the CLI setup: a db session tagged ``frontend_name``, a memory
    index, a workspace under ``settings.workspaces_dir``, and an agent with
    the supplied streaming callbacks. The returned agent keeps its own
    message history, so call ``agent.run(text, reset=False)`` for continuity.

    ``session_id``: when given and present in the db, the existing chat is
    resumed — its persisted history (user/assistant/tool messages) is reloaded
    into the agent so the model continues the same conversation, and (with
    ``chat_memory=True``) its per-chat memory store is used.
    """
    from nelke.core.agent import make_agent

    callbacks = callbacks or Callbacks()
    repo = repo or find_repo(settings)
    llm = llm_factory(profile)
    db = open_db(settings)
    existing = session_id is not None and db.get_session(session_id) is not None
    if existing:
        session_id = str(session_id)
    else:
        session_id = db.create_session(frontend_name, meta={"title": title} if title else {})
    memory = open_chat_memory(repo, session_id) if chat_memory else open_memory(repo)
    memory_index = memory.build_index(max_tokens=settings.index_max_tokens)
    workspace = settings.workspaces_dir / session_id
    workspace.mkdir(parents=True, exist_ok=True)

    user_on_usage = callbacks.on_usage
    callbacks = replace(callbacks, on_usage=_persist_usage(db, session_id, user_on_usage))

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
            temperature=settings.agent_temperature,
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
        on_usage=callbacks.on_usage,
        stream=callbacks.stream,
        iteration_cap=settings.max_agent_iterations,
        code_timeout=settings.code_timeout,
        web_timeout=settings.web_timeout,
        db=db,
        temperature=settings.agent_temperature,
    )
    if existing:
        history = db.list_messages(session_id)
        if history:
            agent._messages = [{"role": "system", "content": agent.system_content()}] + _rows_to_messages(history)
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
    on_token: Any = None,
    on_usage: UsageHandler = None,
) -> Any:
    """Run a self-improvement cycle. Returns the :class:`CycleResult`.

    Raises ``RuntimeError`` if the repo path is not a git checkout. The
    ``human_approve`` callable is handed straight to :class:`CycleEngine`
    and may be sync or async (the engine awaits awaitables). ``governance``
    defaults to a real :class:`Governance`; tests pass a fake to avoid
    running lint/typecheck/tests subprocesses. ``on_token`` streams raw
    cycle-worker tokens if provided; ``on_usage`` receives each LLM call's
    usage in real time (persisted to the cycle's usage rows by the engine).
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
        on_token=on_token,
        on_usage=on_usage,
        agent_temperature=settings.agent_temperature,
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


# --------------------------------------------------------------------------- #
# Chats: multiple named conversations, each with its own history + memory
# --------------------------------------------------------------------------- #
_DEFAULT_CHAT_TITLE = "New chat"


def chat_memory_dir(repo: Path, session_id: str) -> Path:
    """Per-chat memory lives under ``memory/chats/<session_id>/``."""
    return repo / "memory" / "chats" / session_id


def open_chat_memory(repo: Path, session_id: str) -> MemoryStore:
    return MemoryStore(chat_memory_dir(repo, session_id))


def list_chat_memory(
    settings: Settings | None = None, session_id: str | None = None, repo: Path | None = None,
) -> list[dict[str, Any]]:
    """Memory files (name + size) scoped to one chat."""
    if not session_id:
        return []
    repo = repo or find_repo(settings)
    store = open_chat_memory(repo, session_id)
    out: list[dict[str, Any]] = []
    for rel in store.files():
        path = store.memory_dir / rel
        size = path.stat().st_size if path.exists() else 0
        out.append({"name": rel.as_posix(), "size": size})
    return out


def create_chat(
    settings: Settings | None = None,
    *,
    title: str | None = None,
    frontend: str = "web",
) -> str:
    """Create a new (empty) chat session; returns its id."""
    db = open_db(settings)
    return db.create_session(frontend, meta={"title": title} if title else {})


def list_chats(
    settings: Settings | None = None,
    frontend: str = "web",
    limit: int = 200,
) -> list[dict[str, Any]]:
    """Recent chat sessions for a frontend, newest activity first."""
    db = open_db(settings)
    out: list[dict[str, Any]] = []
    for row in db.list_sessions(frontend=frontend, limit=limit):
        meta = _session_meta(row)
        title = meta.get("title") or _title_from_first_user(db, row["id"])
        out.append(
            {
                "id": row["id"],
                "title": title,
                "frontend": row["frontend"],
                "started_at": row["started_at"],
                "ended_at": row["ended_at"],
                "message_count": int(row["message_count"] or 0),
                "last_message_at": row["last_message_at"],
            }
        )
    return out


def get_chat(
    settings: Settings | None = None,
    session_id: str | None = None,
    *,
    repo: Path | None = None,
) -> dict[str, Any] | None:
    """A single chat: metadata + full message history + per-chat memory."""
    if not session_id:
        return None
    db = open_db(settings)
    row = db.get_session(session_id)
    if row is None:
        return None
    messages = get_chat_messages(settings, session_id)
    title = _session_meta(row).get("title") or _title_from_messages(messages)
    return {
        "id": row["id"],
        "title": title,
        "frontend": row["frontend"],
        "started_at": row["started_at"],
        "ended_at": row["ended_at"],
        "message_count": len(messages),
        "messages": messages,
        "memory": list_chat_memory(settings, session_id, repo=repo),
    }


def get_chat_messages(
    settings: Settings | None = None, session_id: str | None = None
) -> list[dict[str, Any]]:
    """The persisted transcript of a chat (role, content, tool metadata)."""
    if not session_id:
        return []
    db = open_db(settings)
    out: list[dict[str, Any]] = []
    for row in db.list_messages(session_id):
        out.append(
            {
                "id": row["id"],
                "role": row["role"],
                "content": row["content"] or "",
                "tool_calls": _safe_json(row["tool_calls"] or "[]", []),
                "tool_call_id": row["tool_call_id"] or "",
            }
        )
    return out


def rename_chat(
    settings: Settings | None = None, session_id: str | None = None, title: str | None = None
) -> bool:
    if not session_id or not title:
        return False
    db = open_db(settings)
    if db.get_session(session_id) is None:
        return False
    db.update_session_meta(session_id, title=title.strip())
    return True


def delete_chat(settings: Settings | None = None, session_id: str | None = None) -> bool:
    if not session_id:
        return False
    db = open_db(settings)
    if db.get_session(session_id) is None:
        return False
    db.delete_session(session_id)
    return True


async def run_chat_turn(
    text: str,
    settings: Settings,
    profile: str | None,
    chat_id: str,
    *,
    frontend_name: str,
    callbacks: Callbacks | None = None,
    repo: Path | None = None,
    llm_factory: LLMFactory = _llm_factory_default,
) -> tuple[AgentResult, str]:
    """Run one user turn inside an existing chat, persisting the new transcript.

    The chat's history (including tool messages) is reloaded into the agent so
    it continues the same conversation; every new user/assistant/tool message
    is written to ``messages`` so the transcript survives a process restart.
    """
    callbacks = callbacks or Callbacks()
    session = build_chat_session(
        settings, profile, frontend_name=frontend_name, callbacks=callbacks,
        repo=repo, llm_factory=llm_factory,
        session_id=chat_id, chat_memory=True,
    )
    before = len(session.agent._messages)
    try:
        result = await session.agent.run(text, reset=False)
    finally:
        session.db.end_session(session.session_id)
    _persist_new_messages(session.db, session.session_id, session.agent._messages[before:])
    return result, session.session_id


def _title_from_first_user(db: Database, session_id: str) -> str:
    row = db.first_user_message(session_id)
    if row is None or not (row["content"] or "").strip():
        return _DEFAULT_CHAT_TITLE
    return " ".join(row["content"].split())[:60] or _DEFAULT_CHAT_TITLE


def _title_from_messages(messages: list[dict[str, Any]]) -> str:
    for m in messages:
        if m.get("role") == "user" and (m.get("content") or "").strip():
            return " ".join(m["content"].split())[:60] or _DEFAULT_CHAT_TITLE
    return _DEFAULT_CHAT_TITLE


def _session_meta(row: Any) -> dict[str, Any]:
    try:
        meta = json.loads(row["meta"] or "{}")
        return meta if isinstance(meta, dict) else {}
    except (ValueError, TypeError):
        return {}


def _safe_json_obj(raw: str) -> dict[str, Any]:
    try:
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else {}
    except (ValueError, TypeError):
        return {}


def _safe_json(raw: str, default: Any) -> Any:
    try:
        return json.loads(raw)
    except (ValueError, TypeError):
        return default


def _rows_to_messages(rows: list[Any]) -> list[dict[str, Any]]:
    """Rebuild the agent's LLM message list from persisted :class:`sqlite3.Row`s."""
    messages: list[dict[str, Any]] = []
    for row in rows:
        role = str(row["role"] or "")
        content = str(row["content"] or "")
        if role == "assistant":
            tool_calls = _safe_json(row["tool_calls"] or "[]", [])
            if isinstance(tool_calls, dict):
                tool_calls = tool_calls.get("tool_calls") or None
            messages.append(
                {
                    "role": "assistant",
                    "content": content or None,
                    "tool_calls": tool_calls if isinstance(tool_calls, list) and tool_calls else None,
                }
            )
        elif role == "tool":
            messages.append(
                {"role": "tool", "tool_call_id": str(row["tool_call_id"] or ""), "content": content}
            )
        else:
            messages.append({"role": role, "content": content})
    return messages


def _persist_new_messages(db: Database, session_id: str, messages: list[dict[str, Any]]) -> None:
    """Persist newly produced messages (skipping the synthetic system message)."""
    for m in messages:
        if m.get("role") == "system":
            continue
        db.add_message(
            session_id,
            str(m.get("role") or ""),
            str(m.get("content") or ""),
            tool_calls=m.get("tool_calls"),
            tool_call_id=m.get("tool_call_id"),
        )


# --------------------------------------------------------------------------- #
# Self-improvement cycles: history browser
# --------------------------------------------------------------------------- #
def list_cycles(settings: Settings | None = None, limit: int = 50) -> list[dict[str, Any]]:
    """Recent cycles (newest first) with their steps + open human review id."""
    db = open_db(settings)
    out: list[dict[str, Any]] = []
    for row in db.list_cycles()[:limit]:
        out.append(_cycle_summary(db, row))
    return out


def get_cycle_detail(
    settings: Settings | None = None, cycle_id: str | None = None
) -> dict[str, Any] | None:
    """A single cycle with full steps/events/reviews trace."""
    if not cycle_id:
        return None
    db = open_db(settings)
    row = db.get_cycle(cycle_id)
    if row is None:
        return None
    detail = _cycle_summary(db, row)
    events = []
    for ev in db.list_cycle_events(cycle_id, limit=500):
        events.append(
            {
                "id": ev["id"], "kind": ev["kind"], "message": ev["message"],
                "payload": _safe_json_obj(ev["payload"] or "{}"), "seq": ev["seq"],
            }
        )
    reviews = []
    for r in db.list_review_requests(cycle_id=cycle_id, open_only=False):
        reviews.append(
            {
                "id": r["id"], "kind": r["kind"], "verdict": r["verdict"],
                "comments": r["comments"] or "", "resolved_at": r["resolved_at"],
            }
        )
    detail["events"] = events
    detail["reviews"] = reviews
    return detail


def _cycle_summary(db: Database, row: Any) -> dict[str, Any]:
    steps = db.get_steps(row["id"])
    human_reqs = [r for r in db.list_review_requests(cycle_id=row["id"], open_only=False)
                  if r["kind"] == "human"]
    open_human = [r for r in human_reqs if r["verdict"] == "pending"]
    return {
        "id": row["id"],
        "objective": row["objective"],
        "branch": row["branch"],
        "status": row["status"],
        "ai_verdict": row["ai_verdict"],
        "human_verdict": row["human_verdict"],
        "started_at": row["started_at"],
        "ended_at": row["ended_at"],
        "steps": [
            {
                "step": s["step"], "status": s["status"],
                "commit_sha": s["commit_sha"], "summary": s["summary"],
            }
            for s in steps
        ],
        "human_review_id": (open_human[0]["id"] if open_human else human_reqs[-1]["id"]
                            if human_reqs else None),
        "reviews": [],
        "events": [],
    }


def reconcile_stale_cycles(
    settings: Settings | None = None,
    repo_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Mark ``running`` cycles that can never complete as ``stuck`` (failed).

    A cycle is stuck when its branch carries no commits ahead of ``main``
    (the process died during the work phase, before committing anything), or
    when its branch no longer exists in the repo. A cycle whose branch is the
    currently checked-out one is presumed live and is skipped. Because a
    genuinely live cycle may legitimately spend time on a green branch with
    zero commits, this is an explicit maintenance check (``nelke db cleanup``),
    not an automatic-side-effect call: the user runs it when cycles look
    abandoned.

    Returns a list of ``{id, branch, status, reason}`` for the cycles marked.
    """
    settings = settings or Settings()
    repo_path = repo_path or find_repo(settings)
    db = open_db(settings)
    if not (repo_path / ".git").exists():
        return []
    git = GitRepo(repo_path)
    current_branch = git.current_branch()
    marked: list[dict[str, Any]] = []
    for row in db.list_cycles(status="running"):
        branch = row["branch"]
        if not branch or branch == current_branch:
            continue
        if not git.branch_exists(branch):
            reason = "branch no longer exists"
        elif git.ahead_counts("main", branch) == 0:
            reason = "no commits on branch (stuck)"
        else:
            continue
        db.update_cycle(row["id"], status="stuck", ended_at=_now())
        try:
            db.add_cycle_event(
                row["id"], "stuck",
                f"cycle marked stuck by cleanup: {reason}",
                {"reason": reason},
            )
        except Exception:  # noqa: BLE001 - observability must not break cleanup
            pass
        marked.append({"id": row["id"], "branch": branch, "status": "stuck", "reason": reason})
    return marked
