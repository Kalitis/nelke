"""SQLite persistence for sessions, cycles, steps, messages and review requests.

The database lives at ``~/.nelke/nelke.db``. Schema follows the v0 plan (§12).
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_SCHEMA = [
    """
    CREATE TABLE IF NOT EXISTS sessions (
        id TEXT PRIMARY KEY, frontend TEXT, started_at TEXT, ended_at TEXT, meta TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS messages (
        id TEXT PRIMARY KEY, session_id TEXT, role TEXT, content TEXT,
        tool_calls TEXT, created_at TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS cycles (
        id TEXT PRIMARY KEY, objective TEXT, branch TEXT, status TEXT,
        ai_verdict TEXT, human_verdict TEXT, started_at TEXT, ended_at TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS cycle_steps (
        id TEXT PRIMARY KEY, cycle_id TEXT, step INTEGER, commit_sha TEXT,
        status TEXT, summary TEXT, created_at TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS review_requests (
        id TEXT PRIMARY KEY, cycle_id TEXT, kind TEXT, verdict TEXT,
        comments TEXT, created_at TEXT, resolved_at TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS tasks (
        id TEXT PRIMARY KEY, session_id TEXT, workspace TEXT, status TEXT,
        summary TEXT, created_at TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS usage_events (
        id TEXT PRIMARY KEY, session_id TEXT, cycle_id TEXT,
        prompt_tokens INTEGER, completion_tokens INTEGER, total_tokens INTEGER,
        created_at TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS cycle_events (
        id TEXT PRIMARY KEY, cycle_id TEXT, kind TEXT, message TEXT,
        payload TEXT, created_at TEXT, seq INTEGER
    )
    """,
]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def new_id() -> str:
    return f"{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:8]}"


class Database:
    def __init__(self, path: Path) -> None:
        self.path = path

    def connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.path))
        conn.row_factory = sqlite3.Row
        return conn

    def migrate(self) -> None:
        with self.connect() as conn:
            for statement in _SCHEMA:
                conn.execute(statement)
            conn.commit()

    def _prepare(self) -> None:
        if not self.path.exists():
            self.migrate()

    # ---- sessions / messages -------------------------------------------------
    def create_session(self, frontend: str, meta: dict | None = None) -> str:
        self._prepare()
        sid = new_id()
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO sessions (id, frontend, started_at, meta) VALUES (?,?,?,?)",
                (sid, frontend, _now(), json.dumps(meta or {})),
            )
        return sid

    def end_session(self, session_id: str) -> None:
        with self.connect() as conn:
            conn.execute(
                "UPDATE sessions SET ended_at=? WHERE id=?", (_now(), session_id)
            )

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        tool_calls: list[dict[str, Any]] | None = None,
    ) -> str:
        self._prepare()
        mid = new_id()
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO messages (id, session_id, role, content, tool_calls, created_at) "
                "VALUES (?,?,?,?,?,?)",
                (mid, session_id, role, content, json.dumps(tool_calls or []), _now()),
            )
        return mid

    def list_messages(self, session_id: str) -> list[sqlite3.Row]:
        with self.connect() as conn:
            return list(
                conn.execute(
                    "SELECT * FROM messages WHERE session_id=? ORDER BY created_at",
                    (session_id,),
                )
            )

    # ---- cycles / steps ------------------------------------------------------
    def create_cycle(self, objective: str, branch: str, cycle_id: str | None = None) -> str:
        self._prepare()
        cid = cycle_id or new_id()
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO cycles (id, objective, branch, status, started_at) VALUES (?,?,?,?,?)",
                (cid, objective, branch, "running", _now()),
            )
        return cid

    def update_cycle(self, cycle_id: str, **fields: str) -> None:
        if not fields:
            return
        cols = ", ".join(f"{k}=?" for k in fields)
        with self.connect() as conn:
            conn.execute(
                f"UPDATE cycles SET {cols} WHERE id=?",
                (*fields.values(), cycle_id),
            )

    def get_cycle(self, cycle_id: str) -> sqlite3.Row | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM cycles WHERE id=?", (cycle_id,)).fetchone()
        return row

    def list_cycles(self, status: str | None = None) -> list[sqlite3.Row]:
        with self.connect() as conn:
            if status:
                return list(conn.execute("SELECT * FROM cycles WHERE status=? ORDER BY started_at DESC", (status,)))
            return list(conn.execute("SELECT * FROM cycles ORDER BY started_at DESC"))

    def add_step(
        self,
        cycle_id: str,
        step: int,
        commit_sha: str | None,
        status: str,
        summary: str,
    ) -> str:
        self._prepare()
        sid = new_id()
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO cycle_steps (id, cycle_id, step, commit_sha, status, summary, created_at) "
                "VALUES (?,?,?,?,?,?,?)",
                (sid, cycle_id, step, commit_sha, status, summary, _now()),
            )
        return sid

    def get_steps(self, cycle_id: str) -> list[sqlite3.Row]:
        with self.connect() as conn:
            return list(
                conn.execute(
                    "SELECT * FROM cycle_steps WHERE cycle_id=? ORDER BY step", (cycle_id,)
                )
            )

    # ---- cycle events (long-lived progress trace) ----------------------------
    def add_cycle_event(
        self,
        cycle_id: str,
        kind: str,
        message: str,
        payload: dict[str, Any] | None = None,
    ) -> str:
        """Persist one cycle event; ``seq`` orders events for e.g. the TUI live log."""
        self._prepare()
        eid = new_id()
        seq = self._next_cycle_event_seq(cycle_id)
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO cycle_events (id, cycle_id, kind, message, payload, created_at, seq) "
                "VALUES (?,?,?,?,?,?,?)",
                (eid, cycle_id, kind, message, json.dumps(payload or {}), _now(), seq),
            )
        return eid

    def _next_cycle_event_seq(self, cycle_id: str) -> int:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT COALESCE(MAX(seq), -1) AS m FROM cycle_events WHERE cycle_id=?", (cycle_id,)
            ).fetchone()
            return int(row["m"]) + 1

    def list_cycle_events(
        self, cycle_id: str, *, after_seq: int | None = None, limit: int | None = None
    ) -> list[sqlite3.Row]:
        q = "SELECT * FROM cycle_events WHERE cycle_id=?"
        args: list[str] = [cycle_id]
        if after_seq is not None:
            q += " AND seq>?"
            args.append(str(after_seq))
        q += " ORDER BY seq"
        if limit is not None:
            q += " LIMIT ?"
            args.append(str(limit))
        with self.connect() as conn:
            return list(conn.execute(q, args))

    # ---- review requests -----------------------------------------------------
    def create_review_request(self, cycle_id: str, kind: str, **fields: str) -> str:
        self._prepare()
        rid = new_id()
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO review_requests (id, cycle_id, kind, verdict, comments, created_at) "
                "VALUES (?,?,?,?,?,?)",
                (rid, cycle_id, kind, fields.get("verdict", "pending"), fields.get("comments", ""), _now()),
            )
        return rid

    def resolve_review_request(self, request_id: str, verdict: str) -> None:
        with self.connect() as conn:
            conn.execute(
                "UPDATE review_requests SET verdict=?, resolved_at=? WHERE id=?",
                (verdict, _now(), request_id),
            )

    def list_review_requests(
        self, cycle_id: str | None = None, open_only: bool = False
    ) -> list[sqlite3.Row]:
        q = "SELECT * FROM review_requests"
        conds: list[str] = []
        args: list[str] = []
        if cycle_id:
            conds.append("cycle_id=?")
            args.append(cycle_id)
        if open_only:
            conds.append("verdict='pending'")
        if conds:
            q += " WHERE " + " AND ".join(conds)
        q += " ORDER BY created_at"
        with self.connect() as conn:
            return list(conn.execute(q, args))

    # ---- tasks ---------------------------------------------------------------
    def create_task(self, session_id: str, workspace: str) -> str:
        self._prepare()
        tid = new_id()
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO tasks (id, session_id, workspace, status, created_at) VALUES (?,?,?,?,?)",
                (tid, session_id, workspace, "running", _now()),
            )
        return tid

    def finish_task(self, task_id: str, status: str, summary: str) -> None:
        with self.connect() as conn:
            conn.execute(
                "UPDATE tasks SET status=?, summary=? WHERE id=?",
                (status, summary, task_id),
            )

    def status(self) -> dict[str, int]:
        self._prepare()
        with self.connect() as conn:
            out: dict[str, int] = {}
            for table in (
                "sessions",
                "messages",
                "cycles",
                "cycle_steps",
                "review_requests",
                "tasks",
                "usage_events",
                "cycle_events",
            ):
                out[table] = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        return out

    # ---- token usage --------------------------------------------------------
    def add_usage(
        self,
        usage: dict[str, Any],
        *,
        session_id: str | None = None,
        cycle_id: str | None = None,
    ) -> str:
        self._prepare()
        uid = new_id()
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO usage_events "
                "(id, session_id, cycle_id, prompt_tokens, completion_tokens, total_tokens, created_at) "
                "VALUES (?,?,?,?,?,?,?)",
                (
                    uid,
                    session_id,
                    cycle_id,
                    int(usage.get("prompt_tokens", 0) or 0),
                    int(usage.get("completion_tokens", 0) or 0),
                    int(usage.get("total_tokens", 0) or 0),
                    _now(),
                ),
            )
        return uid

    def list_usage(
        self, *, session_id: str | None = None, cycle_id: str | None = None
    ) -> list[sqlite3.Row]:
        q = "SELECT * FROM usage_events"
        conds: list[str] = []
        args: list[str] = []
        if session_id:
            conds.append("session_id=?")
            args.append(session_id)
        if cycle_id:
            conds.append("cycle_id=?")
            args.append(cycle_id)
        if conds:
            q += " WHERE " + " AND ".join(conds)
        q += " ORDER BY created_at"
        with self.connect() as conn:
            return list(conn.execute(q, args))

    def usage_totals(
        self, *, session_id: str | None = None, cycle_id: str | None = None
    ) -> dict[str, int]:
        rows = self.list_usage(session_id=session_id, cycle_id=cycle_id)
        totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "calls": len(rows)}
        for row in rows:
            totals["prompt_tokens"] += row["prompt_tokens"]
            totals["completion_tokens"] += row["completion_tokens"]
            totals["total_tokens"] += row["total_tokens"]
        return totals
