"""SQLite persistence for sessions, cycles, steps, messages and review requests.

The database lives at ``~/.nelke/nelke.db``. Schema follows the v0 plan (§12).
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from nelke.core.llm import usage_cache_pct

_SCHEMA = [
    """
    CREATE TABLE IF NOT EXISTS sessions (
        id TEXT PRIMARY KEY, frontend TEXT, started_at TEXT, ended_at TEXT, meta TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS messages (
        id TEXT PRIMARY KEY, session_id TEXT, role TEXT, content TEXT,
        tool_calls TEXT, tool_call_id TEXT, created_at TEXT
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
        cache_read_tokens INTEGER NOT NULL DEFAULT 0,
        cache_read_pct INTEGER NOT NULL DEFAULT 0,
        created_at TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS cycle_events (
        id TEXT PRIMARY KEY, cycle_id TEXT, kind TEXT, message TEXT,
        payload TEXT, created_at TEXT, seq INTEGER
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS cycle_workers (
        id TEXT PRIMARY KEY, cycle_id TEXT, worker_index INTEGER,
        title TEXT, detail TEXT, status TEXT, started_at TEXT, ended_at TEXT
    )
    """,
    # Projects group multiple chats (and future work) under a single
    # user-facing unit. `sessions.project_id` is an optional FK to this table
    # (nullable so a chat without a project stays valid). `stage` is a free-form
    # label (e.g. "idea", "in-progress", "done") shown on the project card.
    """
    CREATE TABLE IF NOT EXISTS projects (
        id TEXT PRIMARY KEY, name TEXT NOT NULL, description TEXT,
        stage TEXT, meta TEXT, created_at TEXT, updated_at TEXT
    )
    """,
    # Kanban boards for projects: each project owns zero or more boards.
    # Cards reference a column and optionally a project task id.
    """
    CREATE TABLE IF NOT EXISTS kanban_boards (
        id TEXT PRIMARY KEY, project_id TEXT NOT NULL, name TEXT NOT NULL,
        description TEXT, created_at TEXT, updated_at TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS kanban_columns (
        id TEXT PRIMARY KEY, board_id TEXT NOT NULL, name TEXT NOT NULL,
        position INTEGER NOT NULL DEFAULT 0, created_at TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS kanban_cards (
        id TEXT PRIMARY KEY, board_id TEXT NOT NULL, column_id TEXT NOT NULL,
        title TEXT NOT NULL, description TEXT, task_id TEXT,
        position INTEGER NOT NULL DEFAULT 0, created_at TEXT, updated_at TEXT
    )
    """,
]

# Additive migrations applied after the base schema (for databases created by
# older versions). Each is wrapped in a try/except so re-running is safe.
_MIGRATIONS = [
    "ALTER TABLE messages ADD COLUMN tool_call_id TEXT",
    # Message tree: branching / swipes / soft delete. Defaults keep legacy
    # behaviour intact (linear active chain, nothing deleted, no parent).
    "ALTER TABLE messages ADD COLUMN parent_id TEXT",
    "ALTER TABLE messages ADD COLUMN is_active INTEGER NOT NULL DEFAULT 1",
    "ALTER TABLE messages ADD COLUMN is_deleted INTEGER NOT NULL DEFAULT 0",
    "ALTER TABLE messages ADD COLUMN sibling_order INTEGER NOT NULL DEFAULT 0",
    "CREATE INDEX IF NOT EXISTS idx_messages_session_active "
    "ON messages(session_id) WHERE is_active = 1 AND is_deleted = 0",
    "CREATE INDEX IF NOT EXISTS idx_messages_parent ON messages(parent_id)",
    "ALTER TABLE usage_events ADD COLUMN cache_read_tokens INTEGER NOT NULL DEFAULT 0",
    "ALTER TABLE usage_events ADD COLUMN cache_read_pct INTEGER NOT NULL DEFAULT 0",
    # Parallel cycle workers: each worker row records the slice of the
    # objective it owns. `cycle_steps.worker_id` (nullable) ties a step to the
    # worker that produced it; null keeps existing single-worker cycles intact.
    "ALTER TABLE cycle_steps ADD COLUMN worker_id TEXT",
    "CREATE INDEX IF NOT EXISTS idx_cycle_workers_cycle ON cycle_workers(cycle_id)",
    # Projects: optional Chat-to-Project relation. Legacy chats keep NULL here.
    "ALTER TABLE sessions ADD COLUMN project_id TEXT",
    "CREATE INDEX IF NOT EXISTS idx_sessions_project ON sessions(project_id)",
    # Projects: free-form stage label on the project card. Added after the
    # initial Projects migration, so older project rows get NULL here.
    "ALTER TABLE projects ADD COLUMN stage TEXT",
    "CREATE INDEX IF NOT EXISTS idx_kanban_boards_project ON kanban_boards(project_id)",
    "CREATE INDEX IF NOT EXISTS idx_kanban_columns_board ON kanban_columns(board_id)",
    "CREATE INDEX IF NOT EXISTS idx_kanban_cards_board ON kanban_cards(board_id)",
    "CREATE INDEX IF NOT EXISTS idx_kanban_cards_column ON kanban_cards(column_id)",
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

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        """Context manager that COMMITS and CLOSES the connection on exit.

        The plain ``with self.connect() as conn:`` form relies on
        sqlite3.Connection's ``__exit__``, which only commits and never
        closes — every query therefore leaked an open file handle (visible
        as ResourceWarning under tracemalloc, and worse under parallel
        workers). This wraps both behaviours: on normal exit we commit then
        close; on an exception we roll back then close.
        """
        conn = self.connect()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def migrate(self) -> None:
        with self._conn() as conn:
            for statement in _SCHEMA:
                conn.execute(statement)
            for statement in _MIGRATIONS:
                try:
                    conn.execute(statement)
                except sqlite3.OperationalError:
                    # column already exists (older DB re-migrated)
                    pass
            self._backfill_message_tree(conn)
            conn.commit()

    @staticmethod
    def _backfill_message_tree(conn: sqlite3.Connection) -> None:
        """Link legacy rows into a linear active chain.

        Pre-tree databases have ``parent_id IS NULL`` on every message. For
        each session we order rows by ``created_at`` and chain each message to
        its predecessor so the existing transcript becomes a single active
        branch (the default view for both legacy and tree-aware callers).
        Idempotent: rows that already carry a non-null ``parent_id`` (or whose
        predecessor already has one) are left alone.
        """
        cur = conn.execute(
            "SELECT DISTINCT session_id FROM messages WHERE parent_id IS NULL "
            "ORDER BY session_id"
        )
        session_ids = [r["session_id"] for r in cur.fetchall()]
        for session_id in session_ids:
            rows = conn.execute(
                "SELECT id FROM messages WHERE session_id=? "
                "ORDER BY created_at, rowid",
                (session_id,),
            ).fetchall()
            prev_id: str | None = None
            for row in rows:
                conn.execute(
                    "UPDATE messages SET parent_id=? WHERE id=? AND parent_id IS NULL",
                    (prev_id, row["id"]),
                )
                prev_id = row["id"]

    def _prepare(self) -> None:
        if not self.path.exists():
            self.migrate()

    # ---- sessions / messages -------------------------------------------------
    def create_session(self, frontend: str, meta: dict | None = None) -> str:
        self._prepare()
        sid = new_id()
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO sessions (id, frontend, started_at, meta) VALUES (?,?,?,?)",
                (sid, frontend, _now(), json.dumps(meta or {})),
            )
        return sid

    def end_session(self, session_id: str) -> None:
        with self._conn() as conn:
            conn.execute(
                "UPDATE sessions SET ended_at=? WHERE id=?", (_now(), session_id)
            )

    def get_session(self, session_id: str) -> sqlite3.Row | None:
        with self._conn() as conn:
            return conn.execute(
                "SELECT * FROM sessions WHERE id=?", (session_id,)
            ).fetchone()

    def list_sessions(
        self, frontend: str | None = None, limit: int | None = None
    ) -> list[sqlite3.Row]:
        """Sessions ordered by last activity (message time, else start time).

        Each row includes ``message_count`` and ``last_message_at`` computed
        from the ``messages`` table so chat lists can render titles/labels
        without a second query per row.
        """
        q = (
            "SELECT s.*, "
            "(SELECT COUNT(*) FROM messages m WHERE m.session_id = s.id) AS message_count, "
            "(SELECT MAX(m.created_at) FROM messages m WHERE m.session_id = s.id) AS last_message_at "
            "FROM sessions s"
        )
        conds: list[str] = []
        args: list[str] = []
        if frontend:
            conds.append("s.frontend = ?")
            args.append(frontend)
        if conds:
            q += " WHERE " + " AND ".join(conds)
        q += (
            " ORDER BY COALESCE(last_message_at, s.started_at) DESC, "
            "(message_count > 0) DESC, s.started_at DESC, rowid DESC"
        )
        if limit is not None:
            q += " LIMIT ?"
            args.append(str(limit))
        with self._conn() as conn:
            return list(conn.execute(q, args))

    def update_session_meta(self, session_id: str, **fields: Any) -> None:
        """Merge ``fields`` into the session's ``meta`` JSON blob."""
        row = self.get_session(session_id)
        if row is None:
            return
        try:
            meta = json.loads(row["meta"] or "{}")
        except (ValueError, TypeError):
            meta = {}
        if not isinstance(meta, dict):
            meta = {}
        meta.update(fields)
        with self._conn() as conn:
            conn.execute(
                "UPDATE sessions SET meta=? WHERE id=?", (json.dumps(meta), session_id)
            )

    def delete_session(self, session_id: str) -> None:
        """Remove a chat session and its messages (chat management)."""
        with self._conn() as conn:
            conn.execute("DELETE FROM messages WHERE session_id=?", (session_id,))
            conn.execute("DELETE FROM sessions WHERE id=?", (session_id,))

    # ---- projects ----------------------------------------------------------
    def create_project(
        self,
        name: str,
        description: str = "",
        stage: str = "",
        meta: dict[str, Any] | None = None,
        project_id: str | None = None,
    ) -> str:
        """Create a project and return its id."""
        self._prepare()
        pid = project_id or new_id()
        now = _now()
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO projects (id, name, description, stage, meta, created_at, updated_at) "
                "VALUES (?,?,?,?,?,?,?)",
                (pid, name, description, stage, json.dumps(meta or {}), now, now),
            )
        return pid

    def get_project(self, project_id: str) -> sqlite3.Row | None:
        with self._conn() as conn:
            return conn.execute(
                "SELECT * FROM projects WHERE id=?", (project_id,)
            ).fetchone()

    def list_projects(self, limit: int | None = None) -> list[sqlite3.Row]:
        """Projects ordered by most-recently updated, with chat counts."""
        q = (
            "SELECT p.*, "
            "(SELECT COUNT(*) FROM sessions s WHERE s.project_id = p.id) AS chat_count "
            "FROM projects p ORDER BY p.updated_at DESC"
        )
        if limit is not None:
            q += " LIMIT ?"
        with self._conn() as conn:
            if limit is not None:
                return list(conn.execute(q, (str(limit),)))
            return list(conn.execute(q))

    def update_project(
        self,
        project_id: str,
        *,
        name: str | None = None,
        description: str | None = None,
        stage: str | None = None,
        meta: dict[str, Any] | None = None,
    ) -> bool:
        """Update a project's metadata. Returns False if the project is missing."""
        row = self.get_project(project_id)
        if row is None:
            return False
        fields: dict[str, Any] = {}
        if name is not None:
            fields["name"] = name
        if description is not None:
            fields["description"] = description
        if stage is not None:
            fields["stage"] = stage
        if meta is not None:
            try:
                merged = json.loads(row["meta"] or "{}")
            except (ValueError, TypeError):
                merged = {}
            if not isinstance(merged, dict):
                merged = {}
            merged.update(meta)
            fields["meta"] = json.dumps(merged)
        fields["updated_at"] = _now()
        cols = ", ".join(f"{k}=?" for k in fields)
        with self._conn() as conn:
            conn.execute(
                f"UPDATE projects SET {cols} WHERE id=?",
                (*fields.values(), project_id),
            )
        return True

    def delete_project(self, project_id: str) -> bool:
        """Remove a project, optionally detaching (not deleting) its chats.

        Returns False if the project does not exist.
        """
        if self.get_project(project_id) is None:
            return False
        with self._conn() as conn:
            conn.execute("UPDATE sessions SET project_id=NULL WHERE project_id=?", (project_id,))
            conn.execute("DELETE FROM projects WHERE id=?", (project_id,))
        return True

    def set_session_project(self, session_id: str, project_id: str | None) -> bool:
        """Attach (or detach, when ``project_id`` is None) a chat to a project."""
        if project_id is not None and self.get_project(project_id) is None:
            return False
        if self.get_session(session_id) is None:
            return False
        with self._conn() as conn:
            conn.execute(
                "UPDATE sessions SET project_id=? WHERE id=?",
                (project_id, session_id),
            )
        return True

    def list_project_sessions(self, project_id: str) -> list[sqlite3.Row]:
        """Chats attached to a project, most recent first.

        Each row carries ``message_count`` and ``last_message_at`` (computed
        from the messages table, like ``list_sessions``) so a project card can
        show per-chat activity without a follow-up query.
        """
        if self.get_project(project_id) is None:
            return []
        with self._conn() as conn:
            return list(
                conn.execute(
                    "SELECT s.*, "
                    "(SELECT COUNT(*) FROM messages m WHERE m.session_id = s.id) AS message_count, "
                    "(SELECT MAX(m.created_at) FROM messages m WHERE m.session_id = s.id) AS last_message_at "
                    "FROM sessions s WHERE s.project_id=? "
                    "ORDER BY COALESCE(last_message_at, s.started_at) DESC",
                    (project_id,),
                )
            )

    # ---- kanban boards -------------------------------------------------------
    def create_board(
        self,
        project_id: str,
        name: str,
        description: str = "",
        columns: list[str] | None = None,
        board_id: str | None = None,
    ) -> str:
        """Create a kanban board for a project and return its id.

        When ``columns`` is given, a column is created for each label (in
        order) with ``position`` 0..n-1. A board with no explicit columns
        starts empty (callers commonly add columns via ``add_column``).
        """
        self._prepare()
        if self.get_project(project_id) is None:
            raise ValueError(f"unknown project: {project_id}")
        bid = board_id or new_id()
        now = _now()
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO kanban_boards (id, project_id, name, description, created_at, updated_at) "
                "VALUES (?,?,?,?,?,?)",
                (bid, project_id, name, description, now, now),
            )
        for i, label in enumerate(columns or []):
            self.add_column(bid, label, position=i)
        return bid

    def get_board(self, board_id: str) -> sqlite3.Row | None:
        with self._conn() as conn:
            return conn.execute(
                "SELECT * FROM kanban_boards WHERE id=?", (board_id,)
            ).fetchone()

    def list_boards(self, project_id: str | None = None) -> list[sqlite3.Row]:
        """Boards for a project (or all boards), with column and card counts."""
        if project_id is not None and self.get_project(project_id) is None:
            return []
        q = (
            "SELECT b.*, "
            "(SELECT COUNT(*) FROM kanban_columns c WHERE c.board_id = b.id) AS column_count, "
            "(SELECT COUNT(*) FROM kanban_cards k WHERE k.board_id = b.id) AS card_count "
            "FROM kanban_boards b"
        )
        args: list[str] = []
        if project_id is not None:
            q += " WHERE b.project_id=?"
            args.append(project_id)
        q += " ORDER BY b.created_at"
        with self._conn() as conn:
            return list(conn.execute(q, args))

    def update_board(
        self, board_id: str, *, name: str | None = None, description: str | None = None
    ) -> bool:
        """Update a board's name/description. Returns False if the board is missing."""
        if self.get_board(board_id) is None:
            return False
        fields: dict[str, Any] = {}
        if name is not None:
            fields["name"] = name
        if description is not None:
            fields["description"] = description
        fields["updated_at"] = _now()
        cols = ", ".join(f"{k}=?" for k in fields)
        with self._conn() as conn:
            conn.execute(
                f"UPDATE kanban_boards SET {cols} WHERE id=?",
                (*fields.values(), board_id),
            )
        return True

    def delete_board(self, board_id: str) -> bool:
        """Remove a board and its columns and cards. Returns False if missing."""
        if self.get_board(board_id) is None:
            return False
        with self._conn() as conn:
            conn.execute("DELETE FROM kanban_cards WHERE board_id=?", (board_id,))
            conn.execute("DELETE FROM kanban_columns WHERE board_id=?", (board_id,))
            conn.execute("DELETE FROM kanban_boards WHERE id=?", (board_id,))
        return True

    # ---- kanban columns ------------------------------------------------------
    def next_column_position(self, board_id: str) -> int:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT COALESCE(MAX(position), -1) AS m FROM kanban_columns WHERE board_id=?",
                (board_id,),
            ).fetchone()
            return int(row["m"]) + 1

    def add_column(
        self,
        board_id: str,
        name: str,
        position: int | None = None,
        column_id: str | None = None,
    ) -> str:
        """Add a column to a board at ``position`` (default: append at end)."""
        self._prepare()
        if self.get_board(board_id) is None:
            raise ValueError(f"unknown board: {board_id}")
        if position is None:
            position = self.next_column_position(board_id)
        cid = column_id or new_id()
        with self._conn() as conn:
            # Make room for an explicit position: shift any existing column at
            # or after the target one slot so positions stay unique/sorted.
            conn.execute(
                "UPDATE kanban_columns SET position=position+1 "
                "WHERE board_id=? AND position>=?",
                (board_id, position),
            )
            conn.execute(
                "INSERT INTO kanban_columns (id, board_id, name, position, created_at) "
                "VALUES (?,?,?,?,?)",
                (cid, board_id, name, position, _now()),
            )
        return cid

    def get_column(self, column_id: str) -> sqlite3.Row | None:
        with self._conn() as conn:
            return conn.execute(
                "SELECT * FROM kanban_columns WHERE id=?", (column_id,)
            ).fetchone()

    def list_columns(self, board_id: str) -> list[sqlite3.Row]:
        """Columns of a board ordered by ``position``, each with its card count."""
        if self.get_board(board_id) is None:
            return []
        with self._conn() as conn:
            return list(
                conn.execute(
                    "SELECT c.*, "
                    "(SELECT COUNT(*) FROM kanban_cards k WHERE k.column_id = c.id) AS card_count "
                    "FROM kanban_columns c WHERE c.board_id=? ORDER BY c.position, c.created_at",
                    (board_id,),
                )
            )

    def update_column(self, column_id: str, *, name: str | None = None, position: int | None = None) -> bool:
        """Update a column's name/position. Returns False if the column is missing."""
        if self.get_column(column_id) is None:
            return False
        fields: dict[str, Any] = {}
        if name is not None:
            fields["name"] = name
        if position is not None:
            fields["position"] = position
        cols = ", ".join(f"{k}=?" for k in fields)
        with self._conn() as conn:
            conn.execute(
                f"UPDATE kanban_columns SET {cols} WHERE id=?",
                (*fields.values(), column_id),
            )
        return True

    def delete_column(self, column_id: str) -> bool:
        """Remove a column and move its cards... to the trash (cards are deleted).

        Returns False if the column is missing. Cards in the column are removed
        with it — callers that want to preserve content should move them first.
        """
        if self.get_column(column_id) is None:
            return False
        with self._conn() as conn:
            conn.execute("DELETE FROM kanban_cards WHERE column_id=?", (column_id,))
            conn.execute("DELETE FROM kanban_columns WHERE id=?", (column_id,))
        return True

    # ---- kanban cards --------------------------------------------------------
    def next_card_position(self, column_id: str) -> int:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT COALESCE(MAX(position), -1) AS m FROM kanban_cards WHERE column_id=?",
                (column_id,),
            ).fetchone()
            return int(row["m"]) + 1

    def add_card(
        self,
        board_id: str,
        column_id: str,
        title: str,
        description: str = "",
        task_id: str | None = None,
        position: int | None = None,
        card_id: str | None = None,
    ) -> str:
        """Add a card to a column. ``position`` defaults to the end of the column."""
        self._prepare()
        if self.get_board(board_id) is None:
            raise ValueError(f"unknown board: {board_id}")
        if self.get_column(column_id) is None:
            raise ValueError(f"unknown column: {column_id}")
        if position is None:
            position = self.next_card_position(column_id)
        kid = card_id or new_id()
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO kanban_cards "
                "(id, board_id, column_id, title, description, task_id, position, created_at, updated_at) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
                (kid, board_id, column_id, title, description, task_id, position, _now(), _now()),
            )
        return kid

    def get_card(self, card_id: str) -> sqlite3.Row | None:
        with self._conn() as conn:
            return conn.execute(
                "SELECT * FROM kanban_cards WHERE id=?", (card_id,)
            ).fetchone()

    def list_cards(self, board_id: str, column_id: str | None = None) -> list[sqlite3.Row]:
        """Cards of a board, optionally filtered to one column, ordered by position."""
        if self.get_board(board_id) is None:
            return []
        q = "SELECT * FROM kanban_cards WHERE board_id=?"
        args: list[str] = [board_id]
        if column_id is not None:
            q += " AND column_id=?"
            args.append(column_id)
        q += " ORDER BY position, created_at"
        with self._conn() as conn:
            return list(conn.execute(q, args))

    def update_card(
        self,
        card_id: str,
        *,
        column_id: str | None = None,
        title: str | None = None,
        description: str | None = None,
        task_id: str | None = None,
        position: int | None = None,
    ) -> bool:
        """Update a card's fields. Returns False if the card is missing."""
        if self.get_card(card_id) is None:
            return False
        fields: dict[str, Any] = {}
        if column_id is not None:
            fields["column_id"] = column_id
        if title is not None:
            fields["title"] = title
        if description is not None:
            fields["description"] = description
        if task_id is not None:
            fields["task_id"] = task_id
        if position is not None:
            fields["position"] = position
        fields["updated_at"] = _now()
        cols = ", ".join(f"{k}=?" for k in fields)
        with self._conn() as conn:
            conn.execute(
                f"UPDATE kanban_cards SET {cols} WHERE id=?",
                (*fields.values(), card_id),
            )
        return True

    def move_card(self, card_id: str, column_id: str, position: int | None = None) -> bool:
        """Move a card to another column (or reorder it in place)."""
        if self.get_card(card_id) is None:
            return False
        if self.get_column(column_id) is None:
            return False
        if position is None:
            position = self.next_card_position(column_id)
        with self._conn() as conn:
            conn.execute(
                "UPDATE kanban_cards SET column_id=?, position=?, updated_at=? WHERE id=?",
                (column_id, position, _now(), card_id),
            )
        return True

    def delete_card(self, card_id: str) -> bool:
        """Remove a card. Returns False if the card is missing."""
        if self.get_card(card_id) is None:
            return False
        with self._conn() as conn:
            conn.execute("DELETE FROM kanban_cards WHERE id=?", (card_id,))
        return True

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        tool_calls: list[dict[str, Any]] | None = None,
        tool_call_id: str | None = None,
        *,
        parent_id: str | None = None,
        is_active: bool = True,
        sibling_order: int = 0,
    ) -> str:
        self._prepare()
        mid = new_id()
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO messages "
                "(id, session_id, role, content, tool_calls, tool_call_id, created_at, "
                " parent_id, is_active, is_deleted, sibling_order) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (
                    mid,
                    session_id,
                    role,
                    content,
                    json.dumps(tool_calls or []),
                    tool_call_id,
                    _now(),
                    parent_id,
                    1 if is_active else 0,
                    0,
                    sibling_order,
                ),
            )
        return mid

    def list_messages(
        self, session_id: str, *, active_only: bool = True, include_deleted: bool = False
    ) -> list[sqlite3.Row]:
        """Persisted message rows for a session, ordered chronologically.

        By default returns the currently visible transcript: the single active
        path with soft-deleted rows filtered out — that matches the legacy
        contract used everywhere outside the branching UI. Pass
        ``active_only=False`` to receive every node (for tree rendering) and
        ``include_deleted=True`` to keep tombstones as well.
        """
        conds = ["session_id=?"]
        if active_only:
            conds.append("is_active=1")
        if not include_deleted:
            conds.append("is_deleted=0")
        where = " AND ".join(conds)
        with self._conn() as conn:
            return list(
                conn.execute(
                    f"SELECT * FROM messages WHERE {where} ORDER BY created_at, rowid",
                    (session_id,),
                )
            )

    def first_user_message(self, session_id: str) -> sqlite3.Row | None:
        """The first user message, used to derive a chat title."""
        with self._conn() as conn:
            return conn.execute(
                "SELECT * FROM messages WHERE session_id=? AND role='user' "
                "ORDER BY created_at LIMIT 1",
                (session_id,),
            ).fetchone()

    # ---- message tree (branching / swipes) ----------------------------------
    def get_message(self, message_id: str) -> sqlite3.Row | None:
        with self._conn() as conn:
            return conn.execute(
                "SELECT * FROM messages WHERE id=?", (message_id,)
            ).fetchone()

    def list_message_tree(self, session_id: str) -> list[sqlite3.Row]:
        """Every non-deleted message node in a session (all branches)."""
        with self._conn() as conn:
            return list(
                conn.execute(
                    "SELECT * FROM messages WHERE session_id=? AND is_deleted=0 "
                    "ORDER BY sibling_order, created_at, rowid",
                    (session_id,),
                )
            )

    def get_children(self, parent_id: str) -> list[sqlite3.Row]:
        """Direct children of a node — the swipe alternatives for that turn."""
        with self._conn() as conn:
            return list(
                conn.execute(
                    "SELECT * FROM messages WHERE parent_id=? AND is_deleted=0 "
                    "ORDER BY sibling_order, created_at, rowid",
                    (parent_id,),
                )
            )

    def next_sibling_order(self, parent_id: str | None, session_id: str) -> int:
        """Next ``sibling_order`` value for a new child of ``parent_id``."""
        with self._conn() as conn:
            if parent_id is None:
                row = conn.execute(
                    "SELECT COALESCE(MAX(sibling_order), -1) AS m FROM messages "
                    "WHERE session_id=? AND parent_id IS NULL",
                    (session_id,),
                ).fetchone()
            else:
                row = conn.execute(
                    "SELECT COALESCE(MAX(sibling_order), -1) AS m FROM messages "
                    "WHERE parent_id=?",
                    (parent_id,),
                ).fetchone()
            return int(row["m"]) + 1

    def update_message_content(self, message_id: str, content: str) -> None:
        with self._conn() as conn:
            conn.execute(
                "UPDATE messages SET content=? WHERE id=?", (content, message_id)
            )

    def soft_delete_message(self, message_id: str) -> list[str]:
        """Mark a message and its descendants as deleted.

        Returns the ids of every row that was tombstoned, in deletion order
        (target first, then its subtree). Useful for callers that want to
        report what was removed.
        """
        deleted: list[str] = []
        with self._conn() as conn:
            stack = [message_id]
            while stack:
                current = stack.pop()
                conn.execute(
                    "UPDATE messages SET is_deleted=1, is_active=0 WHERE id=?",
                    (current,),
                )
                deleted.append(current)
                child_rows = conn.execute(
                    "SELECT id FROM messages WHERE parent_id=?", (current,)
                ).fetchall()
                stack.extend(r["id"] for r in child_rows)
        return deleted

    def set_message_active(self, message_id: str, is_active: bool) -> None:
        with self._conn() as conn:
            conn.execute(
                "UPDATE messages SET is_active=? WHERE id=?",
                (1 if is_active else 0, message_id),
            )

    def set_active_path(self, message_id: str) -> list[str]:
        """Activate the root→leaf path through ``message_id``.

        For every node on the path, its siblings (other children of the same
        parent) are deactivated so only one branch per level is live. Scope is
        the message's own session — activating a branch in one chat never
        disturbs another. Returns the activated message ids in root→leaf order.
        """
        # Walk from the target up to its root, collecting the chain.
        chain: list[str] = []
        session_id: str | None = None
        current: str | None = message_id
        with self._conn() as conn:
            seen: set[str] = set()
            while current and current not in seen:
                seen.add(current)
                chain.append(current)
                row = conn.execute(
                    "SELECT parent_id, session_id FROM messages WHERE id=?", (current,)
                ).fetchone()
                if row and session_id is None:
                    session_id = row["session_id"]
                current = row["parent_id"] if row else None
            chain.reverse()  # root → leaf

            # At every level: deactivate this node's siblings, then mark this
            # node active. Siblings share the node's parent_id, so deactivating
            # them isolates the chosen branch within the session.
            for node_id in chain:
                row = conn.execute(
                    "SELECT parent_id FROM messages WHERE id=?", (node_id,)
                ).fetchone()
                parent = row["parent_id"] if row else None
                conn.execute(
                    "UPDATE messages SET is_active=0 "
                    "WHERE session_id=? AND parent_id IS ? AND id<>? AND is_deleted=0",
                    (session_id, parent, node_id),
                )
                conn.execute(
                    "UPDATE messages SET is_active=1 WHERE id=?", (node_id,)
                )
        return chain

    def active_leaf(self, session_id: str) -> sqlite3.Row | None:
        """The last active message in the session (deepest active node).

        Active nodes with no active children are leaves of the current path.
        Returns ``None`` for an empty chat.
        """
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT m.* FROM messages m "
                "WHERE m.session_id=? AND m.is_active=1 AND m.is_deleted=0 "
                "AND NOT EXISTS ("
                "  SELECT 1 FROM messages c "
                "  WHERE c.parent_id=m.id AND c.is_active=1 AND c.is_deleted=0"
                ") "
                "ORDER BY m.created_at DESC LIMIT 1",
                (session_id,),
            ).fetchall()
            return rows[0] if rows else None

    # ---- cycles / steps ------------------------------------------------------
    def create_cycle(self, objective: str, branch: str, cycle_id: str | None = None) -> str:
        self._prepare()
        cid = cycle_id or new_id()
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO cycles (id, objective, branch, status, started_at) VALUES (?,?,?,?,?)",
                (cid, objective, branch, "running", _now()),
            )
        return cid

    def update_cycle(self, cycle_id: str, **fields: str) -> None:
        if not fields:
            return
        cols = ", ".join(f"{k}=?" for k in fields)
        with self._conn() as conn:
            conn.execute(
                f"UPDATE cycles SET {cols} WHERE id=?",
                (*fields.values(), cycle_id),
            )

    def get_cycle(self, cycle_id: str) -> sqlite3.Row | None:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM cycles WHERE id=?", (cycle_id,)).fetchone()
        return row

    def list_cycles(self, status: str | None = None) -> list[sqlite3.Row]:
        with self._conn() as conn:
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
        worker_id: str | None = None,
    ) -> str:
        self._prepare()
        sid = new_id()
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO cycle_steps (id, cycle_id, step, commit_sha, status, summary, created_at, worker_id) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (sid, cycle_id, step, commit_sha, status, summary, _now(), worker_id),
            )
        return sid

    def get_steps(self, cycle_id: str) -> list[sqlite3.Row]:
        with self._conn() as conn:
            return list(
                conn.execute(
                    "SELECT * FROM cycle_steps WHERE cycle_id=? ORDER BY step", (cycle_id,)
                )
            )

    # ---- cycle workers (parallel planner→worker model) -----------------------
    def create_cycle_worker(
        self,
        cycle_id: str,
        worker_index: int,
        title: str,
        detail: str,
        worker_id: str | None = None,
    ) -> str:
        """Record one slice of the planner's task breakdown.

        ``worker_index`` is the 0-based position in the planner's output; the
        id is stable across retries of the same cycle so events and steps can
        reference it.
        """
        self._prepare()
        wid = worker_id or new_id()
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO cycle_workers (id, cycle_id, worker_index, title, detail, status, started_at) "
                "VALUES (?,?,?,?,?,?,?)",
                (wid, cycle_id, worker_index, title, detail, "pending", _now()),
            )
        return wid

    def update_cycle_worker(self, worker_id: str, **fields: Any) -> None:
        if not fields:
            return
        cols = ", ".join(f"{k}=?" for k in fields)
        with self._conn() as conn:
            conn.execute(
                f"UPDATE cycle_workers SET {cols} WHERE id=?",
                (*fields.values(), worker_id),
            )

    def list_cycle_workers(self, cycle_id: str) -> list[sqlite3.Row]:
        with self._conn() as conn:
            return list(
                conn.execute(
                    "SELECT * FROM cycle_workers WHERE cycle_id=? ORDER BY worker_index",
                    (cycle_id,),
                )
            )

    def get_cycle_worker(self, worker_id: str) -> sqlite3.Row | None:
        with self._conn() as conn:
            return conn.execute(
                "SELECT * FROM cycle_workers WHERE id=?", (worker_id,)
            ).fetchone()

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
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO cycle_events (id, cycle_id, kind, message, payload, created_at, seq) "
                "VALUES (?,?,?,?,?,?,?)",
                (eid, cycle_id, kind, message, json.dumps(payload or {}), _now(), seq),
            )
        return eid

    def _next_cycle_event_seq(self, cycle_id: str) -> int:
        with self._conn() as conn:
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
        with self._conn() as conn:
            return list(conn.execute(q, args))

    # ---- review requests -----------------------------------------------------
    def create_review_request(self, cycle_id: str, kind: str, **fields: str) -> str:
        self._prepare()
        rid = new_id()
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO review_requests (id, cycle_id, kind, verdict, comments, created_at) "
                "VALUES (?,?,?,?,?,?)",
                (rid, cycle_id, kind, fields.get("verdict", "pending"), fields.get("comments", ""), _now()),
            )
        return rid

    def resolve_review_request(self, request_id: str, verdict: str) -> None:
        with self._conn() as conn:
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
        with self._conn() as conn:
            return list(conn.execute(q, args))

    # ---- kanban boards / columns / cards -------------------------------------
    def create_kanban_board(
        self,
        project_id: str,
        name: str,
        description: str = "",
        board_id: str | None = None,
    ) -> str:
        """Create a kanban board for a project; returns its id.

        A default set of columns (Backlog / In Progress / Done) is seeded so
        the board is immediately usable. Raises nothing on missing project —
        the caller is expected to validate the project first.
        """
        self._prepare()
        bid = board_id or new_id()
        now = _now()
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO kanban_boards (id, project_id, name, description, created_at, updated_at) "
                "VALUES (?,?,?,?,?,?)",
                (bid, project_id, name, description, now, now),
            )
            for i, col_name in enumerate(("Backlog", "In Progress", "Done")):
                conn.execute(
                    "INSERT INTO kanban_columns (id, board_id, name, position, created_at) "
                    "VALUES (?,?,?,?,?)",
                    (new_id(), bid, col_name, i, now),
                )
        return bid

    def get_kanban_board(self, board_id: str) -> sqlite3.Row | None:
        with self._conn() as conn:
            return conn.execute(
                "SELECT * FROM kanban_boards WHERE id=?", (board_id,)
            ).fetchone()

    def list_kanban_boards(self, project_id: str) -> list[sqlite3.Row]:
        with self._conn() as conn:
            return list(
                conn.execute(
                    "SELECT * FROM kanban_boards WHERE project_id=? ORDER BY created_at",
                    (project_id,),
                )
            )

    def delete_kanban_board(self, board_id: str) -> bool:
        """Delete a board and all its columns + cards. False if missing."""
        if self.get_kanban_board(board_id) is None:
            return False
        with self._conn() as conn:
            card_rows = conn.execute(
                "SELECT id FROM kanban_cards WHERE board_id=?", (board_id,)
            ).fetchall()
            for cr in card_rows:
                conn.execute("DELETE FROM kanban_cards WHERE id=?", (cr["id"],))
            conn.execute("DELETE FROM kanban_columns WHERE board_id=?", (board_id,))
            conn.execute("DELETE FROM kanban_boards WHERE id=?", (board_id,))
        return True

    def list_kanban_columns(self, board_id: str) -> list[sqlite3.Row]:
        with self._conn() as conn:
            return list(
                conn.execute(
                    "SELECT * FROM kanban_columns WHERE board_id=? ORDER BY position, created_at",
                    (board_id,),
                )
            )

    def add_kanban_column(
        self, board_id: str, name: str, position: int | None = None
    ) -> str:
        """Add a column to a board at the given position (default: end)."""
        if position is None:
            cols = self.list_kanban_columns(board_id)
            position = int(cols[-1]["position"]) + 1 if cols else 0
        cid = new_id()
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO kanban_columns (id, board_id, name, position, created_at) "
                "VALUES (?,?,?,?,?)",
                (cid, board_id, name, position, _now()),
            )
        return cid

    def add_kanban_card(
        self,
        board_id: str,
        column_id: str,
        title: str,
        description: str = "",
        task_id: str | None = None,
    ) -> str:
        """Add a card to a column; returns its id.

        The card is appended at the end of the column (highest position). A
        ``task_id`` optionally links the card to a project task.
        """
        rows = self.list_kanban_cards(board_id, column_id=column_id)
        position = int(rows[-1]["position"]) + 1 if rows else 0
        card_id = new_id()
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO kanban_cards "
                "(id, board_id, column_id, title, description, task_id, position, created_at, updated_at) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
                (card_id, board_id, column_id, title, description, task_id, position, _now(), _now()),
            )
        return card_id

    def get_kanban_card(self, card_id: str) -> sqlite3.Row | None:
        with self._conn() as conn:
            return conn.execute(
                "SELECT * FROM kanban_cards WHERE id=?", (card_id,)
            ).fetchone()

    def list_kanban_cards(
        self,
        board_id: str | None = None,
        column_id: str | None = None,
    ) -> list[sqlite3.Row]:
        q = "SELECT * FROM kanban_cards"
        conds: list[str] = []
        args: list[str] = []
        if board_id:
            conds.append("board_id=?")
            args.append(board_id)
        if column_id:
            conds.append("column_id=?")
            args.append(column_id)
        if conds:
            q += " WHERE " + " AND ".join(conds)
        q += " ORDER BY position, created_at"
        with self._conn() as conn:
            return list(conn.execute(q, args))

    def move_kanban_card(
        self, card_id: str, column_id: str | None = None, position: int | None = None
    ) -> bool:
        """Move a card to another column and/or reorder it within a column.

        When ``column_id`` is provided the card is moved to that column. When
        ``position`` is provided the card is placed at that offset within its
        (possibly new) column, shifting neighbours. Wildcards cannot inject SQL
        — values are bound parameters. Returns False for an unknown card.
        """
        row = self.get_kanban_card(card_id)
        if row is None:
            return False
        target_column = column_id if column_id is not None else row["column_id"]
        target_position = position if position is not None else row["position"]
        with self._conn() as conn:
            # Normalise the target position into [0, count-1].
            count = conn.execute(
                "SELECT COUNT(*) AS c FROM kanban_cards WHERE column_id=?",
                (target_column,),
            ).fetchone()["c"]
            if count == 0:
                target_position = 0
            elif target_position < 0:
                target_position = 0
            elif target_position >= count:
                target_position = count - 1
            # Remove the card from its old slot, then reinsert at the target.
            conn.execute(
                "UPDATE kanban_cards SET position=position-1 "
                "WHERE column_id=? AND position>? AND id<>?",
                (row["column_id"], row["position"], card_id),
            )
            conn.execute(
                "UPDATE kanban_cards SET position=position+1 "
                "WHERE column_id=? AND position>=? AND id<>?",
                (target_column, target_position, card_id),
            )
            conn.execute(
                "UPDATE kanban_cards SET column_id=?, position=?, updated_at=? WHERE id=?",
                (target_column, target_position, _now(), card_id),
            )
        return True

    def set_kanban_card_task(
        self, card_id: str, task_id: str | None
    ) -> bool:
        """Bind a card to a project task (or clear the binding with ``None``)."""
        if self.get_kanban_card(card_id) is None:
            return False
        with self._conn() as conn:
            conn.execute(
                "UPDATE kanban_cards SET task_id=?, updated_at=? WHERE id=?",
                (task_id, _now(), card_id),
            )
        return True

    def update_kanban_card(
        self, card_id: str, title: str | None = None, description: str | None = None
    ) -> bool:
        """Update a card's title/description. Returns False for an unknown card."""
        if self.get_kanban_card(card_id) is None:
            return False
        fields: dict[str, Any] = {"updated_at": _now()}
        if title is not None:
            fields["title"] = title
        if description is not None:
            fields["description"] = description
        cols = ", ".join(f"{k}=?" for k in fields)
        with self._conn() as conn:
            conn.execute(
                f"UPDATE kanban_cards SET {cols} WHERE id=?",
                (*fields.values(), card_id),
            )
        return True

    def delete_kanban_card(self, card_id: str) -> bool:
        """Delete a card. Returns False for an unknown card."""
        row = self.get_kanban_card(card_id)
        if row is None:
            return False
        with self._conn() as conn:
            conn.execute("DELETE FROM kanban_cards WHERE id=?", (card_id,))
        # Re-order the remaining cards in the same column to keep positions tight.
        try:
            for i, r in enumerate(self.list_kanban_cards(column_id=row["column_id"])):
                conn = self.connect()
                try:
                    conn.execute(
                        "UPDATE kanban_cards SET position=? WHERE id=?",
                        (i, r["id"]),
                    )
                    conn.commit()
                finally:
                    conn.close()
        except Exception:  # noqa: BLE001 - re-ordering is best-effort cosmetic
            pass
        return True

    # ---- tasks ---------------------------------------------------------------
    def create_task(self, session_id: str, workspace: str) -> str:
        self._prepare()
        tid = new_id()
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO tasks (id, session_id, workspace, status, created_at) VALUES (?,?,?,?,?)",
                (tid, session_id, workspace, "running", _now()),
            )
        return tid

    def finish_task(self, task_id: str, status: str, summary: str) -> None:
        with self._conn() as conn:
            conn.execute(
                "UPDATE tasks SET status=?, summary=? WHERE id=?",
                (status, summary, task_id),
            )

    def status(self) -> dict[str, int]:
        self._prepare()
        with self._conn() as conn:
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
                "projects",
                "kanban_boards",
                "kanban_columns",
                "kanban_cards",
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
        cache_read = int(usage.get("cache_read_tokens", 0) or 0)
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO usage_events "
                "(id, session_id, cycle_id, prompt_tokens, completion_tokens, total_tokens, "
                " cache_read_tokens, cache_read_pct, created_at) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    uid,
                    session_id,
                    cycle_id,
                    int(usage.get("prompt_tokens", 0) or 0),
                    int(usage.get("completion_tokens", 0) or 0),
                    int(usage.get("total_tokens", 0) or 0),
                    cache_read,
                    int(usage.get("cache_read_pct", usage_cache_pct(usage)) or 0),
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
        with self._conn() as conn:
            return list(conn.execute(q, args))

    def usage_totals(
        self, *, session_id: str | None = None, cycle_id: str | None = None
    ) -> dict[str, int]:
        rows = self.list_usage(session_id=session_id, cycle_id=cycle_id)
        totals = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cache_read_tokens": 0,
            "calls": len(rows),
        }
        for row in rows:
            totals["prompt_tokens"] += row["prompt_tokens"]
            totals["completion_tokens"] += row["completion_tokens"]
            totals["total_tokens"] += row["total_tokens"]
            totals["cache_read_tokens"] += int(row["cache_read_tokens"] or 0)
        totals["cache_read_pct"] = usage_cache_pct(totals)
        return totals
