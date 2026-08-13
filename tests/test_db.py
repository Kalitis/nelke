"""Database schema + CRUD tests."""

from __future__ import annotations

import pytest

from nelke.core.db import Database


def test_migrate_creates_all_tables(tmp_path):
    db = Database(tmp_path / "nelke.db")
    db.migrate()
    with db.connect() as conn:
        tables = {
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
    expected = {
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
    }
    assert expected <= tables


def test_cycle_events_roundtrip(tmp_path):
    db = Database(tmp_path / "nelke.db")
    cid = db.create_cycle("improve docs", "improve/ev-more")

    def _seqs(cid):
        return [r["seq"] for r in db.list_cycle_events(cid)]

    e1 = db.add_cycle_event(cid, "cycle_start", "started", {"cycle_id": cid})
    e2 = db.add_cycle_event(cid, "agent_tool", "", {"tool": "self_write", "args": {"path": "x.py"}})
    e3 = db.add_cycle_event(cid, "merged", "merged")
    assert db.status()["cycle_events"] == 3
    events = db.list_cycle_events(cid)
    assert [r["kind"] for r in events] == ["cycle_start", "agent_tool", "merged"]
    assert _seqs(cid) == [0, 1, 2]
    # after_seq filters
    after = db.list_cycle_events(cid, after_seq=0)
    assert [r["kind"] for r in after] == ["agent_tool", "merged"]
    # limit
    lim = db.list_cycle_events(cid, limit=2)
    assert len(lim) == 2
    import json

    payload = json.loads(events[1]["payload"])
    assert payload["tool"] == "self_write"
    assert e1 and e2 and e3


def test_session_and_messages_roundtrip(tmp_path):
    db = Database(tmp_path / "nelke.db")
    sid = db.create_session("cli", {"user": "t"})
    db.add_message(sid, "user", "hi")
    db.add_message(sid, "assistant", "hello", tool_calls=[{"name": "read"}])
    msgs = db.list_messages(sid)
    assert len(msgs) == 2
    assert msgs[0]["role"] == "user"
    db.end_session(sid)
    assert db.status()["messages"] == 2


def test_cycle_and_review_flow(tmp_path):
    db = Database(tmp_path / "nelke.db")
    cid = db.create_cycle("improve docs", "improve/abc-more-docs")
    db.add_step(cid, 1, "abc123", "committed", "docs")
    db.add_step(cid, 2, "abc124", "ok", "docs2")
    db.update_cycle(cid, ai_verdict="approve")
    rid = db.create_review_request(cid, "human", verdict="pending")
    db.resolve_review_request(rid, "approved")
    db.update_cycle(cid, status="merged", human_verdict="approved", ended_at="now")
    cycle = db.get_cycle(cid)
    assert cycle["status"] == "merged"
    assert cycle["ai_verdict"] == "approve"
    steps = db.get_steps(cid)
    assert [s["status"] for s in steps] == ["committed", "ok"]
    assert not db.list_review_requests(cycle_id=cid, open_only=True)
    assert len(db.list_cycles(status="merged")) == 1


def test_task_and_workspace(tmp_path):
    db = Database(tmp_path / "nelke.db")
    sid = db.create_session("cli")
    tid = db.create_task(sid, "C:\\ws")
    db.finish_task(tid, "completed", "done quickly")
    assert db.status()["tasks"] == 1


def test_usage_events_and_totals(tmp_path):
    db = Database(tmp_path / "nelke.db")
    sid = db.create_session("cli")
    db.add_usage({"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}, session_id=sid)
    db.add_usage({"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7}, session_id=sid)
    cid = db.create_cycle("objective", "improve/x")
    db.add_usage({"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10}, cycle_id=cid)
    assert db.status()["usage_events"] == 3
    totals = db.usage_totals(session_id=sid)
    assert totals["total_tokens"] == 37
    assert totals["calls"] == 2
    cyc = db.usage_totals(cycle_id=cid)
    assert cyc["total_tokens"] == 10
    assert len(db.list_usage(cycle_id=cid)) == 1


def test_usage_persists_cache_metrics(tmp_path):
    db = Database(tmp_path / "nelke.db")
    sid = db.create_session("cli")
    db.add_usage({
        "prompt_tokens": 400, "completion_tokens": 10, "total_tokens": 410,
        "cache_read_tokens": 350,
    }, session_id=sid)
    db.add_usage({
        "prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110,
        "cache_read_tokens": 50,
    }, session_id=sid)
    rows = db.list_usage(session_id=sid)
    assert [r["cache_read_tokens"] for r in rows] == [350, 50]
    assert all(r["cache_read_pct"] == expected for r, expected in
               zip(rows, (88, 50), strict=True))
    totals = db.usage_totals(session_id=sid)
    assert totals["cache_read_tokens"] == 400
    assert totals["cache_read_pct"] == 80  # 400 of 500 prompt tokens


def test_usage_totals_omit_cache_when_absent(tmp_path):
    db = Database(tmp_path / "nelke.db")
    sid = db.create_session("cli")
    db.add_usage({"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11}, session_id=sid)
    totals = db.usage_totals(session_id=sid)
    assert totals["cache_read_tokens"] == 0
    assert totals["cache_read_pct"] == 0


def test_projects_crud_with_stage(tmp_path):
    db = Database(tmp_path / "nelke.db")
    pid = db.create_project("Nelke", description="self-improving agent", stage="idea")
    row = db.get_project(pid)
    assert row is not None
    assert row["name"] == "Nelke"
    assert row["stage"] == "idea"

    # stage is updatable independently
    assert db.update_project(pid, stage="active")
    assert db.get_project(pid)["stage"] == "active"

    # listing carries chat_count
    rows = db.list_projects()
    assert len(rows) == 1
    assert rows[0]["chat_count"] == 0

    # deleting detaches chats, does not raise
    sid = db.create_session("web")
    assert db.set_session_project(sid, pid)
    assert db.delete_project(pid)
    assert db.get_project(pid) is None
    # session still exists, but no longer attached
    assert db.get_session(sid)["project_id"] is None


def test_project_sessions_carry_message_count(tmp_path):
    db = Database(tmp_path / "nelke.db")
    pid = db.create_project("P")
    sid = db.create_session("web")
    db.set_session_project(sid, pid)
    db.add_message(sid, "user", "hello")
    db.add_message(sid, "assistant", "hi")

    rows = db.list_project_sessions(pid)
    assert len(rows) == 1
    assert rows[0]["message_count"] == 2
    assert rows[0]["id"] == sid


def test_set_session_project_rejects_unknown(tmp_path):
    db = Database(tmp_path / "nelke.db")
    sid = db.create_session("web")
    assert db.set_session_project(sid, "nope") is False
    # unknown session also rejected
    assert db.set_session_project("ghost", None) is False


# ---- kanban ---------------------------------------------------------------

def test_kanban_board_columns_cards_roundtrip(tmp_path):
    db = Database(tmp_path / "nelke.db")
    pid = db.create_project("P")
    bid = db.create_kanban_board(pid, "Sprint", columns=["Backlog", "Done"])
    board = db.get_kanban_board(bid)
    assert board["name"] == "Sprint"
    assert board["project_id"] == pid

    cols = db.list_kanban_columns(bid)
    assert [c["name"] for c in cols] == ["Backlog", "Done"]
    assert [c["position"] for c in cols] == [0, 1]

    # a board with no explicit columns ships the default set
    bid2 = db.create_kanban_board(pid, "Default")
    assert [c["name"] for c in db.list_kanban_columns(bid2)] == [
        "Backlog", "In Progress", "Done",
    ]

    # add a card to the first column
    cid = cols[0]["id"]
    card = db.add_kanban_card(bid, cid, "Write docs", "d")
    assert db.get_kanban_card(card)["title"] == "Write docs"
    assert db.get_kanban_card(card)["position"] == 0

    # listing boards returns both
    boards = db.list_kanban_boards(pid)
    assert {b["id"] for b in boards} == {bid, bid2}

    # status reflects the kanban tables
    status = db.status()
    assert status["kanban_boards"] == 2
    assert status["kanban_columns"] == 5
    assert status["kanban_cards"] == 1


def test_kanban_move_and_reorder(tmp_path):
    db = Database(tmp_path / "nelke.db")
    pid = db.create_project("P")
    bid = db.create_kanban_board(pid, "B", columns=["A", "B"])
    cols = {c["name"]: c["id"] for c in db.list_kanban_columns(bid)}

    k1 = db.add_kanban_card(bid, cols["A"], "one")
    k2 = db.add_kanban_card(bid, cols["A"], "two")
    # positions auto-assigned in order
    assert db.get_kanban_card(k1)["position"] == 0
    assert db.get_kanban_card(k2)["position"] == 1

    # move k1 into column B (appended at the end)
    assert db.move_kanban_card(k1, cols["B"])
    assert db.get_kanban_card(k1)["column_id"] == cols["B"]
    assert db.get_kanban_card(k1)["position"] == 0

    # explicit position within a column clamps into [0, count-1]
    assert db.move_kanban_card(k2, position=99)
    assert db.get_kanban_card(k2)["position"] == 0

    # list_kanban_cards filtered by column
    col_b = db.list_kanban_cards(bid, column_id=cols["B"])
    assert [c["title"] for c in col_b] == ["one"]

    # unknown card / unknown column / column of another board rejected
    assert db.move_kanban_card("ghost", cols["B"]) is False
    assert db.move_kanban_card(k2, "ghost") is False
    other = db.create_kanban_board(pid, "Other")
    other_col = db.list_kanban_columns(other)[0]["id"]
    assert db.move_kanban_card(k2, other_col) is False


def test_kanban_column_position_defaults_end(tmp_path):
    db = Database(tmp_path / "nelke.db")
    pid = db.create_project("P")
    bid = db.create_kanban_board(pid, "B", columns=[])
    a = db.add_kanban_column(bid, "A")
    b = db.add_kanban_column(bid, "B")
    assert db.get_kanban_column(a)["position"] == 0
    assert db.get_kanban_column(b)["position"] == 1
    # insertion at an explicit position shifts neighbours
    db.add_kanban_column(bid, "C", position=0)
    assert [c["name"] for c in db.list_kanban_columns(bid)] == ["C", "A", "B"]
    # rename / reposition
    assert db.update_kanban_column(a, name="AA", position=2)
    assert db.get_kanban_column(a)["name"] == "AA"
    assert db.get_kanban_column(a)["position"] == 2
    assert db.update_kanban_column("ghost", name="x") is False


def test_kanban_delete_cascades(tmp_path):
    db = Database(tmp_path / "nelke.db")
    pid = db.create_project("P")
    bid = db.create_kanban_board(pid, "B", columns=["A"])
    col = db.list_kanban_columns(bid)[0]
    db.add_kanban_card(bid, col["id"], "card")

    # deleting a column removes its cards
    assert db.delete_kanban_column(col["id"])
    assert db.get_kanban_column(col["id"]) is None
    assert db.status()["kanban_cards"] == 0

    # deleting a board removes its columns
    assert db.delete_kanban_board(bid)
    assert db.get_kanban_board(bid) is None
    assert db.status()["kanban_columns"] == 0
    assert db.status()["kanban_boards"] == 0

    # deleting unknown board/column returns False
    assert db.delete_kanban_board("ghost") is False
    assert db.delete_kanban_column("ghost") is False


def test_kanban_create_rejects_unknown_project_and_board(tmp_path):
    db = Database(tmp_path / "nelke.db")
    pid = db.create_project("P")
    with pytest.raises(ValueError):
        db.create_kanban_board("ghost", "B")
    with pytest.raises(ValueError):
        db.add_kanban_column("ghost-board", "A")
    with pytest.raises(ValueError):
        db.add_kanban_card("ghost-board", "ghost-col", "x")
    # a column of one board is not accepted by another board's card insert
    bid = db.create_kanban_board(pid, "B")
    other = db.create_kanban_board(pid, "Other")
    assert bid != other
    col = db.list_kanban_columns(bid)[0]
    with pytest.raises(ValueError):
        db.add_kanban_card(other, col["id"], "x")


def test_kanban_update_card_delete_card_and_task(tmp_path):
    db = Database(tmp_path / "nelke.db")
    pid = db.create_project("P")
    bid = db.create_kanban_board(pid, "B", columns=["A"])
    col = db.list_kanban_columns(bid)[0]
    card = db.add_kanban_card(bid, col["id"], "t")

    assert db.update_kanban_card(card, title="new", description="d")
    assert db.get_kanban_card(card)["title"] == "new"
    assert db.get_kanban_card(card)["description"] == "d"
    assert db.update_kanban_card("ghost", title="x") is False

    # bind / clear a project task
    assert db.set_kanban_card_task(card, "task-1")
    assert db.get_kanban_card(card)["task_id"] == "task-1"
    assert db.set_kanban_card_task(card, None)
    assert db.get_kanban_card(card)["task_id"] is None
    assert db.set_kanban_card_task("ghost", "t") is False

    # delete reorders remaining cards in the column to tight positions
    c2 = db.add_kanban_card(bid, col["id"], "two")
    db.add_kanban_card(bid, col["id"], "three")
    assert [c["title"] for c in db.list_kanban_cards(bid, column_id=col["id"])] == [
        "new", "two", "three",
    ]
    assert db.delete_kanban_card(c2)
    assert db.get_kanban_card(c2) is None
    assert [c["title"] for c in db.list_kanban_cards(bid, column_id=col["id"])] == [
        "new", "three",
    ]
    assert [c["position"] for c in db.list_kanban_cards(bid, column_id=col["id"])] == [0, 1]
    assert db.delete_kanban_card("ghost") is False


def test_kanban_update_board(tmp_path):
    db = Database(tmp_path / "nelke.db")
    pid = db.create_project("P")
    bid = db.create_kanban_board(pid, "B")
    assert db.update_kanban_board(bid, name="Renamed", description="D")
    board = db.get_kanban_board(bid)
    assert board["name"] == "Renamed"
    assert board["description"] == "D"
    assert db.update_kanban_board("ghost", name="x") is False
