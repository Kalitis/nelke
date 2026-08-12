"""Database schema + CRUD tests."""

from __future__ import annotations

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
