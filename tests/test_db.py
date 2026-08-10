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
    assert {
        "sessions", "messages", "cycles", "cycle_steps", "review_requests", "tasks", "usage_events",
    } <= tables


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
