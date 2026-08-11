"""End-to-end self-improvement cycle tests on a real temp repo with fake LLM + governance."""

from __future__ import annotations

from conftest import FakeGovernance, driver_fake, final_response, tool_response

from nelke.core.cycle import CycleEngine
from nelke.core.governance import CheckResult, GateResult


def _engine(repo, db, gov, llm, human=lambda req: True, **kw) -> CycleEngine:
    return CycleEngine(
        repo, db, gov, llm,
        human_approve=human,
        max_steps=kw.pop("max_steps", 10),
        max_step_attempts=kw.pop("max_step_attempts", 3),
        max_review_rounds=kw.pop("max_review_rounds", 3),
        on_event=kw.pop("on_event", None),
    )


def _good_fix_plan():
    """Write a memory lesson, finish, then propose completion."""
    return [
        tool_response("self_write", {"path": "memory/facts/work.md", "content": "# Work\n\ndone"}),
        final_response("step done"),
        tool_response("propose_cycle_complete", {}),
        final_response("complete"),
    ]


def _usage_resp(resp, total: int = 2):
    """Attach usage to a scripted LLM response (defaults to total=2 tokens)."""
    from dataclasses import replace

    return replace(
        resp,
        usage={
            "prompt_tokens": total,
            "completion_tokens": total,
            "total_tokens": total,
        },
    )


async def test_cycle_merge_into_main(tmp_repo, db):
    events = []
    llm = driver_fake(worker=_scripted_worker(_good_fix_plan()))
    engine = _engine(tmp_repo, db, FakeGovernance(), llm, on_event=lambda e: events.append(e.kind))
    result = await engine.run("add a memory lesson about cycles")
    assert result.merged
    assert result.status == "merged"
    assert "merged" in events
    # file merged onto main
    assert (tmp_repo.repo / "memory" / "facts" / "work.md").exists()
    # review requests recorded (AI + human)
    reqs = db.list_review_requests(cycle_id=result.cycle_id)
    assert {r["kind"] for r in reqs} == {"ai", "human"}
    assert db.get_cycle(result.cycle_id)["status"] == "merged"


def _scripted_worker(responses):
    from conftest import scripted
    return scripted(responses)


async def test_cycle_boot_failure_reverts_and_recovers(tmp_repo, db):
    gov = FakeGovernance()
    gov.boots = [
        CheckResult("boot-check", ok=False, message="ImportError: no module"),
        CheckResult("boot-check", ok=True),
    ]
    worker = [
        tool_response("self_write", {"path": "memory/facts/roll.md", "content": "# Roll\nBROKEN"}),
        final_response("done"),
        tool_response("self_write", {"path": "memory/facts/roll.md", "content": "# Roll\nGOOD"}),
        final_response("done"),
        tool_response("propose_cycle_complete", {}),
        final_response("done"),
    ]
    llm = driver_fake(worker=_scripted_worker(worker))
    engine = _engine(tmp_repo, db, gov, llm)
    result = await engine.run("add a memory lesson")
    assert result.merged
    # a revert happened somewhere in history
    assert "revert" in tmp_repo.log(50).lower()
    # the good content survived onto main
    assert "GOOD" in (tmp_repo.repo / "memory" / "facts" / "roll.md").read_text()
    # one step was marked failed-boot
    steps = db.get_steps(result.cycle_id)
    assert any(s["status"] == "failed-boot" for s in steps)
    assert len(db.list_review_requests(cycle_id=result.cycle_id)) >= 1


async def test_cycle_human_reject_keeps_branch(tmp_repo, db):
    llm = driver_fake(worker=_scripted_worker(_good_fix_plan()))
    engine = _engine(tmp_repo, db, FakeGovernance(), llm, human=lambda req: False)
    result = await engine.run("add a memory lesson")
    assert result.status == "rejected"
    assert db.get_cycle(result.cycle_id)["human_verdict"] == "rejected"
    # branch kept, not merged: only initial commit on main
    main_log = tmp_repo._run("log", "--oneline", "main", "-10").stdout
    assert "add a memory lesson" not in main_log and "initial" in main_log
    # we stay on the improve branch
    assert tmp_repo.current_branch().startswith("improve/")


async def test_cycle_ai_request_changes_then_approves(tmp_repo, db):
    state = {"n": 0}

    def reviewer(m, t):
        state["n"] += 1
        if state["n"] == 1:
            return final_response("VERDICT: REQUEST_CHANGES\nSUMMARY: needs tests\nCOMMENTS:\n- add a test")
        return final_response("VERDICT: APPROVE\nSUMMARY: ok\nCOMMENTS: none")

    llm = driver_fake(worker=_scripted_worker(_good_fix_plan()), reviewer=reviewer)
    engine = _engine(tmp_repo, db, FakeGovernance(), llm)
    result = await engine.run("add a memory lesson")
    assert result.merged
    ai_reqs = [r for r in db.list_review_requests(cycle_id=result.cycle_id) if r["kind"] == "ai"]
    assert [r["verdict"] for r in ai_reqs] == ["request_changes", "approve"]


async def test_cycle_no_changes(tmp_repo, db):
    llm = driver_fake(worker=_scripted_worker([final_response("nothing to do")]))
    engine = _engine(tmp_repo, db, FakeGovernance(), llm)
    result = await engine.run("do nothing")
    assert result.status == "no-changes"
    assert db.get_cycle(result.cycle_id)["status"] == "no-changes"


async def test_cycle_gate_failure_stops(tmp_repo, db):
    gov = FakeGovernance()
    fail = GateResult(passed=False, checks=[CheckResult("tests", ok=False, message="boom")])
    gov.gates = [fail, fail, fail]  # max_step_attempts = 3
    llm = driver_fake(worker=_scripted_worker([final_response("done")]))
    engine = _engine(tmp_repo, db, gov, llm)
    result = await engine.run("improve something broken")
    assert result.status == "failed-gate"
    assert not tmp_repo.has_changes()  # changes were stashed


async def test_cycle_awaiting_human_without_gate(tmp_repo, db):
    llm = driver_fake(worker=_scripted_worker(_good_fix_plan()))
    engine = CycleEngine(tmp_repo, db, FakeGovernance(), llm, human_approve=None)
    result = await engine.run("add a memory lesson")
    assert result.status == "awaiting-human"
    assert db.get_cycle(result.cycle_id)["status"] == "awaiting-human"
    assert db.list_review_requests(cycle_id=result.cycle_id, open_only=True)


def test_merge_cycle_branch_shared(tmp_repo):
    """The shared merge helper used by both the engine and the CLI review path."""
    from nelke.core.cycle import merge_cycle_branch

    tmp_repo.checkout_new_branch("improve/shared", base="main")
    (tmp_repo.repo / "f.txt").write_text("one\ntwo\n", encoding="utf-8")
    tmp_repo.add_all()
    tmp_repo.commit("improve", "Nelke-Self-Improve: cycle c1 step 1")
    merge_cycle_branch(tmp_repo, "improve/shared", cycle_id="c1")
    assert tmp_repo.current_branch() == "main"
    assert (tmp_repo.repo / "f.txt").read_text() == "one\ntwo\n"
    body = tmp_repo._run("log", "-1", "--format=%B").stdout
    assert "Co-authored-by: Nelke <nelke@local>" in body


def test_cli_review_approve_merges(tmp_repo, db):
    """`nelke review approve <id>` must merge the branch onto main (Phase B2)."""
    from nelke.config import Settings
    from nelke.frontends.cli import _resolve_review

    cid = db.create_cycle("objective", "improve/cli-merge-1")
    db.create_review_request(cid, "human", verdict="pending")
    tmp_repo.checkout_new_branch("improve/cli-merge-1", base="main")
    (tmp_repo.repo / "f.txt").write_text("one\ntwo\n", encoding="utf-8")
    tmp_repo.add_all()
    tmp_repo.commit("improve f", "Nelke-Self-Improve: cycle c step 1")
    tmp_repo.checkout("main")

    req = db.list_review_requests(cycle_id=cid)[0]
    settings = Settings(nelke_home=str(db.path.parent))
    _resolve_review(settings, req["id"], "approved", tmp_repo.repo)

    cycle = db.get_cycle(cid)
    assert cycle["status"] == "merged"
    assert cycle["human_verdict"] == "approved"
    assert db.list_review_requests(cycle_id=cid)[0]["verdict"] == "approved"
    assert tmp_repo.current_branch() == "main"
    assert (tmp_repo.repo / "f.txt").read_text() == "one\ntwo\n"
    assert "Merge branch" in tmp_repo._run("log", "--merges", "--oneline").stdout


def test_cli_review_reject_keeps_branch(tmp_repo, db):
    """`nelke review reject <id>` must leave the branch unmerged (Phase B2)."""
    from nelke.config import Settings
    from nelke.frontends.cli import _resolve_review

    cid = db.create_cycle("objective", "improve/cli-reject-1")
    db.create_review_request(cid, "human", verdict="pending")
    tmp_repo.checkout_new_branch("improve/cli-reject-1", base="main")
    (tmp_repo.repo / "f.txt").write_text("one\ntwo\n", encoding="utf-8")
    tmp_repo.add_all()
    tmp_repo.commit("improve f", "Nelke-Self-Improve: cycle c step 1")
    tmp_repo.checkout("main")

    req = db.list_review_requests(cycle_id=cid)[0]
    settings = Settings(nelke_home=str(db.path.parent))
    _resolve_review(settings, req["id"], "rejected", tmp_repo.repo)

    cycle = db.get_cycle(cid)
    assert cycle["status"] == "rejected"
    assert cycle["human_verdict"] == "rejected"
    # main untouched: the new file exists only on the kept branch
    assert not (tmp_repo.repo / "f.txt").exists()
    assert tmp_repo.branch_exists("improve/cli-reject-1")
    assert db.list_review_requests(cycle_id=cid)[0]["verdict"] == "rejected"


async def test_cycle_records_progress_events(tmp_repo, db):
    """Cycle events (incl. worker tool activity + live usage) are persisted."""
    events = []
    worker = [
        _usage_resp(tool_response("self_write", {"path": "memory/facts/work.md", "content": "# Work\n\ndone"})),
        _usage_resp(final_response("step done")),
        _usage_resp(tool_response("propose_cycle_complete", {})),
        _usage_resp(final_response("complete")),
    ]
    llm = driver_fake(worker=_scripted_worker(worker))
    engine = _engine(tmp_repo, db, FakeGovernance(), llm, on_event=lambda e: events.append(e.kind))
    result = await engine.run("add a memory lesson")
    assert result.merged
    rows = db.list_cycle_events(result.cycle_id)
    kinds = {r["kind"] for r in rows}
    # lifecycle + worker tool activity are all recorded
    assert {"cycle_start", "step_start", "gate", "commit", "step_ok", "merged"} <= kinds
    assert "agent_tool" in kinds
    # the cycle-worker streams its tokens (stream=True) -> agent_token events
    # are emitted, giving the web/TUI/CLI a live view of the agent's reply.
    assert "agent_token" in kinds
    # per-call usage is emitted + persisted to usage_events in real time
    assert "usage" in kinds
    usage_events = db.list_usage(cycle_id=result.cycle_id)
    assert any(u["total_tokens"] > 0 for u in usage_events)
    # seq strictly ordered
    seqs = [r["seq"] for r in rows]
    assert seqs == sorted(seqs)
    # payload carries the tool name for the streamed agent_tool
    tool_ev = next(r for r in rows if r["kind"] == "agent_tool")
    import json

    payload = json.loads(tool_ev["payload"])
    assert payload.get("tool") == "self_write"


async def test_cycles_tool_reports_progress(settings, tmp_repo, db):
    """The normal-mode agent's `cycles` tool reports what the cycle did."""
    from nelke.core.tools.cycles import CyclesTool

    llm = driver_fake(worker=_scripted_worker(_good_fix_plan()))
    engine = _engine(tmp_repo, db, FakeGovernance(), llm)
    result = await engine.run("add a memory lesson")
    assert result.merged

    tool = CyclesTool(db)
    res = await tool.execute(limit=40)
    assert res.ok
    assert "cycle" in res.output.lower()
    assert "self_write" in res.output
    assert "merged" in res.output.lower()


# --------------------------------------------------------------------------- #
# Phase B3 - cycle resilience
# --------------------------------------------------------------------------- #
async def test_cycle_refuses_dirty_main(tmp_repo, db):
    import pytest

    (tmp_repo.repo / "README.md").write_text("# dirty\n", encoding="utf-8")
    assert tmp_repo.has_changes()
    llm = driver_fake(worker=_scripted_worker([final_response("done")]))
    engine = _engine(tmp_repo, db, FakeGovernance(), llm)
    with pytest.raises(RuntimeError, match="uncommitted changes"):
        await engine.run("improve something")
    assert db.list_cycles() == []  # nothing recorded, nothing touched


async def test_cycle_marks_error_on_crash(tmp_repo, db):
    """A crashing worker must not leave the cycle stuck as `running`."""
    from conftest import FakeLLM

    class BoomLLM(FakeLLM):
        async def chat(self, messages, *, tools=None, model=None, temperature=None,
                       max_tokens=None, stream=False, on_token=None):
            raise RuntimeError("simulated provider stall")

    engine = _engine(tmp_repo, db, FakeGovernance(), BoomLLM())
    import pytest

    with pytest.raises(RuntimeError, match="simulated provider stall"):
        await engine.run("improve something")
    # the cycle was recorded and marked as failed, not left running
    rows = db.list_cycles()
    assert rows and rows[0]["status"] == "error"
    assert rows[0]["ended_at"] is not None


async def test_deps_sync_once_on_pyproject_change(tmp_repo, db):
    class RecordingRunner:
        def __init__(self) -> None:
            self.calls = []

        async def run(self, args, cwd, timeout):
            self.calls.append(list(args))
            return (0, "synced")

    class GovWithRunner(FakeGovernance):
        def __init__(self, runner) -> None:
            super().__init__()
            self.runner = runner

    runner = RecordingRunner()
    gov = GovWithRunner(runner)
    llm = driver_fake(worker=_scripted_worker([final_response("done")]))
    engine = _engine(tmp_repo, db, gov, llm)

    # no dependency file changed -> no sync
    assert not await engine._sync_dependencies_if_changed()
    assert runner.calls == []

    # pyproject.toml changed in the working tree -> uv sync runs once
    (tmp_repo.repo / "pyproject.toml").write_text("[project]\nname = 'x'\n", encoding="utf-8")
    assert await engine._sync_dependencies_if_changed()
    assert runner.calls == [["uv", "sync"]]

    # decision is cached -> never runs again
    assert not await engine._sync_dependencies_if_changed()
    assert runner.calls == [["uv", "sync"]]
