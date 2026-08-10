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
