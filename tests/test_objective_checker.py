"""Objective-gate tests: verdict parsing and integration with the cycle."""

from __future__ import annotations

from conftest import (
    FakeGovernance,
    driver_fake,
    final_response,
    scripted,
    tool_response,
)

from nelke.core.cycle import CycleEngine
from nelke.core.llm import LLMResponse
from nelke.core.objective_checker import ObjectiveChecker, parse_objective_verdict


def test_parse_achieved():
    v = parse_objective_verdict("VERDICT: ACHIEVED\nGAPS:\n- none")
    assert v.achieved
    assert v.verdict == "achieved"


def test_parse_not_achieved_with_gaps():
    v = parse_objective_verdict(
        "VERDICT: NOT_ACHIEVED\nGAPS:\n- the loop still terminates early\n- no test"
    )
    assert not v.achieved
    assert "loop still terminates early" in v.gaps
    assert "no test" in v.gaps


def test_parse_conservative_default_is_not_achieved():
    """Garbage / no explicit ACHIEVED must default to NOT achieved so the cycle
    keeps iterating rather than merging on an ambiguous signal."""
    v = parse_objective_verdict("the diff looks fine I guess")
    assert not v.achieved


def test_parse_not_achieved_disambiguates_from_achieved_keyword():
    """The word ACHIEVED inside NOT_ACHIEVED must not flip the verdict."""
    v = parse_objective_verdict("VERDICT: NOT_ACHIEVED\nGAPS:\n- missing")
    assert not v.achieved


async def test_checker_returns_achieved_on_approval():
    """A responder that says ACHIEVED yields an achieved verdict (happy path)."""

    class FakeLLM:
        async def chat(self, messages, **kw):
            return LLMResponse(content="VERDICT: ACHIEVED\nGAPS:\n- none")

    checker = ObjectiveChecker(FakeLLM())
    v = await checker.check("do the thing", "diff --git ...")
    assert v.achieved


def _engine_single(repo, db, gov, llm, **kw) -> CycleEngine:
    return CycleEngine(
        repo, db, gov, llm,
        human_approve=kw.pop("human", lambda req: True),
        max_steps=kw.pop("max_steps", 10),
        max_step_attempts=kw.pop("max_step_attempts", 3),
        max_gate_attempts=kw.pop("max_gate_attempts", 5),
        max_review_rounds=kw.pop("max_review_rounds", 3),
        on_event=kw.pop("on_event", None),
        mode="single",
        explore_budget=kw.pop("explore_budget", 6),
    )


async def test_cycle_does_not_merge_when_objective_not_met(tmp_repo, db):
    """When the objective checker says NOT_ACHIEVED, the cycle must NOT merge:
    workers get the gaps as feedback and iterate, and once the objective-round
    budget is exhausted the cycle ends as request-changes. This is the core fix
    for cycles that ended after one iteration with the objective unfinished."""
    events: list[str] = []

    def objective_resp(m, t):
        return final_response("VERDICT: NOT_ACHIEVED\nGAPS:\n- the widget is missing")

    llm = driver_fake(
        worker=scripted([
            # round 1: write partial work, do NOT propose (objective unmet)
            tool_response("self_write",
                          {"path": "memory/facts/p.md", "content": "# p\npartial"}),
            final_response("partial done"),
        ]),
        objective=objective_resp,
    )
    # objective stays NOT_ACHIEVED forever; with max_step_attempts=1 the cycle
    # must terminate as request-changes after the first objective miss.
    engine = _engine_single(
        tmp_repo, db, FakeGovernance(), llm,
        max_step_attempts=1,
        on_event=lambda e: events.append(e.kind),
    )
    result = await engine.run("add a memory lesson")
    assert result.status == "request-changes"
    assert "objective_not_met" in events
