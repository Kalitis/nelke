"""Degradation analysis: when does Nelke propose improving itself (Phase B1)."""

from __future__ import annotations

from nelke.core.agent import AgentResult
from nelke.core.session_analyzer import analyze_degradation


def _result(**kw) -> AgentResult:
    defaults = dict(
        answer="done",
        iterations=2,
        tool_calls=0,
        tool_errors=0,
        stopped="answer",
    )
    defaults.update(kw)
    return AgentResult(**defaults)


def test_success_is_not_degraded():
    report = analyze_degradation(_result(answer="all good"), "tidy up the repo")
    assert not report.degraded
    assert report.reasons == []
    assert report.suggested_objective == ""


def test_iteration_cap_triggers_degradation():
    report = analyze_degradation(
        _result(answer="", iterations=20, tool_calls=14, stopped="max_iterations"),
        "search the web for X",
    )
    assert report.degraded
    assert any("iteration cap" in r for r in report.reasons)
    assert "X" in report.suggested_objective
    assert "nelke improve" not in report.suggested_objective


def test_empty_answer_triggers_degradation():
    report = analyze_degradation(_result(answer="   "), "do the thing")
    assert report.degraded
    assert any("empty answer" in r for r in report.reasons)


def test_tool_errors_below_threshold_are_fine():
    report = analyze_degradation(_result(answer="ok", tool_errors=2), "task")
    assert not report.degraded


def test_repeated_tool_errors_trigger_degradation():
    report = analyze_degradation(_result(answer="ok", tool_errors=3), "task")
    assert report.degraded
    assert any("3 tool calls ended in errors" in r for r in report.reasons)


def test_custom_threshold():
    report = analyze_degradation(
        _result(answer="ok", tool_errors=4), "task", error_threshold=5
    )
    assert not report.degraded


def test_suggested_objective_slices_long_tasks():
    long_task = "do " * 100
    report = analyze_degradation(
        _result(answer="", stopped="max_iterations"), long_task
    )
    assert len(report.suggested_objective) < 200
    assert report.suggested_objective.startswith("make Nelke handle")
