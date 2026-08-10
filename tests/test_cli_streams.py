"""CLI improve progress panel (Phase B4) — structural tests, no business logic."""

from __future__ import annotations

from nelke.core.cycle import CycleEvent
from nelke.frontends.cli import ImproveStream


def test_improve_stream_accumulates_and_renders():
    s = ImproveStream("objective: harden the cycle")
    s(CycleEvent(kind="cycle_start", message="branch improve/abc"))
    s(CycleEvent(kind="gate", message="[PASS] lint: clean\n[PASS] tests: 10 passed"))
    s(CycleEvent(kind="step_ok", message="step 1 committed, boot-check passed"))
    s(CycleEvent(kind="awaiting_human", message="cycle awaits human approval"))

    assert s.rows[0][0].startswith("[bold]Cycle started") or "Cycle" in s.rows[0][0]
    assert "10 passed" in s.gate_block
    assert len(s.rows) == 4
    # awaiting_human stops the live; the render tree is still buildable
    group = s._render()
    assert group is not None
    s.stop()  # no-op after the live was stopped


def test_improve_stream_default_label_for_unknown_kind():
    s = ImproveStream("x")
    s(CycleEvent(kind="mystery_event", message="hi"))
    assert s.rows[0][0] == "mystery_event"


def test_improve_stream_never_requires_start():
    s = ImproveStream("x")
    s.stop()  # live was never started -> must not raise
    assert s._render() is not None
