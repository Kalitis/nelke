"""Session-level degradation analysis: when should Nelke suggest improving itself?

A normal task that degrades (iteration cap hit without a final answer, repeated
tool errors, or an empty final answer) is treated as a signal that Nelke itself
could be improved. This module turns an :class:`~nelke.core.agent.AgentResult`
into a :class:`DegradationReport`; frontends use it to surface a
``nelke improve "<objective>"`` offer. It lives in the core so every frontend
behaves identically, and it deliberately avoids importing ``agent`` at module
level to keep the dependency graph acyclic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

DEFAULT_ERROR_THRESHOLD = 3
_MAX_OBJECTIVE_SLICE = 120


@dataclass
class DegradationReport:
    """Why a task degraded and the objective Nelke proposes for itself."""

    degraded: bool
    reasons: list[str] = field(default_factory=list)
    suggested_objective: str = ""

    def describe(self) -> str:
        return "; ".join(self.reasons) or "no degradation detected"


def analyze_degradation(
    result: Any,
    task: str,
    *,
    error_threshold: int = DEFAULT_ERROR_THRESHOLD,
) -> DegradationReport:
    """Return a report when ``result`` indicates Nelke struggled with ``task``.

    ``result`` is duck-typed so analysers can work on any agent-like result
    without importing :mod:`nelke.core.agent` (avoids a circular import).
    """
    reasons: list[str] = []
    stopped = getattr(result, "stopped", "answer") or "answer"
    answer = str(getattr(result, "answer", "") or "")
    tool_errors = int(getattr(result, "tool_errors", 0) or 0)
    tool_calls = int(getattr(result, "tool_calls", 0) or 0)
    iterations = int(getattr(result, "iterations", 0) or 0)

    if stopped == "max_iterations":
        reasons.append(
            f"hit the iteration cap ({iterations} iterations, {tool_calls} tool calls) "
            "without a final answer"
        )
    if tool_errors >= max(error_threshold, 1):
        reasons.append(f"{tool_errors} tool calls ended in errors")
    if stopped == "answer" and not answer.strip():
        reasons.append("finished with an empty answer")

    if not reasons:
        return DegradationReport(degraded=False)

    snippet = (task or "").strip().replace("\n", " ")[:_MAX_OBJECTIVE_SLICE]
    objective = f"make Nelke handle '{snippet}' more reliably"
    return DegradationReport(degraded=True, reasons=reasons, suggested_objective=objective)
