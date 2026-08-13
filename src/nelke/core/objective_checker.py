"""Objective-gate: an independent check that a cycle's changes actually achieve
the stated objective.

Distinct from the AI reviewer (which judges code quality / test coverage /
safety), this answers a single question: looking at the diff, is the OBJECTIVE
met? If not, the cycle must keep iterating instead of going to review/merge on
a reviewer's quality approval alone. The conservative default is NOT_ACHIEVED:
an explicit ACHIEVED verdict is required to proceed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

OBJECTIVE_CHECKER_PROMPT = """You are an independent gate judging whether a \
self-improvement cycle's changes actually achieve the stated objective.

You are given:
- OBJECTIVE: what the cycle was supposed to accomplish.
- The combined diff of all changes on the branch.

Judge ONLY whether the objective is met by these changes — not code style, not \
test coverage (a separate reviewer handles those). Be concrete and skeptical: \
if the objective names a behaviour, the diff must implement it; if it names a \
bug, the diff must fix it; if the objective is vague, require evidence the \
stated problem is addressed.

Return EXACTLY this format (no markdown fences):

VERDICT: ACHIEVED
GAPS:
- none

Use ACHIEVED only when the changes concretely satisfy the objective. Otherwise \
use NOT_ACHIEVED and list each specific gap under GAPS (one per line, naming \
what is missing or wrong)."""


_VERDICT_RE = re.compile(r"^\s*VERDICT\s*:\s*(.+)$", re.IGNORECASE | re.MULTILINE)


@dataclass
class ObjectiveVerdict:
    """Result of an objective check."""

    verdict: str  # "achieved" | "not_achieved"
    gaps: str = ""

    @property
    def achieved(self) -> bool:
        return self.verdict == "achieved"


def parse_objective_verdict(text: str) -> ObjectiveVerdict:
    """Parse the checker's text response. Conservative default: if no explicit
    ACHIEVED is found, treat the objective as NOT met so the cycle keeps
    iterating rather than merging unfinished work."""
    m = _VERDICT_RE.search(text)
    if not m:
        achieved = re.search(r"\bACHIEVED\b", text, re.IGNORECASE) is not None and \
            re.search(r"\bNOT[_ ]?ACHIEVED\b", text, re.IGNORECASE) is None
        return ObjectiveVerdict(
            verdict="achieved" if achieved else "not_achieved",
            gaps=text.strip()[:3000],
        )
    raw = m.group(1).strip().upper()
    verdict = "achieved" if "ACHIEVED" in raw and "NOT" not in raw else "not_achieved"
    gaps = _gaps_section(text)
    return ObjectiveVerdict(verdict=verdict, gaps=gaps)


def _gaps_section(text: str) -> str:
    content = text.strip()
    upper = content.upper()
    if "GAPS:" in upper:
        idx = upper.index("GAPS:")
        return content[idx + len("GAPS:"):].strip()[:3000]
    return content[:3000]


class ObjectiveChecker:
    """One-shot LLM call that judges objective achievement from the diff.

    Modelled on :class:`Reviewer` but with no tools: it works purely from the
    objective + diff text, keeping the check cheap and independent of the
    repo-exploring reviewer agent.
    """

    def __init__(
        self,
        llm: Any,
        *,
        temperature: float = 0.0,
        system_prompt: str | None = None,
    ) -> None:
        self.llm = llm
        self.temperature = temperature
        self.system_prompt = system_prompt or OBJECTIVE_CHECKER_PROMPT
        self.last_usage: dict[str, int] = {
            "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "calls": 0,
        }

    async def check(self, objective: str, diff: str | None = None) -> ObjectiveVerdict:
        task = (
            f"OBJECTIVE: {objective}\n\n"
            f"The changes to judge:\n"
            f"```diff\n{diff or '(empty diff)'}\n```\n\n"
            "Is the objective achieved? Answer in the required format."
        )
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": task},
        ]
        resp = await self.llm.chat(
            messages, stream=False, temperature=self.temperature
        )
        self.last_usage = (resp.usage if resp else None) or dict(self.last_usage)
        raw = (resp.content if resp else "") or ""
        return parse_objective_verdict(raw)
