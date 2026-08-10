"""Read-only AI reviewer: the first gate before a cycle can be merged.

A separate ``Agent`` instance with read-only tools (self_read/glob/grep/git_diff)
reviews ``git diff main...branch`` for a cycle and returns a verdict that a human
gate then must confirm.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from nelke.core.agent import Agent
from nelke.core.gitops import GitRepo
from nelke.core.governance import Governance
from nelke.core.tools.selfedit import (
    GitDiffTool,
    SelfEditContext,
    SelfGlobTool,
    SelfGrepTool,
    SelfReadTool,
)

REVIEWER_PROMPT = """You are Nelke's AI review gate for its own self-improvement cycle.

Review the proposed changes (the attached diff) against the stated objective for:
- Correctness and obvious bugs.
- Test coverage: do existing tests still pass conceptually / is new behavior tested?
- Safety: no secrets, no destructive operations, no infinite loops, nothing that
  could brick the project (bad imports, broken syntax, unbounded processes).
- Alignment: do the changes actually serve the objective?

Return EXACTLY this format (no markdown fences):

VERDICT: APPROVE
SUMMARY: <one line summary>
COMMENTS:
- <issue or concern, one per line>
- none if nothing to fix

Only use APPROVE when the changes are safe and aligned; otherwise use REQUEST_CHANGES.
"""

_VERDICT_RE = re.compile(r"^\s*VERDICT\s*:\s*(.+)$", re.IGNORECASE | re.MULTILINE)
_SUMMARY_RE = re.compile(r"^\s*SUMMARY\s*:\s*(.+)$", re.IGNORECASE | re.MULTILINE)


@dataclass
class ReviewVerdict:
    verdict: str  # "approve" | "request_changes"
    summary: str = ""
    comments: str = ""

    @property
    def approved(self) -> bool:
        return self.verdict == "approve"


def parse_verdict(text: str) -> ReviewVerdict:
    m = _VERDICT_RE.search(text)
    if not m:
        # Conservative default: an approval must be explicit.
        approved = re.search(r"\bAPPROVE\b", text, re.IGNORECASE) is not None and \
            re.search(r"\bREQUEST_CHANGES\b|REQUEST-CHANGES", text, re.IGNORECASE) is None
        return ReviewVerdict(
            verdict="approve" if approved else "request_changes",
            summary="",
            comments=text.strip()[:3000],
        )
    raw = m.group(1).strip().upper()
    verdict = "approve" if "APPROVE" in raw else "request_changes"
    s = _SUMMARY_RE.search(text)
    summary = s.group(1).strip() if s else ""
    comments = _comments_section(text)
    return ReviewVerdict(verdict=verdict, summary=summary, comments=comments)


def _comments_section(text: str) -> str:
    content = text.strip()
    if "COMMENTS:" in content.upper():
        idx = content.upper().index("COMMENTS:")
        return content[idx + len("COMMENTS:"):].strip()[:3000]
    return content[:3000]


class Reviewer:
    def __init__(
        self,
        repo: GitRepo,
        llm: Any,
        *,
        name: str = "reviewer",
        system_prompt: str | None = None,
        base: str = "main",
        iteration_cap: int = 10,
    ) -> None:
        self.repo = repo
        self.llm = llm
        self.name = name
        self.base = base
        self.iteration_cap = iteration_cap
        ctx = SelfEditContext(
            repo=repo, governance=Governance(repo), repo_root=repo.repo, state={}
        )
        tools = [
            SelfReadTool(ctx),
            SelfGlobTool(ctx),
            SelfGrepTool(ctx),
            GitDiffTool(ctx),
        ]
        self.agent = Agent(
            name=name,
            system_prompt=system_prompt or REVIEWER_PROMPT,
            tools=tools,
            llm=llm,
            iteration_cap=iteration_cap,
            stream=False,
        )
        self.last_usage: dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "calls": 0}

    async def review(self, objective: str, diff: str | None = None) -> ReviewVerdict:
        if diff is None:
            diff = self.repo.diff(self.base, "HEAD")
        task = (
            f"OBJECTIVE: {objective}\n\n"
            f"The changes to review (git diff -u {self.base}...HEAD):\n"
            f"```diff\n{diff or '(empty diff)'}\n```\n\n"
            "Give your verdict in the required format."
        )
        result = await self.agent.run(task)
        self.last_usage = result.usage
        return parse_verdict(result.answer)
