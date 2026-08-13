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
- Test coverage: every new behavior (new file, function, branch, CLI command, API
  endpoint, …) MUST be covered by a passing test. If new code ships without a test,
  REQUEST_CHANGES and name the missing tests (e.g. "tests/test_<module>.py").
- Safety: no secrets, no destructive operations, no infinite loops, nothing that
  could brick the project (bad imports, broken syntax, unbounded processes).
- Alignment: do the changes actually serve the objective?

You may inspect the repo with the read-only tools to judge the diff in context.
Budget your exploration: a couple of targeted reads at most, then judge.

When you are ready to judge, you MUST end your turn with a plain-text answer
(no further tool calls) in EXACTLY this format (no markdown fences):

VERDICT: APPROVE
SUMMARY: <one line summary>
COMMENTS:
- <issue or concern, one per line>
- none if nothing to fix

Do NOT end your turn without a VERDICT line. If you ran out of tool iterations,
your next message MUST be the verdict above with no tool calls.
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
        # Conservative default: an approval must be explicit. An EMPTY or
        # tool-only answer (the reviewer ran out of iterations without a plain
        # verdict) must not silently count as approval — and must carry a
        # visible comment so the cycle log explains WHY it didn't merge.
        if not text.strip():
            return ReviewVerdict(
                verdict="request_changes",
                summary="reviewer produced no verdict",
                comments="The reviewer produced no verdict text. "
                         "Re-run the review or rework the change.",
            )
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
        iteration_cap: int = 6,
        temperature: float = 0.0,
    ) -> None:
        self.repo = repo
        self.llm = llm
        self.name = name
        self.base = base
        self.iteration_cap = iteration_cap
        self.temperature = temperature
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
            temperature=temperature,
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
        answer = result.answer
        # Fallback: if the reviewer agent spent all its tool iterations reading
        # files and never produced a plain verdict (no VERDICT: line), force one
        # final no-tool LLM call that MUST answer with the verdict format. This
        # is what keeps a review from silently failing as request_changes with
        # empty comments when the model gets lost in exploration.
        if "VERDICT:" not in answer.upper():
            answer = await self._force_verdict(objective, diff)
        self.last_usage = result.usage
        return parse_verdict(answer)

    async def _force_verdict(self, objective: str, diff: str | None) -> str:
        """One-shot, tool-less LLM call demanding a verdict right now.

        Used when the reviewer agent returned no VERDICT line (it either ran out
        of tool iterations or answered with prose only). The prompt leaves the
        model no room to defer — no tools are offered, so it must commit to a
        verdict in this single call.
        """
        messages = [
            {"role": "system", "content": self.agent.system_content()},
            {"role": "user", "content": (
                "You previously inspected the changes but did not return a verdict. "
                "Answer NOW with the verdict format and NOTHING ELSE — no tool calls, "
                "no exploration. Base it on the diff you already saw.\n\n"
                f"OBJECTIVE: {objective}\n\n"
                f"```diff\n{diff or '(empty diff)'}\n```\n\n"
                "Reply with:\nVERDICT: APPROVE|REQUEST_CHANGES\nSUMMARY: ...\n"
                "COMMENTS:\n- ..."
            )},
        ]
        resp = await self.llm.chat(messages, stream=False, temperature=self.temperature)
        return (resp.content if resp else "") or ""
