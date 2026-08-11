"""The self-improvement cycle engine (see plan §8).

Work loop (self-edit tools on, cwd = repo) -> governance gate -> commit ->
boot-check (auto-revert on failure) -> AI review -> human gate -> merge ``--no-ff``.
Every transition is recorded in SQLite (``cycles``/``cycle_steps``/
``review_requests``) and reported to the active frontend via ``CycleEvent``.
"""

from __future__ import annotations

import inspect
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from nelke.core.agent import Agent
from nelke.core.db import Database, _now, new_id
from nelke.core.gitops import GitError, GitRepo
from nelke.core.governance import Governance
from nelke.core.llm import ToolCallback
from nelke.core.memory import MemoryStore
from nelke.core.reviewer import Reviewer, ReviewVerdict
from nelke.core.tools.memory import MemoryWriteTool, RecallTool
from nelke.core.tools.selfedit import (
    BootCheckTool,
    GitBranchInfoTool,
    GitCommitTool,
    GitRevertTool,
    ProposeCycleCompleteTool,
    RunLintTool,
    RunTestsTool,
    RunTypecheckTool,
    SelfEditContext,
    SelfEditTool,
    SelfGlobTool,
    SelfGrepTool,
    SelfReadTool,
    SelfWriteTool,
)

CYCLE_WORK_PROMPT = """You are Nelke operating on your OWN repository. Your job is to
improve the repository to satisfy the user objective. You have self-edit tools scoped
to the repo root, memory tools (recall/memory_write under memory/), and git tools.
The gates (run_tests / run_lint / run_typecheck / boot_check) and commits are run by
the cycle engine automatically after you finish — do not call them yourself.

Workflow:
1. Explore the repo (self_read/self_glob/self_grep) and recall relevant memory.
2. Make focused edits toward the objective; keep existing tests green; add or adjust
   tests for new behavior when relevant.
3. When you have a coherent improvement ready, you are done for this step — do NOT
   loop infinitely; the engine commits, gates and boot-checks, then feeds results back.
4. Wait for gate feedback. If the gate failed or a commit was reverted, fix the
   specific problems reported and try again with minimal changes.
5. When the objective is fully achieved, call propose_cycle_complete.
Avoid broken syntax/imports: any commit that crashes Nelke is reverted automatically."""


@dataclass
class CycleEvent:
    kind: str
    message: str = ""
    data: dict[str, Any] = field(default_factory=dict)

    @property
    def cycle_id(self) -> str:
        return str(self.data.get("cycle_id", ""))

    @property
    def step(self) -> int | None:
        val = self.data.get("step")
        return int(val) if val is not None else None

    @property
    def progress(self) -> tuple[int, int] | None:
        """(completed_steps, total_steps) for progress rendering."""
        step = self.data.get("step")
        total = self.data.get("total_steps")
        if step is None or total is None:
            return None
        return int(step), int(total)


@dataclass
class HumanReviewRequest:
    cycle_id: str
    objective: str
    branch: str
    diff: str
    ai_verdict: ReviewVerdict


@dataclass
class CycleResult:
    cycle_id: str
    objective: str
    branch: str
    status: str
    steps: int = 0
    ai_verdict: str = ""
    human_verdict: str = ""
    message: str = ""

    @property
    def merged(self) -> bool:
        return self.status == "merged"


def slugify(text: str, limit: int = 24) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-").rstrip("-")
    return (slug or "task")[:limit].rstrip("-")


def merge_cycle_branch(repo: GitRepo, branch: str, cycle_id: str = "unknown") -> str:
    """Checkout ``main`` and merge ``branch`` ``--no-ff`` as a Nelke co-authored merge.

    Shared by :meth:`CycleEngine.run` and the CLI ``review approve`` resolution
    path so both behave identically. Raises :class:`GitError` on a conflict;
    callers are responsible for persisting the DB state transition.
    """
    repo.checkout("main")
    repo.merge_no_ff(
        branch,
        f"Merge branch {branch!r} (cycle {cycle_id})",
        "Co-authored-by: Nelke <nelke@local>",
    )
    return branch


class CycleEngine:
    def __init__(
        self,
        repo: GitRepo,
        db: Database,
        governance: Governance,
        llm: Any,
        *,
        on_event: Callable[[CycleEvent], Any] | None = None,
        human_approve: Callable[[HumanReviewRequest], bool | Awaitable[bool]] | None = None,
        max_steps: int = 30,
        max_step_attempts: int = 3,
        max_review_rounds: int = 3,
        on_token: ToolCallback = None,
        on_usage: Callable[[dict[str, Any]], Any] | None = None,
        agent_temperature: float = 0.0,
    ) -> None:
        self.repo = repo
        self.db = db
        self.governance = governance
        self.llm = llm
        self.on_event = on_event
        self.human_approve = human_approve
        self.max_steps = max_steps
        self.max_step_attempts = max_step_attempts
        self.max_review_rounds = max_review_rounds
        self.on_token = on_token
        self.on_usage = on_usage
        self.agent_temperature = agent_temperature
        self._synced = False

    def _emit(self, kind: str, message: str = "", **data: Any) -> None:
        if self.on_event is not None:
            self.on_event(CycleEvent(kind=kind, message=message, data=data))

    async def _sync_dependencies_if_changed(self) -> bool:
        """Run ``uv sync`` when ``pyproject.toml``/``uv.lock`` changed this cycle.

        The decision is cached (at most one sync per cycle) so new dependencies
        are installed before the gate without slowing every step. A missing
        runner (e.g. a fake governance in tests) short-circuits safely.
        """
        if self._synced:
            return False
        if not self.repo.paths_changed(["pyproject.toml", "uv.lock"]):
            return False
        runner = getattr(self.governance, "runner", None)
        self._synced = True
        if runner is None:
            return False
        code, out = await runner.run(["uv", "sync"], str(self.repo.repo), 600)
        if code != 0:
            self._emit("deps_failed", f"uv sync failed: {out[-500:]}")
        else:
            self._emit("deps_synced", "pyproject.toml changed; ran `uv sync`")
        return True

    def _build_working_agent(self, ctx: SelfEditContext, memory: MemoryStore) -> Agent:
        tools = [
            SelfReadTool(ctx),
            SelfWriteTool(ctx),
            SelfEditTool(ctx),
            SelfGlobTool(ctx),
            SelfGrepTool(ctx),
            RecallTool(memory),
            MemoryWriteTool(memory),
            RunLintTool(ctx),
            RunTypecheckTool(ctx),
            RunTestsTool(ctx),
            BootCheckTool(ctx),
            GitBranchInfoTool(ctx),
            GitCommitTool(ctx),
            GitRevertTool(ctx),
            ProposeCycleCompleteTool(ctx),
        ]
        return Agent(
            name="cycle-worker",
            system_prompt=CYCLE_WORK_PROMPT,
            tools=tools,
            llm=self.llm,
            iteration_cap=40,
            stream=True,
            on_token=self.on_token,
            memory_index=memory.index_text() or None,
            temperature=self.agent_temperature,
        )

    def _result(
        self,
        cycle_id: str,
        objective: str,
        branch: str,
        status: str,
        verdict: ReviewVerdict,
        human: str = "",
        steps: int = 0,
        message: str = "",
    ) -> CycleResult:
        return CycleResult(
            cycle_id=cycle_id,
            objective=objective,
            branch=branch,
            status=status,
            steps=steps,
            ai_verdict=verdict.verdict if verdict else "",
            human_verdict=human,
            message=message,
        )

    async def run(
        self,
        objective: str,
        *,
        human_approve: Callable[[HumanReviewRequest], bool | Awaitable[bool]] | None = None,
    ) -> CycleResult:
        repo = self.repo
        if not repo.is_repo():
            raise RuntimeError(f"not a git repo: {repo.repo}")
        human_gate = human_approve if human_approve is not None else self.human_approve
        objective = objective.strip()
        cycle_id = new_id()
        branch = f"improve/{cycle_id}-{slugify(objective)}"

        if repo.has_changes():
            # A dirty tree would be silently carried onto the cycle branch and
            # committed as part of the cycle. Never do that: abort with guidance.
            raise RuntimeError(
                "refusing to start a cycle: the working tree has uncommitted changes. "
                "Commit or stash them first, then re-run the cycle."
            )
        if repo.current_branch() != "main":
            repo.checkout("main")
        repo.checkout_new_branch(branch, base="main")
        self._synced = False
        self.db.create_cycle(objective, branch, cycle_id=cycle_id)
        try:
            return await self._run_impl(repo, objective, branch, cycle_id, human_gate)
        except Exception:  # noqa: BLE001 - never leave a cycle stuck as `running`
            try:
                self.db.update_cycle(cycle_id, status="error", ended_at=_now())
                self._emit("cycle_error", "cycle crashed", cycle_id=cycle_id, branch=branch)
            except Exception:  # noqa: BLE001 - persistence must never mask the crash
                pass
            raise

    async def _run_impl(
        self,
        repo: GitRepo,
        objective: str,
        branch: str,
        cycle_id: str,
        human_gate: Callable[[HumanReviewRequest], bool | Awaitable[bool]] | None,
    ) -> CycleResult:
        state: dict[str, Any] = {"propose_complete": False}
        step_no = 0
        ctx = SelfEditContext(
            repo=repo,
            governance=self.governance,
            repo_root=repo.repo,
            state=state,
            cycle_id_provider=lambda: cycle_id,
            step_provider=lambda: step_no,
        )
        memory = MemoryStore(repo.repo / "memory")
        agent = self._build_working_agent(ctx, memory)

        def emit(kind: str, message: str = "", **data: Any) -> None:
            """Emit + persist one cycle event; ``seq``-ordered rows feed progress UIs."""
            data.setdefault("cycle_id", cycle_id)
            data.setdefault("total_steps", self.max_steps)
            try:
                self.db.add_cycle_event(cycle_id, kind, message, data)
            except Exception:  # noqa: BLE001 - persistence must never break the cycle
                pass
            self._emit(kind, message, **data)

        # Route agent tool activity of the cycle-worker to progress events so
        # the web card / TG / TUI can stream what the agent is actually doing.
        def on_worker_token(tok: str) -> None:
            if self.on_token is not None:
                self.on_token(tok)
            emit("agent_token", "", token=tok)

        def on_worker_tool(name: str, args: dict[str, Any]) -> None:
            emit("agent_tool", f"tool {name}", tool=name, args=args, step=step_no)

        def on_worker_tool_result(name: str, args: dict[str, Any], result: str) -> None:
            emit("agent_tool_result", "", tool=name, snippet=result[:400], step=step_no)

        def on_worker_usage(usage: dict[str, Any]) -> None:
            """Persist each LLM call's usage live and forward it to the frontend."""
            try:
                if usage.get("total_tokens"):
                    self.db.add_usage(usage, cycle_id=cycle_id)
            except Exception:  # noqa: BLE001 - persistence must never break the cycle
                pass
            emit(
                "usage", "",
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
                total_tokens=usage.get("total_tokens", 0),
            )
            if self.on_usage is not None:
                try:
                    self.on_usage(dict(usage))
                except Exception:  # noqa: BLE001 - UI updates must never break the cycle
                    pass

        agent.on_token = on_worker_token
        agent.on_tool = on_worker_tool
        agent.on_tool_result = on_worker_tool_result
        agent.on_usage = on_worker_usage

        emit("cycle_start", f"branch {branch}", cycle_id=cycle_id, branch=branch, objective=objective)

        feedback = ""
        first_run = True
        committed_any = False
        verdict: ReviewVerdict | None = None

        while True:
            # ------------------- WORK PHASE -------------------------------
            step_cap_hit = False
            while step_no < self.max_steps:
                gate_passed = False
                pending_propose = False
                for _ in range(self.max_step_attempts):
                    step_no += 1
                    state["propose_complete"] = False
                    task = objective if not feedback else f"{objective}\n\n[feedback]\n{feedback}"
                    emit("step_start", f"step {step_no}", step=step_no)
                    await agent.run(task, reset=first_run)
                    first_run = False
                    pending_propose = bool(state["propose_complete"])

                    await self._sync_dependencies_if_changed()
                    gate = await self.governance.gate()
                    emit("gate", gate.describe(), step=step_no, passed=gate.passed)
                    if gate.passed:
                        feedback = ""
                        gate_passed = True
                        break
                    pending_propose = False
                    feedback = (
                        "The governance gate rejected your changes:\n"
                        + gate.describe()
                        + "\nFix these exact problems with minimal changes and continue."
                    )

                if not gate_passed:
                    if repo.has_changes():
                        repo.stash_all()
                    self.db.add_step(cycle_id, step_no, None, "failed-gate", feedback[:500])
                    self.db.update_cycle(cycle_id, status="failed-gate", ended_at=_now())
                    emit("cycle_error", "gate could not be satisfied", feedback=feedback[:2000])
                    return self._result(cycle_id, objective, branch, "failed-gate",
                                        ReviewVerdict("request_changes"),
                                        steps=step_no, message=feedback[:500])

                if not repo.has_changes():
                    emit("idle", "no changes after a green step")
                    break

                repo.add_all()
                sha = repo.commit(
                    f"Cycle {cycle_id} step {step_no}: {objective[:80]}",
                    f"Nelke-Self-Improve: cycle {cycle_id} step {step_no}",
                )
                committed_any = True
                self.db.add_step(cycle_id, step_no, sha, "committed", objective[:200])
                emit("commit", f"committed {sha}", step=step_no, sha=sha)

                boot = await self.governance.boot_check()
                if not boot.ok and not boot.skipped:
                    repo.revert_commit(sha)
                    self.db.add_step(cycle_id, step_no, sha, "failed-boot", boot.message[:500])
                    emit("boot_check_failed", f"reverted {sha}: {boot.message[:200]}", step=step_no, sha=sha)
                    feedback = (
                        "Your last commit broke the boot check and was reverted automatically. "
                        "It must import cleanly and pass nelke.boot_check(). Rework the change."
                    )
                    continue
                self.db.add_step(cycle_id, step_no, sha, "ok", "boot check passed")
                emit("step_ok", f"step {step_no} committed, boot-check passed", step=step_no)

                if pending_propose:
                    break
                feedback = ""  # clean committed step clears stale feedback
            else:
                step_cap_hit = True

            final_diff = repo.diff("main", branch)
            if not committed_any and not final_diff.strip():
                self.db.update_cycle(cycle_id, status="no-changes", ended_at=_now())
                emit("cycle_error", "no changes produced")
                return self._result(cycle_id, objective, branch, "no-changes",
                                    ReviewVerdict("request_changes"),
                                    steps=step_no, message="agent made no changes")

            # ------------------- AI REVIEW --------------------------------
            rounds = 0
            review_feedback_pending = False
            while True:
                rounds += 1
                reviewer = Reviewer(repo, self.llm, base="main",
                                    temperature=self.agent_temperature)
                verdict = await reviewer.review(objective, final_diff)
                self.db.add_usage(reviewer.last_usage, cycle_id=cycle_id)
                self.db.create_review_request(cycle_id, "ai", verdict=verdict.verdict, comments=verdict.comments)
                self.db.update_cycle(cycle_id, ai_verdict=verdict.verdict)
                emit("ai_review", f"AI: {verdict.verdict}", verdict=verdict.verdict,
                     comments=verdict.comments[:800], round=rounds)
                if verdict.approved:
                    review_feedback_pending = False
                    break
                if rounds < self.max_review_rounds and step_no < self.max_steps:
                    feedback = "AI reviewer requested changes:\n" + verdict.comments
                    review_feedback_pending = True
                    break
                self.db.update_cycle(cycle_id, status="request-changes", ended_at=_now())
                emit("cycle_error", "reviewer still requests changes", comments=verdict.comments[:800])
                return self._result(cycle_id, objective, branch, "request-changes", verdict,
                                    steps=step_no, message=verdict.comments[:500])

            if review_feedback_pending:
                if step_cap_hit:
                    self.db.update_cycle(cycle_id, status="request-changes", ended_at=_now())
                    return self._result(cycle_id, objective, branch, "request-changes", verdict,
                                        steps=step_no, message=verdict.comments[:500])
                emit("review_feedback", "resuming work with reviewer feedback")
                continue  # back to the work phase

            break

        # ------------------- HUMAN GATE -----------------------------------
        assert verdict is not None
        human_request = HumanReviewRequest(
            cycle_id=cycle_id, objective=objective, branch=branch,
            diff=final_diff, ai_verdict=verdict,
        )
        self.db.create_review_request(cycle_id, "human", verdict="pending",
                                      comments=f"AI: {verdict.verdict}\n{verdict.comments}".strip())
        emit("awaiting_human", "cycle awaits human approval", branch=branch)

        if human_gate is None:
            self.db.update_cycle(cycle_id, status="awaiting-human", ended_at=_now())
            emit("human_pending", "no human gate attached — leaving branch for review")
            return self._result(cycle_id, objective, branch, "awaiting-human", verdict, steps=step_no)

        decision = human_gate(human_request)
        if inspect.isawaitable(decision):
            decision = await decision  # type: ignore[misc]
        if not decision:
            self.db.update_cycle(cycle_id, status="rejected", human_verdict="rejected", ended_at=_now())
            emit("human_rejected", "human rejected cycle", branch=branch)
            return self._result(cycle_id, objective, branch, "rejected", verdict, human="rejected", steps=step_no)

        # approved -> merge --no-ff into main (co-authored by Nelke)
        try:
            merge_cycle_branch(repo, branch, cycle_id=cycle_id)
        except GitError as exc:
            self.db.update_cycle(cycle_id, status="merge-conflict", human_verdict="approved", ended_at=_now())
            emit("cycle_error", "merge failed", error=str(exc))
            return self._result(cycle_id, objective, branch, "merge-conflict", verdict,
                                human="approved", steps=step_no, message=str(exc)[:500])

        self.db.update_cycle(cycle_id, status="merged", human_verdict="approved", ended_at=_now())
        emit("merged", "cycle merged into main", branch=branch)
        return self._result(cycle_id, objective, branch, "merged", verdict,
                            human="approved", steps=step_no)
