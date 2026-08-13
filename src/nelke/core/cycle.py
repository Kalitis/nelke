"""The self-improvement cycle engine (see plan §8).

Work loop (self-edit tools on, cwd = repo) -> governance gate -> commit ->
boot-check (auto-revert on failure) -> AI review -> human gate -> merge ``--no-ff``.
Every transition is recorded in SQLite (``cycles``/``cycle_steps``/
``review_requests``) and reported to the active frontend via ``CycleEvent``.
"""

from __future__ import annotations

import asyncio
import inspect
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Literal, cast

from nelke.core.agent import Agent
from nelke.core.db import Database, _now, new_id
from nelke.core.gitops import GitError, GitRepo
from nelke.core.governance import Governance
from nelke.core.llm import ToolCallback
from nelke.core.memory import MemoryStore
from nelke.core.objective_checker import ObjectiveChecker
from nelke.core.planner import TaskSpec, plan_tasks
from nelke.core.reviewer import Reviewer, ReviewVerdict
from nelke.core.tools.memory import MemoryListTool, MemoryShowTool, MemoryWriteTool, RecallTool
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
to the repo root, memory tools (recall/memory_show/memory_list/memory_write under
memory/), and git tools.
The gates (run_tests / run_lint / run_typecheck / boot_check) and commits are run by
the cycle engine automatically after you finish — do not call them yourself.

Workflow:
1. You have a TIGHT exploration budget (a small number of read/glob/grep/recall
   calls). Open only the file(s) you are about to edit; after ~2-3 reads you
   MUST start editing — do NOT explore the whole codebase first.
2. Make focused edits toward the objective, and ALWAYS add or update tests for any
   new behavior in the same step:
   - Every new or changed src module (src/**/*.py) needs a matching
     tests/test_<module>.py that exercises the new behavior.
   - New functions/methods/branches added to an existing module need a new test
     case in that module's test file (create the test file if missing).
   Writing tests is MANDATORY, not optional: the governance gate now rejects code
     shipped without a matching test file, so you will be sent back to add them.
3. When you have a coherent improvement ready, you are done for this step — do NOT
   loop infinitely; the engine commits, gates and boot-checks, then feeds results back.
4. Wait for gate feedback. If the gate failed or a commit was reverted, fix the
   specific problems reported and try again with minimal changes.
5. When the objective is fully achieved, call propose_cycle_complete.

CRITICAL: reading files is NOT progress — the cycle fails as no-changes if you
never edit. WRITE CODE. If a read/recall tool returns "EXPLORATION BUDGET
EXHAUSTED", stop reading entirely and use self_write/self_edit for the rest of
your turn.
Avoid broken syntax/imports: any commit that crashes Nelke is reverted automatically."""


CYCLE_WORKER_PROMPT = """You are one of several Nelke worker agents operating IN
PARALLEL on the same repository to satisfy a slice of a larger objective. Other
workers are editing DIFFERENT files at the same time — stay within your lane and
do not touch files outside your assigned scope.

You have self-edit tools scoped to the repo root (self_read/self_write/self_edit/
self_glob/self_grep), memory tools (recall/memory_show/memory_list/memory_write
under memory/), and read-only gates (run_lint/run_typecheck/run_tests/boot_check
to validate your own edits). You DO NOT commit, revert or propose cycle
completion — the cycle engine handles commits and gates centrally after all
workers finish.

Workflow:
1. Read your assigned task (title + detail). You have a TIGHT exploration budget
   (a small number of read/glob/grep/recall calls). After ~2-3 reads you MUST
   start editing — do NOT explore the whole codebase first. Open only the file
   you are about to edit, then edit it.
2. Make focused edits toward the task, and ALWAYS add or update tests for any new
   behavior in the same pass:
   - Every new or changed src module (src/**/*.py) needs a matching
     tests/test_<module>.py that exercises the new behavior.
   - New functions/methods/branches added to an existing module need a new test
     case in that module's test file (create the test file if missing).
   Writing tests is MANDATORY, not optional: the governance gate rejects code
   shipped without a matching test file, so you will be sent back to add them.
3. Optionally run the gates to self-check your edits.
4. When your slice is complete, stop and return a short summary of what you did.

CRITICAL: exploration alone produces NO changes and the cycle will fail as
no-changes if you never edit. Reading files is not progress — WRITE CODE.
If a read tool returns "EXPLORATION BUDGET EXHAUSTED", stop reading entirely
and use self_write/self_edit for the rest of your turn.

Do NOT loop infinitely. Do NOT edit files outside your task's scope. Other
workers rely on the shared working tree staying consistent with your slice."""


# Read-only tools counted against a worker's exploration budget. These explore
# the repo without changing it; a worker that only calls these never edits.
_READ_ONLY_TOOLS = frozenset({
    "self_read", "self_glob", "self_grep", "recall",
    "memory_show", "memory_list", "git_diff",
})


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
    project_id: str = ""

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
        max_gate_attempts: int = 5,
        max_review_rounds: int = 3,
        on_token: ToolCallback = None,
        on_usage: Callable[[dict[str, Any]], Any] | None = None,
        agent_temperature: float = 0.0,
        mode: Literal["single", "parallel"] = "single",
        max_workers: int = 6,
        max_steps_per_worker: int | None = None,
        explore_budget: int = 12,
    ) -> None:
        self.repo = repo
        self.db = db
        self.governance = governance
        self.llm = llm
        self.on_event = on_event
        self.human_approve = human_approve
        self.max_steps = max_steps
        self.max_step_attempts = max_step_attempts
        self.max_gate_attempts = max_gate_attempts
        self.max_review_rounds = max_review_rounds
        self.on_token = on_token
        self.on_usage = on_usage
        self.agent_temperature = agent_temperature
        self.mode = mode
        self.max_workers = max(1, max_workers)
        # Per-worker cap on read-only tool calls per round. 0 disables it.
        self.explore_budget = max(0, explore_budget)
        # Per-worker step cap. Default keeps the total roughly inside `max_steps`
        # so a parallel cycle does not exceed the legacy single-worker budget.
        self.max_steps_per_worker = max_steps_per_worker or max(
            3, max_steps // self.max_workers
        )
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
            # Memory read tools share the same exploration budget as self_* reads.
            RecallTool(memory, explore_nudge=ctx.bump_explore),
            MemoryShowTool(memory, explore_nudge=ctx.bump_explore),
            MemoryListTool(memory, explore_nudge=ctx.bump_explore),
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
            memory_location=str(memory.memory_dir),
            temperature=self.agent_temperature,
        )

    def _build_worker_agent(
        self,
        ctx: SelfEditContext,
        memory: MemoryStore,
        worker_id: str,
        task: TaskSpec,
        on_token: ToolCallback | None,
        on_usage: Callable[[dict[str, Any]], Any] | None,
    ) -> Agent:
        """Build one parallel worker agent for a single planner slice.

        Unlike the legacy single worker, parallel workers do NOT get the git
        commit/revert tools or ``propose_cycle_complete``: they only edit files
        in their assigned scope. The engine commits the combined diff centrally
        after all workers finish. A per-worker system prompt names the slice so
        the model stays in its lane.
        """
        tools = [
            SelfReadTool(ctx),
            SelfWriteTool(ctx),
            SelfEditTool(ctx),
            SelfGlobTool(ctx),
            SelfGrepTool(ctx),
            # Memory read tools share the same exploration budget as self_* reads
            # so a worker can't dodge the cap by switching to recall/memory_show.
            RecallTool(memory, explore_nudge=ctx.bump_explore),
            MemoryShowTool(memory, explore_nudge=ctx.bump_explore),
            MemoryListTool(memory, explore_nudge=ctx.bump_explore),
            MemoryWriteTool(memory),
            RunLintTool(ctx),
            RunTypecheckTool(ctx),
            RunTestsTool(ctx),
            BootCheckTool(ctx),
        ]
        slice_prompt = (
            f"{CYCLE_WORKER_PROMPT}\n\n"
            f"Your assigned task:\nTitle: {task.title}\nDetail: {task.detail}\n"
            f"Worker id: {worker_id}"
        )
        return Agent(
            name=f"cycle-worker-{worker_id}",
            system_prompt=slice_prompt,
            tools=tools,
            llm=self.llm,
            iteration_cap=max(5, self.max_steps_per_worker * 2),
            stream=True,
            on_token=on_token,
            on_usage=on_usage,
            memory_index=memory.index_text() or None,
            memory_location=str(memory.memory_dir),
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

    def _build_resume_hint(
        self,
        repo: GitRepo,
        branch: str,
        read_cache: dict[str, str] | None = None,
    ) -> str:
        """Append-only context handed to workers/agent when resuming a round.

        Workers restart from scratch each round (a fresh agent), so without this
        they re-read every file they already explored. The hint tells them what
        is already changed on the branch and which files are already understood,
        so they go straight to the remaining work instead of duplicating prior
        exploration. Best-effort: any git error just yields an empty hint.
        """
        try:
            changed = repo.changed_files("main", branch)
        except Exception:  # noqa: BLE001 - resume hint is best-effort
            changed = []
        parts: list[str] = []
        if changed:
            parts.append("Already changed on this branch (do NOT redo these):")
            parts.extend(f"- {f}" for f in changed[:50])
        if read_cache:
            # Only mention files a worker actually read (not every cache key is
            # a real read, but the approximation is good enough for a hint).
            studied = sorted(read_cache.keys())[:50]
            if studied:
                parts.append("Files already explored (do NOT re-read; recall from here):")
                parts.extend(f"- {f}" for f in studied)
        return "\n".join(parts)

    async def run(
        self,
        objective: str,
        *,
        human_approve: Callable[[HumanReviewRequest], bool | Awaitable[bool]] | None = None,
        project_id: str | None = None,
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
        # Reap orphaned cycles: any row still `running` belongs to a cycle that
        # crashed or was killed (a new run cannot start while another is in
        # flight), so mark it failed. This also frees the UI/API from stuck
        # spinners left by a Ctrl-C'd or OOM-killed run.
        self._reap_orphan_running_cycles()
        if repo.current_branch() != "main":
            repo.checkout("main")
        repo.checkout_new_branch(branch, base="main")
        self._synced = False
        self.db.create_cycle(objective, branch, cycle_id=cycle_id, project_id=project_id)
        completed = False
        try:
            if self.mode == "parallel":
                result = await self._run_impl_parallel(repo, objective, branch, cycle_id, human_gate)
            else:
                result = await self._run_impl(repo, objective, branch, cycle_id, human_gate)
            completed = True
            # Attribute the cycle to its project in the result so callers/UI can
            # link it back without a separate DB lookup.
            result.project_id = project_id or ""
            return result
        except BaseException:  # noqa: BLE001 - catch CancelledError/KeyboardInterrupt too
            try:
                self.db.update_cycle(cycle_id, status="error", ended_at=_now())
                self._emit("cycle_error", "cycle crashed", cycle_id=cycle_id, branch=branch)
            except Exception:  # noqa: BLE001 - persistence must never mask the crash
                pass
            raise
        finally:
            # On any non-success exit (crash, cancel, kill), roll the working
            # tree back to main and delete the orphaned cycle branch so the
            # repo is never left on a stale improve/... branch.
            if not completed:
                self._cleanup_failed_branch(repo, branch)

    def _reap_orphan_running_cycles(self) -> None:
        """Mark every still-`running` cycle as `error`.

        Only one cycle runs at a time, so any pre-existing `running` row is an
        orphan from a crashed/killed run. Best-effort: persistence errors here
        must not block the new cycle.
        """
        try:
            for row in self.db.list_cycles(status="running"):
                self.db.update_cycle(row["id"], status="error", ended_at=_now())
                self._emit("cycle_error", "orphaned running cycle reaped", cycle_id=row["id"])
        except Exception:  # noqa: BLE001
            pass

    def _cleanup_failed_branch(self, repo: GitRepo, branch: str) -> None:
        """Best-effort: return to ``main`` and delete a failed cycle's branch."""
        try:
            if repo.current_branch() != "main":
                repo.checkout("main")
            if repo.branch_exists(branch):
                repo.delete_branch(branch, force=True)
        except Exception:  # noqa: BLE001 - cleanup must never mask the original error
            pass

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
            # Same per-worker exploration budget as parallel mode: keep the
            # single worker from looping on reads forever. The cycle_id/step
            # providers are not thread-safe, but this ctx is used by exactly one
            # agent in one event loop, so the running count is fine.
            explore_limit=self.explore_budget,
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
        # Tokens are streamed live (for the SSE/UI) but NOT persisted per-token;
        # instead one `agent_text` row is written per finished turn via
        # `on_worker_turn_end`, so a multi-million-token run can't blow up the DB.
        turn_buf: list[str] = []

        def on_worker_token(tok: str) -> None:
            if self.on_token is not None:
                self.on_token(tok)
            turn_buf.append(tok)
            self._emit("agent_token", "", token=tok)

        def on_worker_turn_end() -> None:
            text = "".join(turn_buf).strip()
            turn_buf.clear()
            if text:
                emit("agent_text", text, text=text)

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
        agent.on_turn_end = on_worker_turn_end
        agent.on_tool = on_worker_tool
        agent.on_tool_result = on_worker_tool_result
        agent.on_usage = on_worker_usage

        emit("cycle_start", f"branch {branch}", cycle_id=cycle_id, branch=branch, objective=objective)

        feedback = ""
        first_run = True
        committed_any = False
        verdict: ReviewVerdict | None = None
        # Objective-gate rounds: how many times the objective checker has sent
        # the agent back because the objective was not yet met. Bounded by
        # max_step_attempts so an unachievable objective eventually terminates.
        objective_rounds = 0

        while True:
            # ------------------- WORK PHASE -------------------------------
            # When resuming after feedback, augment it with what is already on
            # the branch so the agent skips re-exploring (it keeps its own
            # conversation across steps, but the progress hint is still useful
            # after an objective-gate / review reset).
            if feedback:
                hint = self._build_resume_hint(repo, branch)
                if hint:
                    feedback = f"{feedback}\n\n[progress so far]\n{hint}"
                    emit("round_resume", "resuming with prior-progress context",
                         step=step_no)
            step_cap_hit = False
            while step_no < self.max_steps:
                gate_passed = False
                pending_propose = False
                for _ in range(self.max_gate_attempts):
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

            # ------------------- OBJECTIVE GATE --------------------------
            # Before spending reviewer passes on code quality, check the changes
            # actually achieve the objective. An unfinished objective must keep
            # the agent iterating (with concrete gaps as feedback) instead of
            # going to review on an "ok but incomplete" diff.
            objective_checker = ObjectiveChecker(self.llm, temperature=self.agent_temperature)
            objective_verdict = await objective_checker.check(objective, final_diff)
            self.db.add_usage(objective_checker.last_usage, cycle_id=cycle_id)
            if not objective_verdict.achieved:
                objective_rounds += 1
                emit("objective_not_met", "objective not yet achieved",
                     step=step_no, gaps=objective_verdict.gaps[:800])
                if objective_rounds >= self.max_step_attempts:
                    self.db.update_cycle(cycle_id, status="request-changes", ended_at=_now())
                    return self._result(
                        cycle_id, objective, branch, "request-changes",
                        ReviewVerdict("request_changes"), steps=step_no,
                        message=f"objective not achieved: {objective_verdict.gaps[:500]}",
                    )
                feedback = (
                    "The objective is NOT yet achieved. Address each gap and "
                    "make targeted edits:\n" + objective_verdict.gaps
                )
                emit("review_feedback", "resuming work to close objective gaps")
                continue

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
        human_req_id = self.db.create_review_request(
            cycle_id, "human", verdict="pending",
            comments=f"AI: {verdict.verdict}\n{verdict.comments}".strip(),
        )
        emit("awaiting_human", "cycle awaits human approval", branch=branch)

        if human_gate is None:
            self.db.update_cycle(cycle_id, status="awaiting-human", ended_at=_now())
            emit("human_pending", "no human gate attached — leaving branch for review")
            return self._result(cycle_id, objective, branch, "awaiting-human", verdict, steps=step_no)

        decision = human_gate(human_request)
        if inspect.isawaitable(decision):
            decision = await decision  # type: ignore[misc]
        if not decision:
            self.db.resolve_review_request(human_req_id, "rejected")
            self.db.update_cycle(cycle_id, status="rejected", human_verdict="rejected", ended_at=_now())
            emit("human_rejected", "human rejected cycle", branch=branch)
            return self._result(cycle_id, objective, branch, "rejected", verdict, human="rejected", steps=step_no)

        # approved -> merge --no-ff into main (co-authored by Nelke)
        try:
            merge_cycle_branch(repo, branch, cycle_id=cycle_id)
        except GitError as exc:
            self.db.resolve_review_request(human_req_id, "approved")
            self.db.update_cycle(cycle_id, status="merge-conflict", human_verdict="approved", ended_at=_now())
            emit("cycle_error", "merge failed", error=str(exc))
            return self._result(cycle_id, objective, branch, "merge-conflict", verdict,
                                human="approved", steps=step_no, message=str(exc)[:500])

        self.db.resolve_review_request(human_req_id, "approved")
        self.db.update_cycle(cycle_id, status="merged", human_verdict="approved", ended_at=_now())
        emit("merged", "cycle merged into main", branch=branch)
        return self._result(cycle_id, objective, branch, "merged", verdict,
                            human="approved", steps=step_no)

    # ------------------------------------------------------------------ #
    # Parallel mode: planner -> N workers in parallel -> central gate    #
    # ------------------------------------------------------------------ #
    async def _run_impl_parallel(
        self,
        repo: GitRepo,
        objective: str,
        branch: str,
        cycle_id: str,
        human_gate: Callable[[HumanReviewRequest], bool | Awaitable[bool]] | None,
    ) -> CycleResult:
        """Plan task slices, run worker agents in parallel, then gate+commit.

        The planner splits the objective into ``<= max_workers`` slices. Each
        slice is executed by its own agent in parallel against the shared
        working tree (no per-worker commits). After all workers finish the
        combined diff is gated, committed once, boot-checked, AI-reviewed and
        handed to the human gate — exactly the same tail as the single-worker
        ``_run_impl``.
        """
        memory = MemoryStore(repo.repo / "memory")

        def emit(kind: str, message: str = "", **data: Any) -> None:
            data.setdefault("cycle_id", cycle_id)
            data.setdefault("total_steps", self.max_steps)
            try:
                self.db.add_cycle_event(cycle_id, kind, message, data)
            except Exception:  # noqa: BLE001
                pass
            self._emit(kind, message, **data)

        emit("cycle_start", f"branch {branch}", cycle_id=cycle_id, branch=branch, objective=objective)

        # ---- PLAN ---------------------------------------------------------
        try:
            tasks = await plan_tasks(
                self.llm, objective,
                max_tasks=self.max_workers,
                temperature=self.agent_temperature,
            )
        except Exception:  # noqa: BLE001 - planner must never abort the cycle
            tasks = [TaskSpec(title="all", detail=objective)]
        emit("planned", f"{len(tasks)} task(s)", cycle_id=cycle_id, task_count=len(tasks),
             titles=[t.title for t in tasks])

        # Persist each slice as a cycle_worker row; the worker_id is stable for
        # the whole cycle so events/steps can reference it.
        worker_specs: list[tuple[str, int, TaskSpec]] = []
        for index, task in enumerate(tasks):
            wid = self.db.create_cycle_worker(cycle_id, index, task.title, task.detail)
            worker_specs.append((wid, index, task))

        # Shared read cache for the whole cycle: when one worker reads a file,
        # every other worker (and later round) gets the cached copy instead of
        # re-reading it. This is what stops parallel workers from duplicating
        # each other's exploration of the same files.
        shared_read_cache: dict[str, str] = {}

        # ---- WORK (parallel) ---------------------------------------------
        feedback = ""
        verdict: ReviewVerdict | None = None
        committed_any = False
        round_no = 0
        # Objective-gate rounds: how many times the objective checker has sent
        # workers back because the objective was not yet met. Bounded by
        # max_step_attempts so an unachievable objective eventually terminates.
        objective_rounds = 0
        # Consecutive rounds in which NO worker produced any changes. A worker
        # that burns its whole budget on exploration (read/grep/glob) and never
        # edits must be re-prompted to actually implement, not immediately sunk
        # as a dead `no-changes` cycle. We retry a bounded number of times,
        # then give up.
        no_progress_rounds = 0

        while True:
            round_no += 1
            # Snapshot the diff baseline so we can tell if workers produced
            # anything new this round (clean state -> gate idle path).
            had_changes_before = repo.has_changes()
            # When resuming after feedback (objective gaps / request_changes /
            # gate failure), workers start a fresh agent each round. Augment
            # the feedback with what is already done so they skip re-exploring
            # and go straight to the remaining work.
            if feedback:
                hint = self._build_resume_hint(repo, branch, shared_read_cache)
                if hint:
                    feedback = f"{feedback}\n\n[progress so far]\n{hint}"
                    emit("round_resume", "resuming with prior-progress context",
                         round=round_no)

            # Build per-worker agents and run them concurrently. Each worker
            # gets its own SelfEditContext.state so they do not stomp on each
            # other's flags (e.g. propose_complete).
            results = await asyncio.gather(
                *[
                    self._run_worker(worker_id, index, task, memory, cycle_id, emit, feedback,
                                     shared_read_cache)
                    for worker_id, index, task in worker_specs
                ],
                return_exceptions=True,
            )
            worker_summaries = []
            for (worker_id, _index, _task), result in zip(worker_specs, results, strict=True):
                if isinstance(result, Exception):
                    self.db.update_cycle_worker(worker_id, status="error", ended_at=_now())
                    emit("worker_error", f"worker crashed: {result}", worker_id=worker_id,
                         error=str(result)[:500])
                    worker_summaries.append(f"{worker_id}: error")
                    continue
                # asyncio.gather(return_exceptions=True) types the item as
                # dict | BaseException; the isinstance check above narrows the
                # exception branch, so this cast is safe.
                res: dict[str, Any] = cast("dict[str, Any]", result)
                self.db.update_cycle_worker(worker_id, status="done", ended_at=_now())
                emit("worker_done", f"worker finished: {res.get('answer', '')[:160]}",
                     worker_id=worker_id, answer=res.get("answer", "")[:400])
                worker_summaries.append(f"{worker_id}: {res.get('answer', '')[:80]}")

            await self._sync_dependencies_if_changed()

            # If no worker changed anything AND there was nothing pending from
            # earlier rounds, the cycle has nothing to commit.
            if not repo.has_changes():
                if not had_changes_before:
                    # Give the workers a bounded chance to turn their exploration
                    # into actual edits before declaring the cycle dead. See
                    # `no_progress_rounds` above.
                    if no_progress_rounds < self.max_step_attempts:
                        no_progress_rounds += 1
                        feedback = (
                            "No worker edited any files in the previous round — you only "
                            "explored the codebase. Exploration alone is not a result. "
                            "You MUST make concrete edits to the repo (add the module, "
                            "wire the service/API/UI, and add matching tests) that move "
                            "toward your assigned task. Do not distribute more exploration: "
                            "write code now."
                        )
                        emit(
                            "no_progress",
                            "workers made no changes; re-prompting to implement "
                            f"({no_progress_rounds}/{self.max_step_attempts})",
                            round=round_no,
                            attempts=no_progress_rounds,
                            limit=self.max_step_attempts,
                        )
                        continue
                    self.db.update_cycle(cycle_id, status="no-changes", ended_at=_now())
                    emit("cycle_error", "no changes produced by any worker")
                    return self._result(cycle_id, objective, branch, "no-changes",
                                        ReviewVerdict("request_changes"),
                                        steps=round_no,
                                        message="workers made no changes")

            # ---- GATE (one, central) -------------------------------------
            gate = await self.governance.gate()
            emit("gate", gate.describe(), passed=gate.passed)
            if not gate.passed:
                if round_no >= self.max_gate_attempts:
                    self.db.add_step(cycle_id, round_no, None, "failed-gate", gate.describe()[:500])
                    self.db.update_cycle(cycle_id, status="failed-gate", ended_at=_now())
                    emit("cycle_error", "gate could not be satisfied", feedback=gate.describe()[:2000])
                    return self._result(cycle_id, objective, branch, "failed-gate",
                                        ReviewVerdict("request_changes"),
                                        steps=round_no, message=gate.describe()[:500])
                # Send workers back to fix the gate failures with targeted feedback.
                feedback = (
                    "The governance gate rejected your combined changes:\n"
                    + gate.describe()
                    + "\nFix these exact problems with minimal changes."
                )
                emit("review_feedback", "workers retrying after gate failure",
                     feedback=feedback[:500])
                continue

            # ---- COMMIT (single, combined) -------------------------------
            repo.add_all()
            sha = repo.commit(
                f"Cycle {cycle_id} round {round_no}: {objective[:80]}",
                f"Nelke-Self-Improve: cycle {cycle_id} round {round_no} (parallel)",
            )
            committed_any = True
            summary = " | ".join(worker_summaries)[:500]
            self.db.add_step(cycle_id, round_no, sha, "committed", summary)
            emit("commit", f"committed {sha}", sha=sha, round=round_no, summary=summary)
            feedback = ""

            # ---- BOOT CHECK ----------------------------------------------
            boot = await self.governance.boot_check()
            if not boot.ok and not boot.skipped:
                repo.revert_commit(sha)
                self.db.add_step(cycle_id, round_no, sha, "failed-boot", boot.message[:500])
                emit("boot_check_failed", f"reverted {sha}: {boot.message[:200]}", sha=sha)
                if round_no >= self.max_step_attempts:
                    self.db.update_cycle(cycle_id, status="failed-boot", ended_at=_now())
                    return self._result(cycle_id, objective, branch, "failed-boot",
                                        ReviewVerdict("request_changes"),
                                        steps=round_no, message=boot.message[:500])
                feedback = (
                    "The combined commit broke the boot check and was reverted. "
                    "Rework the change so it imports cleanly and passes nelke.boot_check()."
                )
                continue
            self.db.add_step(cycle_id, round_no, sha, "ok", "boot check passed")
            emit("step_ok", f"round {round_no} committed, boot-check passed", round=round_no)

            # ---- OBJECTIVE GATE ------------------------------------------
            # Before spending a reviewer pass on code quality, check that the
            # changes actually achieve the objective. An unfinished objective
            # must keep the workers iterating (with concrete gaps as feedback)
            # instead of going to review on an "ok but incomplete" diff.
            final_diff = repo.diff("main", branch)
            objective_checker = ObjectiveChecker(
                self.llm, temperature=self.agent_temperature,
            )
            objective_verdict = await objective_checker.check(objective, final_diff)
            self.db.add_usage(objective_checker.last_usage, cycle_id=cycle_id)
            if not objective_verdict.achieved:
                objective_rounds += 1
                emit("objective_not_met",
                     "objective not yet achieved",
                     round=round_no, gaps=objective_verdict.gaps[:800])
                if objective_rounds >= self.max_step_attempts:
                    # The objective stays unmet after the budget: surface it as a
                    # request-changes outcome so the branch is left for review
                    # rather than silently merged unfinished.
                    self.db.update_cycle(cycle_id, status="request-changes", ended_at=_now())
                    return self._result(
                        cycle_id, objective, branch, "request-changes",
                        ReviewVerdict("request_changes"), steps=round_no,
                        message=f"objective not achieved: {objective_verdict.gaps[:500]}",
                    )
                feedback = (
                    "The objective is NOT yet achieved. Address each gap and "
                    "make targeted edits:\n" + objective_verdict.gaps
                )
                emit("review_feedback", "resuming work to close objective gaps")
                continue

            # ---- AI REVIEW -----------------------------------------------
            reviewer = Reviewer(repo, self.llm, base="main",
                                temperature=self.agent_temperature)
            verdict = await reviewer.review(objective, final_diff)
            self.db.add_usage(reviewer.last_usage, cycle_id=cycle_id)
            self.db.create_review_request(cycle_id, "ai", verdict=verdict.verdict,
                                          comments=verdict.comments)
            self.db.update_cycle(cycle_id, ai_verdict=verdict.verdict)
            emit("ai_review", f"AI: {verdict.verdict}", verdict=verdict.verdict,
                 comments=verdict.comments[:800], round=round_no)
            if verdict.approved:
                break
            if round_no < self.max_step_attempts:
                feedback = "AI reviewer requested changes:\n" + verdict.comments
                emit("review_feedback", "resuming work with reviewer feedback")
                continue
            self.db.update_cycle(cycle_id, status="request-changes", ended_at=_now())
            emit("cycle_error", "reviewer still requests changes",
                 comments=verdict.comments[:800])
            return self._result(cycle_id, objective, branch, "request-changes", verdict,
                                steps=round_no, message=verdict.comments[:500])

        # ---- HUMAN GATE (identical tail to _run_impl) ---------------------
        final_diff = repo.diff("main", branch)
        if not committed_any and not final_diff.strip():
            self.db.update_cycle(cycle_id, status="no-changes", ended_at=_now())
            emit("cycle_error", "no changes produced")
            return self._result(cycle_id, objective, branch, "no-changes",
                                ReviewVerdict("request_changes"),
                                steps=round_no, message="workers made no changes")
        assert verdict is not None
        human_request = HumanReviewRequest(
            cycle_id=cycle_id, objective=objective, branch=branch,
            diff=final_diff, ai_verdict=verdict,
        )
        human_req_id = self.db.create_review_request(
            cycle_id, "human", verdict="pending",
            comments=f"AI: {verdict.verdict}\n{verdict.comments}".strip(),
        )
        emit("awaiting_human", "cycle awaits human approval", branch=branch)

        if human_gate is None:
            self.db.update_cycle(cycle_id, status="awaiting-human", ended_at=_now())
            emit("human_pending", "no human gate attached — leaving branch for review")
            return self._result(cycle_id, objective, branch, "awaiting-human", verdict, steps=round_no)

        decision = human_gate(human_request)
        if inspect.isawaitable(decision):
            decision = await decision  # type: ignore[misc]
        if not decision:
            self.db.resolve_review_request(human_req_id, "rejected")
            self.db.update_cycle(cycle_id, status="rejected", human_verdict="rejected", ended_at=_now())
            emit("human_rejected", "human rejected cycle", branch=branch)
            return self._result(cycle_id, objective, branch, "rejected", verdict,
                                human="rejected", steps=round_no)

        try:
            merge_cycle_branch(repo, branch, cycle_id=cycle_id)
        except GitError as exc:
            self.db.resolve_review_request(human_req_id, "approved")
            self.db.update_cycle(cycle_id, status="merge-conflict", human_verdict="approved", ended_at=_now())
            emit("cycle_error", "merge failed", error=str(exc))
            return self._result(cycle_id, objective, branch, "merge-conflict", verdict,
                                human="approved", steps=round_no, message=str(exc)[:500])

        self.db.resolve_review_request(human_req_id, "approved")
        self.db.update_cycle(cycle_id, status="merged", human_verdict="approved", ended_at=_now())
        emit("merged", "cycle merged into main", branch=branch)
        return self._result(cycle_id, objective, branch, "merged", verdict,
                            human="approved", steps=round_no)

    async def _run_worker(
        self,
        worker_id: str,
        worker_index: int,
        task: TaskSpec,
        memory: MemoryStore,
        cycle_id: str,
        emit: Callable[..., None],
        feedback: str,
        read_cache: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Run a single parallel worker to completion and emit its progress.

        Returns a summary dict ``{worker_id, answer, error}`` suitable for the
        combined commit message. All agent activity (tokens, tools, usage) is
        forwarded to ``emit`` with ``worker_id`` so the UI can route it to the
        right worker card.
        """
        # Each worker has its own SelfEditContext.state so per-worker flags do
        # not collide; the worker has no propose_cycle_complete tool anyway.
        state: dict[str, Any] = {"worker_index": worker_index}
        # File ownership: only enforced when the planner gave this slice an
        # explicit file list. The fallback whole-objective slice has files=[]
        # -> allowed_files stays None (unrestricted), preserving old behaviour.
        allowed = set(task.files) if task.has_file_scope else None
        ctx = SelfEditContext(
            repo=self.repo,
            governance=self.governance,
            repo_root=self.repo.repo,
            state=state,
            cycle_id_provider=lambda: "",
            step_provider=lambda: 0,
            allowed_files=allowed,
            read_cache=read_cache,
            # Exploration budget: enforced inside the read-only tools. Once a
            # worker crosses the cap, further reads return a "switch to editing"
            # failure the model can see and react to — rather than the run being
            # silently yanked, which only made the worker re-explore next round.
            explore_limit=self.explore_budget,
        )
        self.db.update_cycle_worker(worker_id, status="running", started_at=_now())
        emit("worker_start", f"worker {worker_index} started",
             worker_id=worker_id, worker_index=worker_index, title=task.title)

        # Tokens stream live (for SSE/UI) but are not persisted per-token; one
        # `agent_text` row per finished turn keeps multi-million-token runs from
        # blowing up the DB.
        turn_buf: list[str] = []

        def on_worker_token(tok: str) -> None:
            turn_buf.append(tok)
            self._emit("agent_token", tok, worker_id=worker_id, token=tok)

        def on_worker_turn_end() -> None:
            text = "".join(turn_buf).strip()
            turn_buf.clear()
            if text:
                emit("agent_text", text, worker_id=worker_id, text=text)

        def on_worker_tool(name: str, args: dict[str, Any]) -> None:
            try:
                emit("agent_tool", f"{name}", worker_id=worker_id, tool=name, args=args)
            except Exception:  # noqa: BLE001
                pass

        def on_worker_tool_result(name: str, args: dict[str, Any], result: str) -> None:
            try:
                emit("agent_tool_result", f"{name}", worker_id=worker_id, tool=name,
                     snippet=(result or "")[:400])
            except Exception:  # noqa: BLE001
                pass
            # Surface when the budget has just been exhausted so the UI/timeline
            # shows the worker was nudged toward editing. Detected from the
            # canonical nudge text the tools return once over the cap.
            if (
                name in _READ_ONLY_TOOLS
                and self.explore_budget > 0
                and "EXPLORATION BUDGET EXHAUSTED" in (result or "")
            ):
                try:
                    emit("explore_budget_exceeded",
                         f"worker hit the exploration cap ({self.explore_budget}); "
                         "read-only calls now return a 'write code' nudge",
                         worker_id=worker_id, budget=self.explore_budget)
                except Exception:  # noqa: BLE001
                    pass

        def on_worker_usage(usage: dict[str, Any]) -> None:
            try:
                self.db.add_usage(usage, cycle_id=cycle_id)
            except Exception:  # noqa: BLE001
                pass
            try:
                emit("usage", "usage", worker_id=worker_id, usage=usage)
            except Exception:  # noqa: BLE001
                pass

        agent = self._build_worker_agent(
            ctx, memory, worker_id, task,
            on_token=on_worker_token, on_usage=on_worker_usage,
        )
        agent.on_turn_end = on_worker_turn_end
        agent.on_tool = on_worker_tool
        agent.on_tool_result = on_worker_tool_result

        prompt = task.as_prompt()
        if feedback:
            prompt = f"{prompt}\n\n[feedback]\n{feedback}"
        try:
            result = await agent.run(prompt, reset=True)
        except Exception as exc:  # noqa: BLE001 - one worker crashing must not kill the cycle
            return {"worker_id": worker_id, "answer": "", "error": str(exc)}
        return {"worker_id": worker_id, "answer": result.answer, "error": ""}
