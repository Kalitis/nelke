"""Self-edit tools: available ONLY inside the self-improvement cycle.

Working directory is the Nelke repo. Every path is scoped to the repo root; git
operations go through ``GitRepo`` wrappers, not raw shell, for auditability.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from nelke.core.gitops import GitRepo
from nelke.core.governance import Governance
from nelke.core.tools.base import BaseTool, ToolResult, resolve_within

DEFAULT_ENCODING = "utf-8"
MAX_SELF_OUTPUT = 60_000
MAX_GLOB_RESULTS = 400

# Directories/files that must never be globbed/grepped: vendor stacks and caches
# balloon every tool result (and thus every following LLM prompt), make the cycle
# slow/stall-prone and defeat prompt caching. Anything here is irrelevant to
# improving the repo's own source.
_SKIP_DIR_PARTS = {
    ".git", ".hg", ".svn", ".venv", "venv", "env", "__pycache__",
    "node_modules", ".mypy_cache", ".pytest_cache", ".ruff_cache",
    ".cache", ".tox", ".eggs", "dist", "build", "target", "site-packages",
}
_SKIP_FILE_SUFFIXES = {".pyc", ".pyo"}


def _is_ignored(rel: Path) -> bool:
    parts = rel.parts
    if any(p in _SKIP_DIR_PARTS for p in parts):
        return True
    if rel.suffix in _SKIP_FILE_SUFFIXES:
        return True
    return bool(parts and parts[-1].endswith(".egg-info"))


@dataclass
class SelfEditContext:
    repo: GitRepo
    governance: Governance
    repo_root: Path
    state: dict[str, Any] = field(default_factory=dict)
    cycle_id_provider: Callable[[], str] = lambda: ""
    step_provider: Callable[[], int] = lambda: 0
    # File ownership: when set, self_edit/self_write refuse paths outside this
    # set (repo-relative forward-slash). None means "no restriction" (reviewer,
    # sequential worker, fallback whole-objective slice).
    allowed_files: set[str] | None = None
    # Shared read cache keyed by repo-relative forward-slash path. When set
    # (parallel workers share one cache per cycle), self_read serves a cached
    # copy instead of re-reading + re-streaming the same file. Workers exploring
    # the same files thus stop duplicating each other's reads.
    read_cache: dict[str, str] | None = None
    # Exploration budget: caps how many read-only tool calls a worker may make
    # in a single run. ``explore_limit`` is the cap (0 disables); ``explore_used``
    # is the running count, bumped by each read-only tool. Once the cap is hit,
    # further read-only calls return a failure that tells the model to switch to
    # editing — instead of silently letting it loop on exploration. The engine
    # sets these per worker per round.
    explore_limit: int = 0
    explore_used: int = 0

    def bump_explore(self) -> str | None:
        """Count one read-only call. Returns a 'stop exploring, start editing'
        failure message once the budget is exhausted, else None (call proceeds).

        On the call that crosses the limit we still allow it (the model already
        issued it), but every call AFTER returns the nudge. That keeps the tool
        result visible to the model so it learns to switch, rather than the run
        being yanked away mid-thought (which just restarted exploration next round).
        """
        if self.explore_limit <= 0:
            return None
        self.explore_used += 1
        if self.explore_used <= self.explore_limit:
            return None
        return (
            "EXPLORATION BUDGET EXHAUSTED. You have already explored enough. "
            "Do NOT call self_read/self_glob/self_grep/recall again. "
            "Use self_write or self_edit NOW to make the concrete edits your "
            "task requires. Exploration alone produces no changes and the cycle "
            "will fail — write code."
        )

    def trailer(self) -> str:
        return f"Nelke-Self-Improve: cycle {self.cycle_id_provider()} step {self.step_provider()}"

    def _rel(self, abs_path: Path) -> str:
        """Repo-relative forward-slash form of ``abs_path`` for scope/cache keys."""
        try:
            return abs_path.resolve().relative_to(self.repo_root.resolve()).as_posix()
        except ValueError:
            return abs_path.as_posix()

    def check_write_scope(self, abs_path: Path) -> str | None:
        """Return an error message if ``abs_path`` is outside the allowed scope,
        else None. ``allowed_files is None`` means unrestricted."""
        if self.allowed_files is None:
            return None
        rel = self._rel(abs_path)
        if rel in self.allowed_files:
            return None
        allowed = ", ".join(sorted(self.allowed_files)) or "(none)"
        return (
            f"{rel} is outside your assigned scope. You may only edit: {allowed}. "
            "Stay in your lane — other workers own the rest."
        )


def _read(path: Path) -> str | None:
    if not path.is_file():
        return None
    content = path.read_text(encoding=DEFAULT_ENCODING, errors="replace")
    if len(content) > MAX_SELF_OUTPUT:
        content = content[:MAX_SELF_OUTPUT] + "\n...[truncated]"
    return content


class SelfReadTool(BaseTool):
    name = "self_read"
    description = "Read a file inside the Nelke repo (self-edit mode)."
    parameters = {
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
    }

    def __init__(self, ctx: SelfEditContext) -> None:
        self.ctx = ctx

    async def execute(self, **kwargs: Any) -> ToolResult:
        nudge = self.ctx.bump_explore()
        if nudge is not None:
            return ToolResult.failure(nudge)
        path = resolve_within(self.ctx.repo_root, kwargs.get("path", ""))
        rel = self.ctx._rel(path)
        cache = self.ctx.read_cache
        if cache is not None and rel in cache:
            # Another worker already read this file; serve the cached copy so
            # parallel workers stop duplicating each other's exploration.
            return ToolResult.success(f"[cached]\n{cache[rel]}")
        content = _read(path)
        if content is None:
            return ToolResult.failure(f"file not found: {path}")
        if cache is not None:
            cache[rel] = content
        return ToolResult.success(content)


class SelfWriteTool(BaseTool):
    name = "self_write"
    description = "Create or overwrite a file inside the Nelke repo (self-edit mode)."
    parameters = {
        "type": "object",
        "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
        "required": ["path", "content"],
    }

    def __init__(self, ctx: SelfEditContext) -> None:
        self.ctx = ctx

    async def execute(self, **kwargs: Any) -> ToolResult:
        path = resolve_within(self.ctx.repo_root, kwargs.get("path", ""))
        scope_error = self.ctx.check_write_scope(path)
        if scope_error is not None:
            return ToolResult.failure(scope_error)
        content = str(kwargs.get("content", ""))
        # Write then, if the content is effectively empty, remove the scratch
        # file so an accidental empty probe doesn't linger as a git ghost.
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding=DEFAULT_ENCODING)
            if not content.strip():
                _scrub_empty(path)
        except OSError as exc:
            return ToolResult.failure(f"self_write failed: {exc}")
        return ToolResult.success(f"wrote {path.relative_to(self.ctx.repo_root)}")


def _scrub_empty(path: Path) -> None:
    """Unlink a just-written empty file and any empty parents created for it.

    Lets the agent "delete" a file by writing empty content: the inode vanishes
    from the working tree (git sees only the file's prior committed state), so an
    accidental empty scratch file created mid-cycle does not linger as a ghost in
    the diff.
    """
    try:
        path.unlink()
        parent = path.parent
        while parent != path.anchor and not any(parent.iterdir()):
            parent.rmdir()
            parent = parent.parent
    except OSError:
        pass  # best-effort: never let cleanup break a write


class SelfEditTool(BaseTool):
    name = "self_edit"
    description = "Replace an exact string inside a file in the Nelke repo (self-edit mode)."
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "old_string": {"type": "string"},
            "new_string": {"type": "string"},
            "replace_all": {"type": "boolean", "default": False},
        },
        "required": ["path", "old_string", "new_string"],
    }

    def __init__(self, ctx: SelfEditContext) -> None:
        self.ctx = ctx

    async def execute(self, **kwargs: Any) -> ToolResult:
        path = resolve_within(self.ctx.repo_root, kwargs.get("path", ""))
        scope_error = self.ctx.check_write_scope(path)
        if scope_error is not None:
            return ToolResult.failure(scope_error)
        if not path.is_file():
            return ToolResult.failure(f"file not found: {path}")
        old = str(kwargs.get("old_string", ""))
        new = str(kwargs.get("new_string", ""))
        replace_all = bool(kwargs.get("replace_all", False))
        content = path.read_text(encoding=DEFAULT_ENCODING)
        count = content.count(old)
        if count == 0:
            return ToolResult.failure("old_string not found")
        if count > 1 and not replace_all:
            return ToolResult.failure("old_string found multiple times; pass replace_all=true")
        content = content.replace(old, new) if replace_all else content.replace(old, new, 1)
        path.write_text(content, encoding=DEFAULT_ENCODING)
        return ToolResult.success(f"edited {path.relative_to(self.ctx.repo_root)}")


class SelfGlobTool(BaseTool):
    name = "self_glob"
    description = "List files matching a glob pattern inside the Nelke repo."
    parameters = {
        "type": "object",
        "properties": {"pattern": {"type": "string"}},
        "required": ["pattern"],
    }

    def __init__(self, ctx: SelfEditContext) -> None:
        self.ctx = ctx

    async def execute(self, **kwargs: Any) -> ToolResult:
        nudge = self.ctx.bump_explore()
        if nudge is not None:
            return ToolResult.failure(nudge)
        root = self.ctx.repo_root.resolve()
        matches = sorted(
            p.relative_to(root).as_posix()
            for p in root.glob(str(kwargs.get("pattern", "")))
            if not _is_ignored(p.relative_to(root))
        )
        if len(matches) > MAX_GLOB_RESULTS:
            body = "\n".join(matches[:MAX_GLOB_RESULTS])
            body += f"\n...[truncated {len(matches) - MAX_GLOB_RESULTS} more matches]"
            return ToolResult.success(body)
        return ToolResult.success("\n".join(matches) if matches else "no matches")


class SelfGrepTool(BaseTool):
    name = "self_grep"
    description = "Search file contents inside the Nelke repo with a regular expression."
    parameters = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string"},
            "include": {"type": "string", "description": "Glob filter, e.g. '*.py'", "default": "**/*"},
        },
        "required": ["pattern"],
    }

    def __init__(self, ctx: SelfEditContext) -> None:
        self.ctx = ctx

    async def execute(self, **kwargs: Any) -> ToolResult:
        import re

        nudge = self.ctx.bump_explore()
        if nudge is not None:
            return ToolResult.failure(nudge)
        pattern = str(kwargs.get("pattern", ""))
        include = str(kwargs.get("include") or "**/*")
        try:
            regex = re.compile(pattern)
        except re.error as exc:
            return ToolResult.failure(f"invalid regex: {exc}")
        root = self.ctx.repo_root.resolve()
        hits: list[str] = []
        for p in root.glob(include):
            rel = p.relative_to(root)
            if not p.is_file() or _is_ignored(rel):
                continue
            try:
                for lineno, line in enumerate(p.read_text(encoding=DEFAULT_ENCODING, errors="replace").splitlines(), 1):
                    if regex.search(line):
                        hits.append(f"{rel.as_posix()}:{lineno}: {line[:200]}")
                        if len(hits) >= 200:
                            return ToolResult.success("\n".join(hits) + "\n...[truncated]")
            except OSError:
                continue
        return ToolResult.success("\n".join(hits) if hits else "no matches")


class GitDiffTool(BaseTool):
    name = "git_diff"
    description = "Show the diff between two refs (default main...HEAD) of the Nelke repo (read-only)."
    parameters = {
        "type": "object",
        "properties": {
            "base": {"type": "string", "default": "main"},
            "head": {"type": "string", "default": "HEAD"},
        },
        "required": [],
    }

    def __init__(self, ctx: SelfEditContext) -> None:
        self.ctx = ctx

    async def execute(self, **kwargs: Any) -> ToolResult:
        base = str(kwargs.get("base") or "main")
        head = str(kwargs.get("head") or "HEAD")
        return ToolResult.success(self.ctx.repo.diff(base, head))


class RunLintTool(BaseTool):
    name = "run_lint"
    description = "Run the lint gate (ruff) over the repo. Returns PASS/FAIL output."
    parameters = {"type": "object", "properties": {}, "required": []}

    def __init__(self, ctx: SelfEditContext) -> None:
        self.ctx = ctx

    async def execute(self, **kwargs: Any) -> ToolResult:
        return _check_result(await self.ctx.governance.run_lint())


class RunTypecheckTool(BaseTool):
    name = "run_typecheck"
    description = "Run the typecheck gate (mypy) over src/nelke. Returns PASS/FAIL output."
    parameters = {"type": "object", "properties": {}, "required": []}

    def __init__(self, ctx: SelfEditContext) -> None:
        self.ctx = ctx

    async def execute(self, **kwargs: Any) -> ToolResult:
        return _check_result(await self.ctx.governance.run_typecheck())


class RunTestsTool(BaseTool):
    name = "run_tests"
    description = "Run the test gate (pytest). Returns PASS/FAIL output."
    parameters = {"type": "object", "properties": {}, "required": []}

    def __init__(self, ctx: SelfEditContext) -> None:
        self.ctx = ctx

    async def execute(self, **kwargs: Any) -> ToolResult:
        return _check_result(await self.ctx.governance.run_tests())


class BootCheckTool(BaseTool):
    name = "boot_check"
    description = "Run the boot-check subprocess (imports nelke + runs a no-op-LLM smoke). Returns result."
    parameters = {"type": "object", "properties": {}, "required": []}

    def __init__(self, ctx: SelfEditContext) -> None:
        self.ctx = ctx

    async def execute(self, **kwargs: Any) -> ToolResult:
        return _check_result(await self.ctx.governance.boot_check())


class GitBranchInfoTool(BaseTool):
    name = "git_branch_info"
    description = "Show current branch, working tree status and recent commits of the Nelke repo."
    parameters = {"type": "object", "properties": {}, "required": []}

    def __init__(self, ctx: SelfEditContext) -> None:
        self.ctx = ctx

    async def execute(self, **kwargs: Any) -> ToolResult:
        repo = self.ctx.repo
        parts = [
            f"branch: {repo.current_branch()}",
            f"changes: {'yes' if repo.has_changes() else 'no'}",
            "log:",
            repo.log(10),
        ]
        return ToolResult.success("\n".join(parts))


class GitCommitTool(BaseTool):
    name = "git_commit"
    description = (
        "Stage all changes and commit them on the current branch with a "
        "Nelke-Self-Improve trailer. Returns the commit sha."
    )
    parameters = {
        "type": "object",
        "properties": {
            "message": {
                "type": "string",
                "description": "Optional concise commit message",
                "default": None,
            }
        },
        "required": [],
    }

    def __init__(self, ctx: SelfEditContext) -> None:
        self.ctx = ctx

    async def execute(self, **kwargs: Any) -> ToolResult:
        repo = self.ctx.repo
        if not repo.has_changes():
            return ToolResult.success("no changes to commit")
        message = kwargs.get("message") or f"Cycle {self.ctx.cycle_id_provider()} step {self.ctx.step_provider()}"
        try:
            repo.add_all()
            sha = repo.commit(str(message), self.ctx.trailer())
        except Exception as exc:  # noqa: BLE001
            return ToolResult.failure(f"commit failed: {exc}")
        return ToolResult.success(f"committed {sha}: {message}")


class GitRevertTool(BaseTool):
    name = "git_revert"
    description = "Create a revert commit for the most recent commit (rollback after a boot failure)."
    parameters = {"type": "object", "properties": {}, "required": []}

    def __init__(self, ctx: SelfEditContext) -> None:
        self.ctx = ctx

    async def execute(self, **kwargs: Any) -> ToolResult:
        head = self.ctx.repo.head_sha()
        result = self.ctx.repo.revert_commit(head)
        if not result.ok:
            return ToolResult.failure(f"revert failed: {result.text}")
        return ToolResult.success(f"reverted {head}: {result.text}")


class ProposeCycleCompleteTool(BaseTool):
    name = "propose_cycle_complete"
    description = (
        "Call this when the objective is fully achieved: stops the step loop and "
        "sends the branch to AI + human review. No arguments."
    )
    parameters = {"type": "object", "properties": {}, "required": []}

    def __init__(self, ctx: SelfEditContext) -> None:
        self.ctx = ctx

    async def execute(self, **kwargs: Any) -> ToolResult:
        self.ctx.state["propose_complete"] = True
        return ToolResult.success("proposing cycle completion")


def _check_result(result: Any) -> ToolResult:
    if result.skipped:
        return ToolResult.success(f"(skipped) {result.name}: {result.message}")
    if result.ok:
        return ToolResult.success(f"[PASS] {result.name}: {result.message}")
    return ToolResult.failure(f"[FAIL] {result.name}: {result.message}")
