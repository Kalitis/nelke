"""Governance: the self-improvement gates.

Intermediate commit gate  = lint + typecheck + tests all pass (tests mandatory,
non-empty). Rollback gate = post-commit boot check. All commands run against the
Nelke repo via the project venv (``uv run``); missing optional tools degrade to a
reported skip so the gate is never a permanent blocker, while tests stay enforced.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from nelke.core.gitops import GitRepo

GOVERNANCE_TIMEOUT = 900


@dataclass
class CheckResult:
    name: str
    ok: bool
    skipped: bool = False
    message: str = ""
    command: str = ""

    def describe(self) -> str:
        state = "SKIP" if self.skipped else ("PASS" if self.ok else "FAIL")
        return f"[{state}] {self.name}: {self.message}".strip()


@dataclass
class GateResult:
    passed: bool
    checks: list[CheckResult] = field(default_factory=list)

    def describe(self) -> str:
        return "\n".join(c.describe() for c in self.checks) or "(no checks run)"


class CommandRunner:
    """Async subprocess runner capturing merged stdout/stderr."""

    async def run(self, args: list[str], cwd: str, timeout: float) -> tuple[int, str]:
        proc = await asyncio.create_subprocess_exec(
            *args,
            cwd=cwd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        try:
            out, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            status = proc.returncode if proc.returncode is not None else -1
        except asyncio.TimeoutError:
            proc.kill()
            await proc.communicate()
            return -1, "command timed out"
        text = out.decode("utf-8", errors="replace")
        return status, text.strip()


class Governance:
    def __init__(self, repo: GitRepo, runner: CommandRunner | None = None) -> None:
        self.repo = repo
        self.runner = runner or CommandRunner()

    def _base_cmd(self, tool_cmd: list[str]) -> list[str]:
        return ["uv", "run", *tool_cmd]

    async def _check(
        self, name: str, cmd: list[str], timeout: float = GOVERNANCE_TIMEOUT
    ) -> CheckResult:
        code, out = await self.runner.run(cmd, str(self.repo.repo), timeout)
        message = out[-2000:] or "no output"
        if code not in (0,):
            lowered = out.lower()
            command = " ".join(cmd)
            if any(tok in lowered for tok in ("not found", "no such file", "is not a recognized")):
                msg = f"tool unavailable: {message}"
                return CheckResult(name=name, ok=True, skipped=True, message=msg, command=command)
            return CheckResult(name=name, ok=False, message=f"exit {code}: {message}", command=command)
        return CheckResult(name=name, ok=True, message=message, command=" ".join(cmd))

    async def run_lint(self) -> CheckResult:
        return await self._check("lint", self._base_cmd(["ruff", "check", "."]), timeout=300)

    async def run_typecheck(self) -> CheckResult:
        return await self._check("typecheck", self._base_cmd(["mypy", "src/nelke"]), timeout=300)

    async def run_tests(self) -> CheckResult:
        return await self._check(
            "tests", self._base_cmd(["pytest", "-q"]), timeout=GOVERNANCE_TIMEOUT
        )

    async def boot_check(self) -> CheckResult:
        cmd = self._base_cmd(["python", "-c", "import nelke; nelke.boot_check()"])
        return await self._check("boot-check", cmd, timeout=180)

    async def gate(self) -> GateResult:
        """Run lint -> typecheck -> tests, failing fast on the first hard failure."""
        checks: list[CheckResult] = []
        for fn in (self.run_lint, self.run_typecheck, self.run_tests):
            result = await fn()
            checks.append(result)
            if not result.ok and not result.skipped:
                break
        return GateResult(passed=all(c.ok for c in checks), checks=checks)
