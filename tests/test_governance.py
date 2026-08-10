"""Governance gate tests with a fake command runner + the real boot_check."""

from __future__ import annotations

from nelke.core.gitops import GitRepo
from nelke.core.governance import CheckResult, Governance


class FakeRunner:
    def __init__(self, results: list[tuple[int, str]]):
        self.results = list(results)

    async def run(self, args, cwd, timeout):
        code, out = self.results.pop(0)
        return code, out


async def test_lint_pass(tmp_path):
    repo = GitRepo(tmp_path / "r")
    gov = Governance(repo, runner=FakeRunner([(0, "All checks passed!")]))
    r = await gov.run_lint()
    assert r.ok and not r.skipped


async def test_tests_failure(tmp_path):
    repo = GitRepo(tmp_path / "r")
    gov = Governance(repo, runner=FakeRunner([(1, "4 failed, 2 passed")]))
    r = await gov.run_tests()
    assert not r.ok


async def test_missing_tool_degrades_to_skip(tmp_path):
    repo = GitRepo(tmp_path / "r")
    gov = Governance(repo, runner=FakeRunner([(1, "ruff: command not found")]))
    r = await gov.run_lint()
    assert r.ok and r.skipped


async def test_gate_requires_all_checks(tmp_path):
    repo = GitRepo(tmp_path / "r")
    gov = Governance(
        repo,
        runner=FakeRunner(
            [(0, "lint ok"), (0, "mypy ok"), (0, "3 passed")],
        ),
    )
    gate = await gov.gate()
    assert gate.passed
    assert [c.name for c in gate.checks] == ["lint", "typecheck", "tests"]


async def test_gate_fails_fast(tmp_path):
    repo = GitRepo(tmp_path / "r")
    gov = Governance(
        repo,
        runner=FakeRunner([(0, "lint ok"), (1, "mypy error")]),
    )
    gate = await gov.gate()
    assert not gate.passed
    assert len(gate.checks) == 2  # stopped after typecheck failure


def test_boot_check_import_smoke():
    """The real nelke.boot_check() must pass: imports all core, no network."""
    import nelke

    nelke.boot_check()


def test_check_result_describe_changed_message():
    r = CheckResult(name="tests", ok=False, message="boom")
    assert "[FAIL]" in r.describe()
