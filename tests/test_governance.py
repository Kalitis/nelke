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
    assert [c.name for c in gate.checks] == ["lint", "typecheck", "test-gap", "tests"]


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


# --------------------------------------------------------------------------- #
# test-gap: code changes must ship with a matching test file
# --------------------------------------------------------------------------- #
async def test_test_gap_skipped_when_no_code_changes(tmp_repo):
    gov = Governance(tmp_repo)
    r = await gov.run_code_test_gap()
    assert r.ok and r.skipped


async def test_test_gap_rejects_new_src_without_test(tmp_repo):
    (tmp_repo.repo / "src").mkdir(parents=True)
    (tmp_repo.repo / "src" / "widget.py").write_text("def f():\n    return 1\n", encoding="utf-8")
    gov = Governance(tmp_repo)
    r = await gov.run_code_test_gap()
    assert not r.ok
    assert "tests/test_widget.py" in r.message


async def test_test_gap_passes_with_matching_test(tmp_repo):
    (tmp_repo.repo / "tests").mkdir(parents=True)
    (tmp_repo.repo / "src").mkdir(parents=True)
    (tmp_repo.repo / "src" / "widget.py").write_text("def f():\n    return 1\n", encoding="utf-8")
    (tmp_repo.repo / "tests" / "test_widget.py").write_text("def test_f():\n    pass\n", encoding="utf-8")
    gov = Governance(tmp_repo)
    r = await gov.run_code_test_gap()
    assert r.ok and not r.skipped


async def test_test_gap_passes_on_edit_of_covered_module(tmp_repo):
    (tmp_repo.repo / "tests").mkdir(parents=True)
    (tmp_repo.repo / "src").mkdir()
    (tmp_repo.repo / "src" / "cycle.py").write_text("def run():\n    return 'x'\n", encoding="utf-8")
    (tmp_repo.repo / "tests" / "test_cycle.py").write_text("def test_run():\n    pass\n", encoding="utf-8")
    (tmp_repo.repo / "src" / "cycle.py").write_text("def run():\n    return 'new'\n", encoding="utf-8")
    gov = Governance(tmp_repo)
    r = await gov.run_code_test_gap()
    assert r.ok and not r.skipped


async def test_test_gap_exempts_init_and_can_be_disabled(tmp_repo):
    (tmp_repo.repo / "src").mkdir(parents=True)
    (tmp_repo.repo / "src" / "__init__.py").write_text("__version__ = '1.0'\n", encoding="utf-8")
    gov = Governance(tmp_repo)
    r = await gov.run_code_test_gap()
    assert r.ok and r.skipped  # __init__ is exempt, nothing missing

    gov2 = Governance(tmp_repo, require_tests=False)
    r2 = await gov2.run_code_test_gap()
    assert r2.ok and r2.skipped  # enforcement disabled
