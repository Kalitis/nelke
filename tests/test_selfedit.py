"""Self-edit tools: repo scoping and commit trailers."""

from __future__ import annotations

import pytest

from nelke.core.gitops import GitRepo
from nelke.core.governance import Governance
from nelke.core.tools.base import ToolError
from nelke.core.tools.selfedit import (
    GitCommitTool,
    SelfEditContext,
    SelfGrepTool,
    SelfGlobTool,
    SelfReadTool,
    SelfWriteTool,
)


def _ctx(tmp_repo: GitRepo) -> SelfEditContext:
    return SelfEditContext(
        repo=tmp_repo,
        governance=Governance(tmp_repo),
        repo_root=tmp_repo.repo,
        state={},
        cycle_id_provider=lambda: "c1",
        step_provider=lambda: 1,
    )


async def test_self_write_read_roundtrip(tmp_repo):
    ctx = _ctx(tmp_repo)
    w = SelfWriteTool(ctx)
    r = await w.execute(path="memory/facts/xyz.md", content="# Xyz\n\nBody")
    assert r.ok
    read = SelfReadTool(ctx)
    out = await read.execute(path="memory/facts/xyz.md")
    assert "Body" in out.output


async def test_self_write_refuses_outside_repo(tmp_repo, tmp_path):
    ctx = _ctx(tmp_repo)
    w = SelfWriteTool(ctx)
    with pytest.raises(ToolError):
        await w.execute(path=str(tmp_path / "evil.txt"), content="x")


async def test_git_commit_adds_trailer(tmp_repo):
    ctx = _ctx(tmp_repo)
    w = SelfWriteTool(ctx)
    await w.execute(path="memory/facts/t1.md", content="# T1\n\nnote")
    commit = GitCommitTool(ctx)
    r = await commit.execute()
    assert r.ok
    assert "c1" in r.output
    head = tmp_repo.head_sha()
    body = tmp_repo._run("log", "-1", "--format=%B", head).stdout
    assert "Nelke-Self-Improve: cycle c1 step 1" in body


async def test_self_glob_grep_skip_vendor_dirs(tmp_repo):
    """Vendor/cache trees must never pollute glob/grep (they balloon prompts and
    defeat prompt caching, and were the trigger for the cycle freezing)."""
    (tmp_repo.repo / "src").mkdir()
    (tmp_repo.repo / "src" / "main.py").write_text("def main():\n    pass\n", encoding="utf-8")
    (tmp_repo.repo / ".venv").mkdir(parents=True)
    (tmp_repo.repo / ".venv" / "lib.py").write_text("def from_venv():\n    pass\n", encoding="utf-8")
    (tmp_repo.repo / "__pycache__").mkdir()
    (tmp_repo.repo / "__pycache__" / "c.py").write_text("x = 1\n", encoding="utf-8")

    ctx = _ctx(tmp_repo)
    globbed = await SelfGlobTool(ctx).execute(pattern="**/*.py")
    assert "src/main.py" in globbed.output
    assert ".venv" not in globbed.output
    assert "__pycache__" not in globbed.output

    grepped = await SelfGrepTool(ctx).execute(pattern="def ", include="**/*")
    assert "src/main.py" in grepped.output
    assert "from_venv" not in grepped.output
