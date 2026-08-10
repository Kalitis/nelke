"""Self-edit tools: repo scoping and commit trailers."""

from __future__ import annotations

import pytest

from nelke.core.gitops import GitRepo
from nelke.core.governance import Governance
from nelke.core.tools.base import ToolError
from nelke.core.tools.selfedit import (
    GitCommitTool,
    SelfEditContext,
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
