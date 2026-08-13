"""Self-edit tools: repo scoping and commit trailers."""

from __future__ import annotations

import pytest

from nelke.core.gitops import GitRepo
from nelke.core.governance import Governance
from nelke.core.tools.base import ToolError
from nelke.core.tools.selfedit import (
    GitCommitTool,
    SelfEditContext,
    SelfEditTool,
    SelfGlobTool,
    SelfGrepTool,
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


async def test_allowed_files_blocks_write_outside_scope(tmp_repo):
    """A worker whose slice owns memory/facts/in.md must not be able to write a
    file outside that scope: self_write refuses with a clear message."""
    ctx = _ctx(tmp_repo)
    ctx.allowed_files = {"memory/facts/in.md"}
    w = SelfWriteTool(ctx)
    blocked = await w.execute(path="src/nelke/core/cycle.py", content="x")
    assert not blocked.ok
    assert "outside your assigned scope" in blocked.error
    # writing an in-scope file still works
    ok = await w.execute(path="memory/facts/in.md", content="# ok")
    assert ok.ok


async def test_allowed_files_blocks_edit_outside_scope(tmp_repo):
    """self_edit is scoped the same way as self_write."""
    # seed a file via an unrestricted ctx, then restrict
    seed_ctx = _ctx(tmp_repo)
    await SelfWriteTool(seed_ctx).execute(path="memory/facts/in.md", content="hello")
    (tmp_repo.repo / "src").mkdir(exist_ok=True)
    (tmp_repo.repo / "src" / "other.py").write_text("X = 1", encoding="utf-8")

    ctx = _ctx(tmp_repo)
    ctx.allowed_files = {"memory/facts/in.md"}
    blocked = await SelfEditTool(ctx).execute(
        path="src/other.py", old_string="X = 1", new_string="X = 2")
    assert not blocked.ok
    assert "outside your assigned scope" in blocked.error
    ok = await SelfEditTool(ctx).execute(
        path="memory/facts/in.md", old_string="hello", new_string="hi")
    assert ok.ok


async def test_allowed_files_none_is_unrestricted(tmp_repo):
    """allowed_files=None (reviewer/sequential/fallback) allows any path — the
    default, preserving the legacy single-worker behaviour."""
    ctx = _ctx(tmp_repo)
    assert ctx.allowed_files is None
    w = SelfWriteTool(ctx)
    r = await w.execute(path="memory/facts/anywhere.md", content="x")
    assert r.ok


async def test_read_cache_serves_cached_copy(tmp_repo):
    """A shared read_cache returns a [cached] copy on the second read of the
    same file, so parallel workers stop duplicating each other's reads."""
    await SelfWriteTool(_ctx(tmp_repo)).execute(
        path="memory/facts/c.md", content="# C\n\nunique body text")

    cache: dict[str, str] = {}
    ctx = _ctx(tmp_repo)
    ctx.read_cache = cache
    r1 = await SelfReadTool(ctx).execute(path="memory/facts/c.md")
    assert r1.ok and "unique body text" in r1.output
    assert not r1.output.startswith("[cached]")
    assert "memory/facts/c.md" in cache

    # Second read (e.g. by another worker sharing the cache) is served cached.
    r2 = await SelfReadTool(ctx).execute(path="memory/facts/c.md")
    assert r2.ok
    assert r2.output.startswith("[cached]")
    assert "unique body text" in r2.output


async def test_explore_budget_nudges_after_limit(tmp_repo):
    """Once the read-only budget is exhausted, further reads return a 'write
    code' nudge instead of the file content — so the model sees a failure it
    can react to and switch to editing, rather than the run being silently
    yanked (which only restarted exploration next round)."""
    await SelfWriteTool(_ctx(tmp_repo)).execute(
        path="memory/facts/d.md", content="# D\nbody")

    ctx = _ctx(tmp_repo)
    ctx.explore_limit = 2  # allow two reads, then nudge
    reader = SelfReadTool(ctx)

    first = await reader.execute(path="memory/facts/d.md")
    second = await reader.execute(path="memory/facts/d.md")
    assert first.ok and second.ok  # within budget

    third = await reader.execute(path="memory/facts/d.md")
    assert not third.ok
    assert "EXPLORATION BUDGET EXHAUSTED" in third.error
    assert "self_write" in third.error  # points the model at the editing tools


async def test_explore_budget_zero_disables_nudge(tmp_repo):
    """explore_limit=0 means no cap (reviewer/legacy) — reads never nudge."""
    await SelfWriteTool(_ctx(tmp_repo)).execute(
        path="memory/facts/e.md", content="x")
    ctx = _ctx(tmp_repo)
    ctx.explore_limit = 0
    reader = SelfReadTool(ctx)
    for _ in range(20):
        r = await reader.execute(path="memory/facts/e.md")
        assert r.ok


async def test_explore_budget_does_not_block_writes(tmp_repo):
    """The budget only applies to read-only tools; self_write/self_edit are
    never nudged, so an over-budget worker can still make its edits."""
    ctx = _ctx(tmp_repo)
    ctx.explore_limit = 1
    # exhaust the read budget
    await SelfWriteTool(_ctx(tmp_repo)).execute(path="memory/facts/f.md", content="v1")
    reader = SelfReadTool(ctx)
    await reader.execute(path="memory/facts/f.md")  # 1/1 — still allowed
    over = await reader.execute(path="memory/facts/f.md")  # 2 — nudged
    assert not over.ok
    # writes still work after the budget is blown
    w = await SelfWriteTool(ctx).execute(path="memory/facts/f.md", content="v2")
    assert w.ok
    e = await SelfEditTool(ctx).execute(
        path="memory/facts/f.md", old_string="v2", new_string="v3")
    assert e.ok
