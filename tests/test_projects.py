"""Tests for the project agent tools (project dir, memory, kanban).

These exercise the new capabilities the user asked for: projects now have a
localised directory (``<repo>/projects/<id>/``), their memory can be read and
written by agents, and kanban boards can be read and written by agents.
"""

from __future__ import annotations

from pathlib import Path

from nelke.core.services import create_project
from nelke.core.tools.projects import (
    KanbanAddCardTool,
    KanbanBoardTool,
    KanbanCreateBoardTool,
    ProjectDirectoryTool,
    ProjectMemoryReadTool,
    ProjectMemoryWriteTool,
    project_directory,
)


def _make_db(tmp_path: Path):
    from nelke.core.db import Database

    db = Database(tmp_path / "test.db")
    db.migrate()
    return db


class _Settings:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path


def _run(coro):
    import asyncio

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    return loop.run_until_complete(coro)


def test_project_directory_path():
    repo = Path("/repo")
    assert project_directory(repo, "abc") == Path("/repo") / "projects" / "abc"


def test_project_directory_tool_reports_path(tmp_path):
    db = _make_db(tmp_path)
    repo = tmp_path / "repo"
    repo.mkdir()
    project_id = create_project(_Settings(db.path), name="Demo", repo=repo)
    tool = ProjectDirectoryTool(db, repo)
    result = _run(tool.execute(project_id=project_id))
    assert result.ok
    assert project_id in result.render()
    assert result.render().startswith(f"project_id: {project_id}")
    # Directory is created on create_project.
    assert (repo / "projects" / project_id).exists()
    assert (repo / "projects" / project_id / "README.md").exists()


def test_project_memory_read_write(tmp_path):
    db = _make_db(tmp_path)
    repo = tmp_path / "repo"
    repo.mkdir()
    project_id = create_project(_Settings(db.path), name="Demo", repo=repo)
    w = ProjectMemoryWriteTool(db, repo)
    r = ProjectMemoryReadTool(db, repo)
    assert _run(w.execute(project_id=project_id, name="notes.md", content="hello world")).ok
    res = _run(r.execute(project_id=project_id, name="notes.md"))
    assert res.ok
    assert "hello world" in res.render()
    # Overwrite mode replaces.
    _run(w.execute(project_id=project_id, name="notes.md", content="replaced", overwrite=True))
    res2 = _run(r.execute(project_id=project_id, name="notes.md"))
    assert "replaced" in res2.render()
    assert "hello world" not in res2.render()


def test_project_memory_read_missing_file(tmp_path):
    db = _make_db(tmp_path)
    repo = tmp_path / "repo"
    repo.mkdir()
    project_id = create_project(_Settings(db.path), name="Demo", repo=repo)
    r = ProjectMemoryReadTool(db, repo)
    res = _run(r.execute(project_id=project_id, name="nope.md"))
    assert not res.ok
    assert "not found" in res.render()


def test_kanban_board_and_add_card(tmp_path):
    db = _make_db(tmp_path)
    repo = tmp_path / "repo"
    repo.mkdir()
    project_id = create_project(_Settings(db.path), name="Demo", repo=repo)
    board_tool = KanbanBoardTool(db, repo)
    create_tool = KanbanCreateBoardTool(db, repo)
    add_tool = KanbanAddCardTool(db, repo)

    # No boards yet.
    res = _run(board_tool.execute(project_id=project_id))
    assert res.ok
    assert "no kanban boards" in res.render()

    # Create a board.
    res = _run(create_tool.execute(project_id=project_id, name="Roadmap"))
    assert res.ok

    boards = db.list_kanban_boards(project_id)
    assert len(boards) == 1
    bid = boards[0]["id"]
    # Find the Backlog column id.
    backlog = next(c for c in db.list_kanban_columns(bid) if c["name"] == "Backlog")
    res = _run(add_tool.execute(
        project_id=project_id, board_id=bid, column=backlog["id"], title="Ship it",
    ))
    assert res.ok
    assert "Ship it" in res.render()

    # Read it back.
    res = _run(board_tool.execute(project_id=project_id))
    assert res.ok
    assert "Roadmap" in res.render()
    assert "Ship it" in res.render()


def test_kanban_add_card_by_column_name(tmp_path):
    db = _make_db(tmp_path)
    repo = tmp_path / "repo"
    repo.mkdir()
    project_id = create_project(_Settings(db.path), name="Demo", repo=repo)
    bid = db.create_kanban_board(project_id, "Board")
    add = KanbanAddCardTool(db, repo)
    res = _run(add.execute(project_id=project_id, board_id=bid, column="In Progress", title="ByName"))
    assert res.ok
    assert "ByName" in res.render()


def test_project_tools_reject_unknown_project(tmp_path):
    db = _make_db(tmp_path)
    repo = tmp_path / "repo"
    repo.mkdir()
    tool = ProjectDirectoryTool(db, repo)
    res = _run(tool.execute(project_id="does-not-exist"))
    assert not res.ok
    assert "not found" in res.render()
