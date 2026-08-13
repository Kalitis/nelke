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
    KanbanDeleteCardTool,
    KanbanMoveCardTool,
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


def test_project_directory_reports_path(tmp_path):
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


def test_project_directory_for_legacy_project(tmp_path):
    """A project created without a repo dir still gets a localised root."""
    from nelke.core.services import ensure_project_directory

    repo = tmp_path / "repo"
    repo.mkdir()
    # Create a project in the DB that ensure_project_directory will read,
    # WITHOUT seeding a directory.
    db = _make_db(tmp_path)
    legacy = db.create_project("Legacy")
    assert not (repo / "projects" / legacy).exists()

    path = ensure_project_directory(repo, legacy, settings=_Settings(db.path))
    assert path is not None
    assert path.exists()
    assert (path / "README.md").exists()
    # Idempotent: calling again does not error and returns the same path.
    assert ensure_project_directory(repo, legacy, settings=_Settings(db.path)) == path
    # Unknown project -> None.
    assert ensure_project_directory(repo, "no-such-project", settings=_Settings(db.path)) is None


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


def test_kanban_move_and_delete_card(tmp_path):
    db = _make_db(tmp_path)
    repo = tmp_path / "repo"
    repo.mkdir()
    project_id = create_project(_Settings(db.path), name="Demo", repo=repo)
    bid = db.create_kanban_board(project_id, "Board")
    # Backlog -> In Progress -> Done (default columns).
    backlog = next(c for c in db.list_kanban_columns(bid) if c["name"] == "Backlog")
    in_prog = next(c for c in db.list_kanban_columns(bid) if c["name"] == "In Progress")
    card_id = db.add_kanban_card(bid, backlog["id"], "Task A")

    move = KanbanMoveCardTool(db, repo)
    res = _run(move.execute(card_id=card_id, column=in_prog["id"]))
    assert res.ok
    assert db.get_kanban_card(card_id)["column_id"] == in_prog["id"]

    # Move by column name too.
    res = _run(move.execute(card_id=card_id, column="Done"))
    assert res.ok
    done = next(c for c in db.list_kanban_columns(bid) if c["name"] == "Done")
    assert db.get_kanban_card(card_id)["column_id"] == done["id"]

    delete = KanbanDeleteCardTool(db, repo)
    res = _run(delete.execute(card_id=card_id))
    assert res.ok
    assert db.get_kanban_card(card_id) is None


def test_create_chat_in_project(tmp_path):
    db = _make_db(tmp_path)
    from nelke.core.services import create_chat

    project_id = db.create_project("Demo")
    chat_id = create_chat(_Settings(db.path), title="In project", project_id=project_id)
    assert chat_id
    rows = db.list_project_sessions(project_id)
    assert [r["id"] for r in rows] == [chat_id]
    # The chat is attached: set_session_project would have failed without it.
    assert db.get_session(chat_id)["project_id"] == project_id


def test_make_agent_registers_project_tools(tmp_path):
    from nelke.core.agent import make_agent

    db = _make_db(tmp_path)
    repo = tmp_path / "repo"
    (repo / "memory").mkdir(parents=True)
    (repo / "src" / "nelke").mkdir(parents=True)

    class FakeLLM:
        async def chat(self, *a, **k):
            from nelke.core.llm import LLMResponse

            return LLMResponse(content="ok", tool_calls=[])

    agent = make_agent(
        workspace=repo / "workspaces" / "ws1", llm=FakeLLM(),
        name="t", db=db, system_prompt="sys",
    )
    names = agent.registry.names()
    for expected in (
        "project_directory",
        "project_memory_read",
        "project_memory_write",
        "kanban_board",
        "kanban_create_board",
        "kanban_add_card",
        "kanban_move_card",
        "kanban_update_card",
        "kanban_delete_card",
    ):
        assert expected in names, expected
    # The system prompt mentions project capabilities so agents actually use them.
    assert "Projects" in agent.system_content()
    assert "kanban" in agent.system_content()
    assert "project_memory_read" in agent.system_content()


def test_project_tools_reject_unknown_project(tmp_path):
    db = _make_db(tmp_path)
    repo = tmp_path / "repo"
    repo.mkdir()
    tool = ProjectDirectoryTool(db, repo)
    res = _run(tool.execute(project_id="does-not-exist"))
    assert not res.ok
    assert "not found" in res.render()
