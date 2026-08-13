"""Agent tools for projects: project directory, project memory, and kanban.

These tools let an agent read and write a project's localised directory
(``<repo>/projects/<id>/``), read/write its memory notes, and query/mutate its
kanban board — closing the loop the user described: kanban was neither read nor
written by agents, per-project memory could not be read, and projects had no
directory around which they are localised.

All project tools share one db handle and one project-dir resolver so the agent
acts on the *same* project throughout a run.
"""

from __future__ import annotations

from pathlib import Path

from nelke.core.db import Database
from nelke.core.services import open_project_memory
from nelke.core.tools.base import BaseTool, ToolResult

# Root under which each project directory lives (sibling of the global memory dir).
PROJECTS_ROOT_NAME = "projects"


def project_directory(repo_root: Path, project_id: str) -> Path:
    """The localised directory for a project: ``<repo>/projects/<id>/``.

    This is the directory *around which* the project is organized and *in which*
    its artifacts live. Created lazily on first use. It sits next to the global
    memory tree (``<repo>/memory``) and the per-project memory store lives under
    ``<repo>/memory/projects/<id>/`` — the project dir is the working/localisation
    root, distinct from its git-tracked memory notes.
    """
    return repo_root / PROJECTS_ROOT_NAME / project_id


def _resolve_repo(settings) -> Path:
    from nelke.core.services import find_repo

    return find_repo(settings or None)


class ProjectDirectoryTool(BaseTool):
    name = "project_directory"
    description = (
        "Show the localised directory of a project (repo/projects/<id>) and whether "
        "it has been created yet. Use to discover where a project's files live."
    )
    parameters = {
        "type": "object",
        "properties": {"project_id": {"type": "string"}},
        "required": ["project_id"],
    }

    def __init__(self, db: Database, repo_root: Path) -> None:
        self.db = db
        self.repo_root = repo_root

    async def execute(self, **kwargs) -> ToolResult:
        project_id = str(kwargs.get("project_id", ""))
        if self.db.get_project(project_id) is None:
            return ToolResult.failure(f"project not found: {project_id}")
        path = project_directory(self.repo_root, project_id)
        exists = path.exists()
        lines = [
            f"project_id: {project_id}",
            f"directory: {path}",
            f"created: {exists}",
            f"memory: {self.repo_root / 'memory' / 'projects' / project_id}",
        ]
        if exists:
            files = sorted(p.relative_to(path).as_posix()
                           for p in path.rglob("*") if p.is_file())
            lines.append("files: " + (", ".join(files) if files else "(empty)"))
        return ToolResult.success("\n".join(lines))


class ProjectMemoryReadTool(BaseTool):
    name = "project_memory_read"
    description = (
        "Read the full text of a memory note belonging to a project (a flat *.md file "
        "under memory/projects/<id>/). Returns 'file not found' for an unknown note."
    )
    parameters = {
        "type": "object",
        "properties": {
            "project_id": {"type": "string"},
            "name": {"type": "string", "description": "Flat *.md filename, e.g. notes.md"},
        },
        "required": ["project_id", "name"],
    }

    def __init__(self, db: Database, repo_root: Path) -> None:
        self.db = db
        self.repo_root = repo_root

    async def execute(self, **kwargs) -> ToolResult:
        project_id = str(kwargs.get("project_id", ""))
        name = str(kwargs.get("name", ""))
        if self.db.get_project(project_id) is None:
            return ToolResult.failure(f"project not found: {project_id}")
        if "/" in name or "\\" in name or not name.endswith(".md"):
            return ToolResult.failure("memory note name must be a flat *.md filename")
        store = open_project_memory(self.repo_root, project_id)
        try:
            content = store.read(name)
        except (FileNotFoundError, OSError):
            return ToolResult.failure(f"file not found: {name}")
        if len(content) > 60_000:
            content = content[:60_000] + "\n...[truncated]"
        return ToolResult.success(content)


class ProjectMemoryWriteTool(BaseTool):
    name = "project_memory_write"
    description = (
        "Append to (or overwrite) a memory note of a project. Writes a flat *.md "
        "file under memory/projects/<id>/. Use to record durable project knowledge."
    )
    parameters = {
        "type": "object",
        "properties": {
            "project_id": {"type": "string"},
            "name": {"type": "string", "description": "Flat *.md filename, e.g. notes.md"},
            "content": {"type": "string", "description": "Markdown to write/append"},
            "overwrite": {"type": "boolean", "default": False},
        },
        "required": ["project_id", "name", "content"],
    }

    def __init__(self, db: Database, repo_root: Path) -> None:
        self.db = db
        self.repo_root = repo_root

    async def execute(self, **kwargs) -> ToolResult:
        project_id = str(kwargs.get("project_id", ""))
        name = str(kwargs.get("name", ""))
        content = str(kwargs.get("content", ""))
        overwrite = bool(kwargs.get("overwrite", False))
        if self.db.get_project(project_id) is None:
            return ToolResult.failure(f"project not found: {project_id}")
        name = (name or "").strip()
        if not name:
            return ToolResult.failure("project_memory_write failed: memory note name is required")
        if "/" in name or "\\" in name or not name.endswith(".md"):
            return ToolResult.failure("project_memory_write failed: name must be a flat *.md file")
        store = open_project_memory(self.repo_root, project_id)
        target = store.memory_dir / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if not overwrite and target.exists():
            body = target.read_text(encoding="utf-8", errors="replace").rstrip() + "\n\n" + content.strip()
        else:
            body = content
        target.write_text(body, encoding="utf-8")
        return ToolResult.success(f"wrote memory/projects/{project_id}/{name}")


class KanbanBoardTool(BaseTool):
    name = "kanban_board"
    description = (
        "Read a project's kanban board: columns and cards (ids, titles, positions). "
        "Pass board_id to read one board. Lists all boards for the project when "
        "board_id is omitted."
    )
    parameters = {
        "type": "object",
        "properties": {
            "project_id": {"type": "string"},
            "board_id": {"type": "string", "default": None},
        },
        "required": ["project_id"],
    }

    def __init__(self, db: Database, repo_root: Path) -> None:
        self.db = db
        self.repo_root = repo_root

    async def execute(self, **kwargs) -> ToolResult:
        project_id = str(kwargs.get("project_id", ""))
        board_id = kwargs.get("board_id")
        if self.db.get_project(project_id) is None:
            return ToolResult.failure(f"project not found: {project_id}")
        if board_id:
            board = self.db.get_kanban_board(board_id)
            if board is None or board["project_id"] != project_id:
                return ToolResult.failure(f"board not found: {board_id}")
            boards = [_board_dto(self.db, board)]
        else:
            boards = self._list_boards(project_id)
        if not boards:
            return ToolResult.success(
                f"project {project_id} has no kanban boards. "
                "Create one with kaanban_create_board."
            )
        return ToolResult.success(_render_boards(boards))

    def _list_boards(self, project_id: str):
        return [_board_dto(self.db, board) for board in self.db.list_kanban_boards(project_id)]


def _board_dto(db: Database, board) -> dict:
    return {
        "id": board["id"],
        "project_id": board["project_id"],
        "name": board["name"],
        "description": board["description"] or "",
        "created_at": board["created_at"],
        "columns": [
            {
                "id": col["id"],
                "name": col["name"],
                "position": int(col["position"]),
                "cards": [
                    {
                        "id": c["id"],
                        "title": c["title"],
                        "description": c["description"] or "",
                        "task_id": c["task_id"],
                        "position": int(c["position"]),
                        "column_id": col["id"],
                        "created_at": c["created_at"],
                    }
                    for c in db.list_kanban_cards(column_id=col["id"])
                ],
            }
            for col in db.list_kanban_columns(board["id"])
        ],
    }


def _render_boards(boards) -> str:
    out: list[str] = []
    for b in boards:
        out.append(f"board: {b['name']} (id={b['id'][:8]})")
        for col in b["columns"]:
            cards = ", ".join(
                f"{c['title']} (id={c['id'][:8]})" for c in col["cards"]
            )
            out.append(f"  [{col['name']}] {cards or '(empty)'}")
    return "\n".join(out)


class KanbanCreateBoardTool(BaseTool):
    name = "kanban_create_board"
    description = (
        "Create a kanban board for a project (seeds Backlog / In Progress / Done "
        "columns). Returns the new board id."
    )
    parameters = {
        "type": "object",
        "properties": {
            "project_id": {"type": "string"},
            "name": {"type": "string"},
            "description": {"type": "string", "default": ""},
        },
        "required": ["project_id", "name"],
    }

    def __init__(self, db: Database, repo_root: Path) -> None:
        self.db = db
        self.repo_root = repo_root

    async def execute(self, **kwargs) -> ToolResult:
        project_id = str(kwargs.get("project_id", ""))
        name = str(kwargs.get("name", ""))
        description = str(kwargs.get("description", ""))
        name = (name or "").strip()
        if not name:
            return ToolResult.failure("kanban_create_board failed: board name is required")
        if self.db.get_project(project_id) is None:
            return ToolResult.failure(f"project not found: {project_id}")
        board_id = self.db.create_kanban_board(project_id, name, description)
        return ToolResult.success(f"created board {name} id={board_id[:8]}")


class KanbanAddCardTool(BaseTool):
    name = "kanban_add_card"
    description = (
        "Add a card to a kanban column of a project's board. Looks up a column by "
        "id or by exact column name within a board."
    )
    parameters = {
        "type": "object",
        "properties": {
            "project_id": {"type": "string"},
            "board_id": {"type": "string"},
            "column": {"type": "string", "description": "Column id or exact column name"},
            "title": {"type": "string"},
            "description": {"type": "string", "default": ""},
        },
        "required": ["project_id", "board_id", "column", "title"],
    }

    def __init__(self, db: Database, repo_root: Path) -> None:
        self.db = db
        self.repo_root = repo_root

    async def execute(self, **kwargs) -> ToolResult:
        project_id = str(kwargs.get("project_id", ""))
        board_id = str(kwargs.get("board_id", ""))
        column = str(kwargs.get("column", ""))
        title = str(kwargs.get("title", "")).strip()
        description = str(kwargs.get("description", ""))
        if self.db.get_project(project_id) is None:
            return ToolResult.failure(f"project not found: {project_id}")
        board = self.db.get_kanban_board(board_id)
        if board is None or board["project_id"] != project_id:
            return ToolResult.failure(f"board not found: {board_id}")
        col_id = _find_column(_board_dto(self.db, board), column)
        if col_id is None:
            return ToolResult.failure(f"column not found: {column}")
        if not title:
            return ToolResult.failure("card title is required")
        card_id = self.db.add_kanban_card(board_id, col_id, title, description)
        return ToolResult.success(f"added card '{title}' id={card_id[:8]}")


def _find_column(board, column: str) -> str | None:
    if not column:
        return None
    for col in board["columns"]:
        if col["id"] == column or col["id"].startswith(column) or col["name"] == column:
            return col["id"]
    return None
