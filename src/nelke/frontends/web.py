"""Web frontend (FastAPI + Jinja2 + SSE) — a thin I/O adapter over the Nelke core.

Endpoints:
  GET  /                 chat UI (Jinja2)
  GET  /memory           memory browser
  GET  /review/{id}      human-gate page (diff + AI verdict + approve/reject)
  GET  /api/profiles     provider switcher data
  GET  /api/reviews      open human review requests
  POST /api/chat         stream tokens/tools/usage as SSE
  POST /api/improve      run a self-improvement cycle (human gate via SSE + /api/review)
  POST /api/review/{id}  resolve the human gate (approved/rejected)
  GET  /api/health       liveness probe

The cycle's ``human_approve`` callable is async: it parks on an
``asyncio.Future`` stored per cycle id; ``POST /api/review/{id}`` resolves it.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sse_starlette.sse import EventSourceResponse

from nelke.config import Settings, load_env_files, load_profiles
from nelke.core import services
from nelke.core.services import Callbacks

load_env_files()

_PKG_DIR = Path(__file__).resolve().parent


def _web_asset_dir(name: str, marker: str) -> Path:
    """Locate the templates/ or static/ directory (single source of truth).

    The canonical copies live at the repo root (``<repo>/<name>``). For wheels
    built from this repo the same files are bundled into the package
    (``nelke/<name>``), so fall back to that for installed installs.
    """
    from nelke.core import services

    for root in (services.find_repo(), Path.cwd(), _PKG_DIR.parent):
        cand = root / name
        if (cand / marker).is_file():
            return cand
    return _PKG_DIR.parent / name


TEMPLATES_DIR = _web_asset_dir("templates", "base.html")
STATIC_DIR = _web_asset_dir("static", "app.js")


def _spa_dist_dir() -> Path | None:
    """Locate the built SPA bundle (``static/dist``) if it exists.

    Searched in the same roots as the other assets. Returns ``None`` when no
    build is present (dev checkout without Node, or ``NELKE_SKIP_WEB_BUILD``).
    Honours ``NELKE_WEB_LEGACY=1`` to force the legacy Jinja2 UI (used by
    tests and as an escape hatch when the SPA misbehaves).
    """
    if _legacy_forced():
        return None
    from nelke.core import services

    for root in (services.find_repo(), Path.cwd(), _PKG_DIR.parent):
        cand = root / "static" / "dist"
        if (cand / "index.html").is_file():
            return cand
    return None


def _spa_dev_mode() -> bool:
    """True when the SPA should be served from the Vite dev server (:5173)."""
    return os.environ.get("NELKE_WEB_DEV", "").strip().lower() in {"1", "true", "yes"}


def _legacy_forced() -> bool:
    """True when the legacy Jinja2 UI should be served instead of the SPA."""
    return os.environ.get("NELKE_WEB_LEGACY", "").strip().lower() in {"1", "true", "yes"}


def _spa_index_html() -> str:
    """HTML for the SPA entry: dev-server redirect or the built bundle."""
    if _spa_dev_mode():
        # Redirect to the Vite dev server which serves the SPA with HMR.
        return (
            "<!doctype html><html><head>"
            '<meta http-equiv="refresh" content="0; url=http://localhost:5173/">'
            "<title>Nelke (dev)</title></head><body></body></html>"
        )
    dist = _spa_dist_dir()
    if dist is None:
        return ""
    return (dist / "index.html").read_text(encoding="utf-8")


def _spa_response_or_none() -> HTMLResponse | None:
    """Return the SPA entry as an HTMLResponse when a build/dev mode is active.

    The SPA owns client-side routing for ``/``, ``/cycles``, ``/cycles/{id}``,
    ``/memory`` and ``/review/{id}``; when it is available each of those page
    routes serves the SPA bundle so the React router can take over. Legacy
    Jinja2 pages remain the fallback when ``NELKE_WEB_LEGACY=1`` (in which
    case ``_spa_index_html()`` already returns ``""``).
    """
    html = _spa_index_html()
    if not html:
        return None
    return HTMLResponse(html)


# --------------------------------------------------------------------------- #
# App state
# --------------------------------------------------------------------------- #
class AppState:
    """Per-process state: dependencies + in-flight human-gate futures."""

    def __init__(
        self,
        settings: Settings | None = None,
        llm_factory: Callable[[str | None], Any] | None = None,
        governance: Any = None,
        repo_path: Path | None = None,
    ) -> None:
        self.settings = settings or Settings()
        self.llm_factory = llm_factory or services._llm_factory_default
        self.governance = governance
        self.repo_path = repo_path
        # cycle_id -> (Future, review_request_id once known)
        self.gate_futures: dict[str, asyncio.Future[bool]] = {}
        self.gate_request_ids: dict[str, str] = {}
        # stream-kind -> subscriber_id -> asyncio.Queue (SSE broadcast fan-out)
        self.stream_events: dict[str, dict[str, asyncio.Queue[dict[str, Any]]]] = {}


def _safe_json(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else {}
    except (ValueError, TypeError):
        return {}


def _truthy(value: str | None) -> bool:
    """Parse a query-param truth value: ``1``/``true``/``yes`` (case-insensitive)."""
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


def new_id_short() -> str:
    import uuid

    return uuid.uuid4().hex[:8]


def create_app(state: AppState | None = None) -> FastAPI:
    state = state or AppState()
    app = FastAPI(title="Nelke", docs_url=None, redoc_url=None)
    app.state.nelke = state

    templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
    app.state.templates = templates
    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    # Serve the built SPA assets (Vite emits hashed bundles under /assets)
    # when a build is available. The SPA entry itself is served by `/`.
    spa_dist = _spa_dist_dir()
    if spa_dist is not None and not _spa_dev_mode():
        app.mount("/assets", StaticFiles(directory=str(spa_dist / "assets")), name="spa-assets")
        # favicon + any other root-level static from the build.
        app.mount("/favicon.svg", StaticFiles(directory=str(spa_dist), html=False), name="favicon")

    # ---- pages -------------------------------------------------------------
    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request) -> HTMLResponse:
        # Prefer the SPA when a build (or dev-mode redirect) is available;
        # fall back to the legacy Jinja2 chat UI otherwise.
        spa_html = _spa_index_html()
        if spa_html:
            return HTMLResponse(spa_html)
        profiles = _profile_rows()
        return templates.TemplateResponse(request, "index.html", {"profiles": profiles})

    @app.get("/memory", response_class=HTMLResponse)
    async def memory_page(request: Request) -> HTMLResponse:
        spa = _spa_response_or_none()
        if spa is not None:
            return spa
        repo = state.repo_path or services.find_repo(state.settings)
        files = services.memory_overview(repo)
        return templates.TemplateResponse(request, "memory.html", {"files": files})

    @app.get("/review/{request_id}", response_class=HTMLResponse)
    async def review_page(request: Request, request_id: str) -> HTMLResponse:
        spa = _spa_response_or_none()
        if spa is not None:
            return spa
        review = services.get_review(state.settings, request_id, repo_path=state.repo_path)
        if review is None:
            return templates.TemplateResponse(request, "review.html", {"review": None})
        return templates.TemplateResponse(request, "review.html", {"review": review})

    @app.get("/cycles", response_class=HTMLResponse)
    async def cycles_page(request: Request) -> HTMLResponse:
        spa = _spa_response_or_none()
        if spa is not None:
            return spa
        cycles = services.list_cycles(state.settings)
        return templates.TemplateResponse(request, "cycles.html", {"cycles": cycles})

    @app.get("/cycles/{cycle_id}", response_class=HTMLResponse)
    async def cycle_detail_page(request: Request, cycle_id: str) -> HTMLResponse:
        spa = _spa_response_or_none()
        if spa is not None:
            return spa
        cycle = services.get_cycle_detail(state.settings, cycle_id)
        return templates.TemplateResponse(request, "cycle_detail.html", {"cycle": cycle})

    # ---- JSON api ----------------------------------------------------------
    @app.get("/api/health")
    async def health() -> dict[str, Any]:
        from nelke import __version__

        return {"ok": True, "version": __version__}

    @app.get("/api/profiles")
    async def api_profiles() -> list[dict[str, str]]:
        return _profile_rows()

    @app.get("/api/memory")
    async def api_memory() -> list[dict[str, Any]]:
        """Memory file index (name + size) for the SPA memory viewer."""
        repo = state.repo_path or services.find_repo(state.settings)
        return services.memory_overview(repo)

    @app.get("/api/memory/{name:path}")
    async def api_memory_file(name: str) -> JSONResponse:
        """Contents of a single memory file (markdown), 404 if unknown.

        ``name`` is the posix-relative path (e.g. ``chats/<id>/roadmap.md``).
        ``services.memory_file_content`` only accepts files actually present
        in the memory dir, so crafted paths cannot escape it.
        """
        repo = state.repo_path or services.find_repo(state.settings)
        content = services.memory_file_content(repo, name)
        if content is None:
            return JSONResponse({"error": "memory file not found"}, status_code=404)
        return JSONResponse({"name": name, "content": content})

    @app.get("/api/reviews")
    async def api_reviews() -> list[dict[str, Any]]:
        return services.list_open_reviews(state.settings)

    @app.get("/api/usage")
    async def api_usage(
        session_id: str | None = None, cycle_id: str | None = None,
    ) -> dict[str, Any]:
        """DB-backed token usage: running totals + the most recent per-call events."""
        db = services.open_db(state.settings)
        totals = db.usage_totals(session_id=session_id, cycle_id=cycle_id)
        events = db.list_usage(session_id=session_id, cycle_id=cycle_id)
        return {
            "totals": totals,
            "events": [
                {
                    "id": r["id"],
                    "prompt_tokens": r["prompt_tokens"],
                    "completion_tokens": r["completion_tokens"],
                    "total_tokens": r["total_tokens"],
                    "cache_read_tokens": r["cache_read_tokens"],
                    "cache_read_pct": r["cache_read_pct"],
                    "created_at": r["created_at"],
                    "session_id": r["session_id"],
                    "cycle_id": r["cycle_id"],
                }
                for r in events[-50:]
            ],
        }

    @app.post("/api/chat")
    async def api_chat(payload: dict[str, Any]) -> EventSourceResponse:
        text = str(payload.get("text", "")).strip()
        profile = payload.get("profile")
        if not text:
            return EventSourceResponse(_error_event("empty text"))

        async def stream() -> AsyncIterator[dict[str, Any]]:
            queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
            done = object()

            def on_token(tok: str) -> None:
                queue.put_nowait({"event": "token", "data": json.dumps({"text": tok})})

            def on_tool(name: str, args: dict[str, Any]) -> None:
                queue.put_nowait({"event": "tool", "data": json.dumps({"name": name, "args": _trim(args)})})

            def on_tool_result(name: str, args: dict[str, Any], result: str) -> None:
                queue.put_nowait({
                    "event": "tool_result",
                    "data": json.dumps({"name": name, "snippet": result[:200]}),
                })

            def on_usage(usage: dict[str, Any]) -> None:
                queue.put_nowait({"event": "usage", "data": json.dumps(usage)})

            async def runner() -> None:
                try:
                    result, session_id = await services.run_task(
                        text, state.settings, profile,
                        frontend_name="web",
                        callbacks=Callbacks(on_token=on_token, on_tool=on_tool,
                                            on_tool_result=on_tool_result,
                                            on_usage=on_usage, stream=True),
                        repo=state.repo_path,
                        llm_factory=state.llm_factory,
                    )
                    await queue.put({
                        "event": "done",
                        "data": json.dumps({"answer": result.answer, "usage": result.usage,
                                            "session_id": session_id}),
                    })
                except Exception as exc:  # noqa: BLE001 - surface to the client
                    await queue.put({"event": "error", "data": json.dumps({"message": str(exc)})})
                finally:
                    await queue.put(done)  # type: ignore[arg-type]

            asyncio.create_task(runner())
            while True:
                item = await queue.get()
                if item is done:
                    break
                yield item

        return EventSourceResponse(stream())

    # ---- chats (multiple conversations, each with history + memory) --------
    # Chats are shared across all frontends (web/TUI/Telegram): a conversation
    # started anywhere is listed and resumable here, and vice versa.
    @app.get("/api/chats")
    async def api_chats() -> list[dict[str, Any]]:
        return services.list_chats(state.settings)

    @app.post("/api/chats")
    async def api_create_chat(payload: dict[str, Any]) -> JSONResponse:
        title = str(payload.get("title") or "").strip() or None
        chat_id = services.create_chat(state.settings, title=title, frontend="web")
        return JSONResponse({"id": chat_id, "title": title or "New chat"})

    @app.get("/api/chats/{chat_id}")
    async def api_chat_detail(chat_id: str) -> JSONResponse:
        chat = services.get_chat(state.settings, chat_id, repo=state.repo_path)
        if chat is None:
            return JSONResponse({"error": "chat not found"}, status_code=404)
        return JSONResponse(chat)

    @app.patch("/api/chats/{chat_id}")
    async def api_rename_chat(chat_id: str, payload: dict[str, Any]) -> JSONResponse:
        title = str(payload.get("title") or "").strip()
        if not title:
            return JSONResponse({"error": "empty title"}, status_code=400)
        if not services.rename_chat(state.settings, chat_id, title):
            return JSONResponse({"error": "chat not found"}, status_code=404)
        return JSONResponse({"ok": True, "id": chat_id, "title": title})

    @app.delete("/api/chats/{chat_id}")
    async def api_delete_chat(chat_id: str) -> JSONResponse:
        if not services.delete_chat(state.settings, chat_id):
            return JSONResponse({"error": "chat not found"}, status_code=404)
        return JSONResponse({"ok": True})

    @app.post("/api/chats/{chat_id}/messages")
    async def api_chat_message(chat_id: str, payload: dict[str, Any]) -> Response:
        """Stream one turn inside an existing chat (persisted transcript)."""
        db = services.open_db(state.settings)
        if db.get_session(chat_id) is None:
            return _error_response_not_found()
        text = str(payload.get("text", "")).strip()
        profile = payload.get("profile")
        if not text:
            return EventSourceResponse(_error_event("empty text"))

        async def stream() -> AsyncIterator[dict[str, Any]]:
            queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
            done = object()

            def on_token(tok: str) -> None:
                queue.put_nowait({"event": "token", "data": json.dumps({"text": tok})})

            def on_tool(name: str, args: dict[str, Any]) -> None:
                queue.put_nowait({"event": "tool", "data": json.dumps({"name": name, "args": _trim(args)})})

            def on_tool_result(name: str, args: dict[str, Any], result: str) -> None:
                queue.put_nowait({
                    "event": "tool_result",
                    "data": json.dumps({"name": name, "snippet": result[:200]}),
                })

            def on_usage(usage: dict[str, Any]) -> None:
                queue.put_nowait({"event": "usage", "data": json.dumps(usage)})

            async def runner() -> None:
                try:
                    result, cid, user_msg_id = await services.run_chat_turn(
                        text, state.settings, profile, chat_id,
                        frontend_name="web",
                        callbacks=Callbacks(on_token=on_token, on_tool=on_tool,
                                            on_tool_result=on_tool_result,
                                            on_usage=on_usage, stream=True),
                        repo=state.repo_path,
                        llm_factory=state.llm_factory,
                        parent_message_id=payload.get("parent_message_id"),
                    )
                    leaf_id = None
                    leaf = services.open_db(state.settings).active_leaf(cid)
                    if leaf is not None:
                        leaf_id = leaf["id"]
                    await queue.put({
                        "event": "done",
                        "data": json.dumps({
                            "answer": result.answer, "usage": result.usage,
                            "chat_id": cid,
                            "user_message_id": user_msg_id,
                            "assistant_message_id": leaf_id,
                        }),
                    })
                except Exception as exc:  # noqa: BLE001 - surface to the client
                    await queue.put({"event": "error", "data": json.dumps({"message": str(exc)})})
                finally:
                    await queue.put(done)  # type: ignore[arg-type]

            asyncio.create_task(runner())
            while True:
                item = await queue.get()
                if item is done:
                    break
                yield item

        return EventSourceResponse(stream())

    # ---- projects (group chats + per-project memory) -----------------------
    @app.get("/api/projects")
    async def api_projects() -> list[dict[str, Any]]:
        return services.list_projects(state.settings)

    @app.post("/api/projects")
    async def api_create_project(payload: dict[str, Any]) -> JSONResponse:
        name = str(payload.get("name", "")).strip()
        if not name:
            return JSONResponse({"error": "project name is required"}, status_code=400)
        description = str(payload.get("description") or "").strip()
        stage = str(payload.get("stage") or "").strip()
        pid = services.create_project(
            state.settings, name=name, description=description, stage=stage,
            repo=state.repo_path,
        )
        return JSONResponse({"id": pid, "name": name})

    @app.get("/api/projects/{project_id}")
    async def api_project_detail(project_id: str) -> JSONResponse:
        project = services.get_project(state.settings, project_id, repo=state.repo_path)
        if project is None:
            return JSONResponse({"error": "project not found"}, status_code=404)
        return JSONResponse(project)

    @app.patch("/api/projects/{project_id}")
    async def api_update_project(project_id: str, payload: dict[str, Any]) -> JSONResponse:
        ok = services.update_project(
            state.settings, project_id,
            name=payload.get("name"),
            description=payload.get("description"),
            stage=payload.get("stage"),
        )
        if not ok:
            return JSONResponse({"error": "project not found"}, status_code=404)
        return JSONResponse({"ok": True, "id": project_id})

    @app.delete("/api/projects/{project_id}")
    async def api_delete_project(project_id: str) -> JSONResponse:
        if not services.delete_project(state.settings, project_id):
            return JSONResponse({"error": "project not found"}, status_code=404)
        return JSONResponse({"ok": True})

    @app.post("/api/projects/{project_id}/chats/{chat_id}")
    async def api_link_chat(project_id: str, chat_id: str) -> JSONResponse:
        """Attach a chat to a project."""
        if not services.attach_chat_to_project(
            state.settings, chat_id=chat_id, project_id=project_id,
        ):
            return JSONResponse({"error": "project or chat not found"}, status_code=404)
        return JSONResponse({"ok": True, "project_id": project_id, "chat_id": chat_id})

    @app.post("/api/projects/{project_id}/memory")
    async def api_set_project_memory(project_id: str, payload: dict[str, Any]) -> JSONResponse:
        """Write a ``.md`` note into the project's memory directory."""
        name = str(payload.get("name") or "notes.md").strip()
        content = str(payload.get("content") or "")
        try:
            ok = services.set_project_memory(
                state.settings, project_id, name, content,
                append=bool(payload.get("append", False)),
                repo=state.repo_path,
            )
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        if not ok:
            return JSONResponse({"error": "project not found"}, status_code=404)
        return JSONResponse({"ok": True, "name": name})

    # ---- message tree: edit / regenerate / branch (swipe) / delete ----------
    @app.get("/api/chats/{chat_id}/tree")
    async def api_chat_tree(chat_id: str) -> JSONResponse:
        """Full non-deleted message tree of a chat (all branches)."""
        db = services.open_db(state.settings)
        if db.get_session(chat_id) is None:
            return _error_response_not_found()
        tree = services.build_message_tree(db, chat_id)
        leaf = db.active_leaf(chat_id)
        return JSONResponse({
            "nodes": tree["nodes"], "children": tree["children"],
            "root_id": tree["root_id"],
            "active_leaf_id": leaf["id"] if leaf else None,
        })

    @app.patch("/api/chats/{chat_id}/messages/{message_id}")
    async def api_edit_message(chat_id: str, message_id: str, payload: dict[str, Any]) -> JSONResponse:
        """Edit a user message: soft-delete its subtree and create a sibling.

        The frontend then POSTs ``/api/chats/{chat_id}/messages`` with the
        returned ``message_id`` as ``parent_message_id`` to generate the new
        assistant answer on the edited branch.
        """
        content = str(payload.get("content", "")).strip()
        if not content:
            return JSONResponse({"error": "empty content"}, status_code=400)
        result = services.edit_message(state.settings, chat_id, message_id, content)
        if result is None:
            return JSONResponse({"error": "message not found or not editable"}, status_code=404)
        return JSONResponse(result)

    @app.post("/api/chats/{chat_id}/messages/{message_id}/regenerate")
    async def api_regenerate_message(chat_id: str, message_id: str, payload: dict[str, Any]) -> Response:
        """Stream a fresh assistant answer for an existing assistant message.

        Soft-deletes the old assistant subtree and re-runs the parent turn,
        streaming tokens/tools/usage exactly like ``/api/chats/{id}/messages``.
        """
        db = services.open_db(state.settings)
        if db.get_session(chat_id) is None:
            return _error_response_not_found()
        row = db.get_message(message_id)
        if row is None or row["session_id"] != chat_id or row["role"] != "assistant":
            return JSONResponse({"error": "assistant message not found"}, status_code=404)
        profile = payload.get("profile")

        async def stream() -> AsyncIterator[dict[str, Any]]:
            queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
            done = object()

            def on_token(tok: str) -> None:
                queue.put_nowait({"event": "token", "data": json.dumps({"text": tok})})

            def on_tool(name: str, args: dict[str, Any]) -> None:
                queue.put_nowait({"event": "tool", "data": json.dumps({"name": name, "args": _trim(args)})})

            def on_tool_result(name: str, args: dict[str, Any], result: str) -> None:
                queue.put_nowait({
                    "event": "tool_result",
                    "data": json.dumps({"name": name, "snippet": result[:200]}),
                })

            def on_usage(usage: dict[str, Any]) -> None:
                queue.put_nowait({"event": "usage", "data": json.dumps(usage)})

            async def runner() -> None:
                try:
                    result, cid, _ = await services.regenerate_response(
                        state.settings, profile, chat_id, message_id,
                        frontend_name="web",
                        callbacks=Callbacks(on_token=on_token, on_tool=on_tool,
                                            on_tool_result=on_tool_result,
                                            on_usage=on_usage, stream=True),
                        repo=state.repo_path,
                        llm_factory=state.llm_factory,
                    )
                    leaf = services.open_db(state.settings).active_leaf(cid)
                    leaf_id = leaf["id"] if leaf else None
                    await queue.put({
                        "event": "done",
                        "data": json.dumps({
                            "answer": result.answer, "usage": result.usage,
                            "chat_id": cid, "assistant_message_id": leaf_id,
                        }),
                    })
                except Exception as exc:  # noqa: BLE001 - surface to the client
                    await queue.put({"event": "error", "data": json.dumps({"message": str(exc)})})
                finally:
                    await queue.put(done)  # type: ignore[arg-type]

            asyncio.create_task(runner())
            while True:
                item = await queue.get()
                if item is done:
                    break
                yield item

        return EventSourceResponse(stream())

    @app.delete("/api/chats/{chat_id}/messages/{message_id}")
    async def api_delete_message(chat_id: str, message_id: str) -> JSONResponse:
        """Soft-delete a message and its subtree; returns the new active leaf."""
        result = services.delete_message(state.settings, chat_id, message_id)
        if result is None:
            return JSONResponse({"error": "message not found"}, status_code=404)
        return JSONResponse(result)

    @app.post("/api/chats/{chat_id}/messages/{message_id}/activate")
    async def api_activate_message(chat_id: str, message_id: str) -> JSONResponse:
        """Switch the visible branch (swipe) to the one through ``message_id``."""
        result = services.set_active_message(state.settings, chat_id, message_id)
        if result is None:
            return JSONResponse({"error": "message not found"}, status_code=404)
        return JSONResponse(result)

    @app.post("/api/improve")
    async def api_improve(payload: dict[str, Any]) -> JSONResponse:
        objective = str(payload.get("objective", "")).strip()
        if not objective:
            return JSONResponse({"error": "empty objective"}, status_code=400)
        auto = bool(payload.get("auto_approve", False))
        # Optional project attribution. When omitted, run_cycle attaches the
        # cycle to the default "nelke" dogfooding project automatically.
        project_id = str(payload.get("project_id", "")).strip() or None

        async def human_gate(req: Any) -> bool:
            if auto:
                return True
            future: asyncio.Future[bool] = asyncio.get_event_loop().create_future()
            state.gate_futures[req.cycle_id] = future
            # The review request row is created by the cycle just before this
            # callback fires; record its id so the resolve endpoint can find it.
            db = services.open_db(state.settings)
            human_reqs = [r for r in db.list_review_requests(cycle_id=req.cycle_id, open_only=True)
                          if r["kind"] == "human"]
            if human_reqs:
                state.gate_request_ids[req.cycle_id] = human_reqs[0]["id"]
            return await future

        # The cycle_id is minted inside services.run_cycle; capture it from the
        # first `cycle_start` event so we can return it to the caller. The UI
        # uses it to navigate straight to the cycle's detail page.
        cycle_id_future: asyncio.Future[str] = asyncio.get_event_loop().create_future()

        def push_events(ev: Any) -> None:
            """Broadcast each live cycle event to all SSE subscribers."""
            data = {"cycle_id": getattr(ev, "cycle_id", ""), "kind": ev.kind,
                    "message": ev.message, "payload": ev.data}
            if ev.kind == "cycle_start" and not cycle_id_future.done():
                cycle_id_future.set_result(data["cycle_id"])
            for _, events in state.stream_events.items():
                for q in list(events.values()):
                    try:
                        q.put_nowait({"event": "cycle_event",
                                      "data": json.dumps(data)})
                    except Exception:  # noqa: BLE001
                        pass

        async def runner() -> None:
            try:
                result = await services.run_cycle(
                    objective, state.settings, None,
                    human_approve=human_gate,
                    repo_path=state.repo_path,
                    llm_factory=state.llm_factory,
                    governance=state.governance,
                    on_event=push_events,
                    # The web frontend renders one card per parallel worker;
                    # the planner splits the objective into <= 6 slices.
                    mode="parallel",
                    project_id=project_id,
                )
                # Notify any SSE subscribers that the cycle finished.
                for _, events in state.stream_events.items():
                    for q in list(events.values()):
                        q.put_nowait({
                            "event": "cycle_result",
                            "data": json.dumps({
                                "cycle_id": result.cycle_id, "status": result.status,
                                "branch": result.branch, "steps": result.steps,
                                "project_id": result.project_id,
                            }),
                        })
            except Exception:  # noqa: BLE001 - the gate future is abandoned on failure
                if not cycle_id_future.done():
                    cycle_id_future.set_result("")

        asyncio.create_task(runner())
        # Wait briefly for the cycle_id so the response can include it. If the
        # engine is slow to emit `cycle_start` (or fails before that), fall
        # back to a bare "started" and let the UI refresh the list instead.
        cycle_id = ""
        try:
            cycle_id = await asyncio.wait_for(cycle_id_future, timeout=2.0)
        except asyncio.TimeoutError:
            pass
        body: dict[str, Any] = {"status": "started"}
        if cycle_id:
            body["cycle_id"] = cycle_id
        return JSONResponse(body)

    @app.get("/api/cycles/stream")
    async def api_cycles_stream(request: Request) -> EventSourceResponse:
        """SSE stream of live cycle progress (persisted trace + live callbacks).

        With ``?replay_only=1`` the generator replays the persisted trace and
        returns immediately instead of entering the live ``while True`` loop.
        That gives the response a finite end so it can be consumed by a plain
        HTTP GET in tests (sse_starlette closes the response once the generator
        returns); production subscribers use the default infinite stream.
        """
        replay_only = _truthy(request.query_params.get("replay_only"))
        db = services.open_db(state.settings)
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        sub_id = new_id_short()

        # Replay the persisted trace so a late subscriber sees current state.
        for cycle in list(db.list_cycles())[:3]:
            events = db.list_cycle_events(cycle["id"], limit=60)
            for ev in events:
                try:
                    payload = json.loads(ev["payload"] or "{}")
                except Exception:  # noqa: BLE001
                    payload = {}
                queue.put_nowait({
                    "event": "cycle_event",
                    "data": json.dumps({"cycle_id": cycle["id"], "kind": ev["kind"],
                                        "message": ev["message"], "payload": payload}),
                })

        if replay_only:
            # Finite: drain the replayed events and end the response.
            async def replay_gen() -> AsyncIterator[dict[str, Any]]:
                while not queue.empty():
                    yield queue.get_nowait()
            return EventSourceResponse(replay_gen())

        state.stream_events.setdefault("cycle", {})[sub_id] = queue

        async def live_gen() -> AsyncIterator[dict[str, Any]]:
            try:
                while True:
                    if await request.is_disconnected():
                        break
                    try:
                        item = await asyncio.wait_for(queue.get(), timeout=10)
                    except asyncio.TimeoutError:
                        yield {"event": "ping", "data": "{}"}
                        continue
                    yield item
            finally:
                state.stream_events.get("cycle", {}).pop(sub_id, None)

        return EventSourceResponse(live_gen())

    @app.get("/api/cycles/list")
    async def api_cycles_list() -> list[dict[str, Any]]:
        """Full self-improvement cycle history (steps + open review links)."""
        return services.list_cycles(state.settings)

    @app.get("/api/cycles/{cycle_id}")
    async def api_cycle_detail(cycle_id: str) -> JSONResponse:
        detail = services.get_cycle_detail(state.settings, cycle_id)
        if detail is None:
            return JSONResponse({"error": "cycle not found"}, status_code=404)
        return JSONResponse(detail)

    @app.get("/api/cycles")
    async def api_cycles() -> list[dict[str, Any]]:
        """Return the persisted cycle progress trace for the web card."""
        db = services.open_db(state.settings)
        out: list[dict[str, Any]] = []
        for cycle in list(db.list_cycles())[:5]:
            events = db.list_cycle_events(cycle["id"], limit=80)
            out.append({
                "id": cycle["id"],
                "objective": cycle["objective"],
                "branch": cycle["branch"],
                "status": cycle["status"],
                "ai_verdict": cycle["ai_verdict"],
                "human_verdict": cycle["human_verdict"],
                "events": [
                    {
                        "id": ev["id"], "kind": ev["kind"], "message": ev["message"],
                        "payload": _safe_json(ev["payload"]), "seq": ev["seq"],
                        "step": _safe_json(ev["payload"]).get("step") or ev["seq"],
                    }
                    for ev in events
                ],
            })
        return out

    @app.post("/api/review/{request_id}")
    async def api_resolve_review(request_id: str, payload: dict[str, Any]) -> JSONResponse:
        decision = str(payload.get("decision", "")).strip().lower()
        if decision not in ("approved", "rejected"):
            return JSONResponse({"error": "decision must be approved or rejected"}, status_code=400)
        # If a cycle is parked on this request, wake it; let the cycle do the merge.
        cycle_id = _cycle_id_for_request(state, request_id)
        if cycle_id is not None and cycle_id in state.gate_futures:
            state.gate_futures[cycle_id].set_result(decision == "approved")
            return JSONResponse({"status": "resolved", "cycle_id": cycle_id, "decision": decision})
        # No live cycle (e.g. cycle ran with no human gate, or was approved via CLI):
        # apply the resolution directly.
        try:
            outcome = services.resolve_review(state.settings, request_id, decision, repo_path=state.repo_path)
        except RuntimeError as exc:
            return JSONResponse({"error": str(exc)}, status_code=404)
        return JSONResponse({
            "status": outcome.status, "cycle_id": outcome.cycle_id,
            "branch": outcome.branch, "error": outcome.error,
        })

    # ---- SPA catch-all -----------------------------------------------------
    # Any non-API GET that no explicit route matched falls back to the SPA
    # entry (client-side router takes over). When no SPA build exists we 404
    # rather than shadow the JSON error path. The response type is deliberately
    # ``Response``: API/static misses return JSON, everything else returns HTML.
    @app.get("/{path:path}")
    async def spa_fallback(path: str) -> Response:
        if path.startswith("api/") or path.startswith("static/") or path.startswith("assets/"):
            return JSONResponse({"error": "not found"}, status_code=404)
        # Legacy pages render server-side; if they matched we never reach here.
        spa_html = _spa_index_html()
        if spa_html:
            return HTMLResponse(spa_html)
        return JSONResponse({"error": "not found"}, status_code=404)

    return app


def _cycle_id_for_request(state: AppState, request_id: str) -> str | None:
    """Find the cycle id whose parked gate corresponds to this request id."""
    for cycle_id, rid in state.gate_request_ids.items():
        if rid == request_id or rid.startswith(request_id) or request_id.startswith(rid):
            return cycle_id
    return None


def _profile_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for name, p in load_profiles().items():
        rows.append({"name": name, "base_url": p.base_url, "model": p.model})
    return rows


def _trim(args: dict[str, Any]) -> dict[str, Any]:
    """Trim tool-call args for the streaming UI.

    Frontend renders each call as a collapsed block; a few keys with a generous
    truncation give enough context to identify the call without leaking whole
    file bodies into the SSE stream.
    """
    out: dict[str, Any] = {}
    for k, v in list(args.items())[:4]:
        text = str(v)
        out[k] = text[:200] + "..." if len(text) > 200 else text
    return out


async def _error_event(message: str) -> AsyncIterator[dict[str, Any]]:
    yield {"event": "error", "data": json.dumps({"message": message})}


def _error_response_not_found() -> JSONResponse:
    return JSONResponse({"error": "chat not found"}, status_code=404)


def launch(host: str | None = None, port: int | None = None) -> None:
    """Run the web server (uvicorn). Host/port from env or args."""
    import uvicorn

    from nelke import __version__

    host = host or os.environ.get("NELKE_WEB_HOST", "127.0.0.1")
    port = port or int(os.environ.get("NELKE_WEB_PORT", "8000"))
    print(f"nelke v{__version__} — web UI at http://{host}:{port}", file=sys.stderr)
    app = create_app()
    uvicorn.run(app, host=host, port=port)
