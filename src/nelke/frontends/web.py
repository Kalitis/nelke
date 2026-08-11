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
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sse_starlette.sse import EventSourceResponse

from nelke.config import Settings, load_env_files, load_profiles
from nelke.core import services
from nelke.core.services import Callbacks

load_env_files()

_PKG_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = _PKG_DIR.parent / "templates"
STATIC_DIR = _PKG_DIR.parent / "static"


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

    # ---- pages -------------------------------------------------------------
    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request) -> HTMLResponse:
        profiles = _profile_rows()
        return templates.TemplateResponse(request, "index.html", {"profiles": profiles})

    @app.get("/memory", response_class=HTMLResponse)
    async def memory_page(request: Request) -> HTMLResponse:
        repo = state.repo_path or services.find_repo(state.settings)
        files = services.memory_overview(repo)
        return templates.TemplateResponse(request, "memory.html", {"files": files})

    @app.get("/review/{request_id}", response_class=HTMLResponse)
    async def review_page(request: Request, request_id: str) -> HTMLResponse:
        review = services.get_review(state.settings, request_id, repo_path=state.repo_path)
        if review is None:
            return templates.TemplateResponse(request, "review.html", {"review": None})
        return templates.TemplateResponse(request, "review.html", {"review": review})

    # ---- JSON api ----------------------------------------------------------
    @app.get("/api/health")
    async def health() -> dict[str, Any]:
        return {"ok": True}

    @app.get("/api/profiles")
    async def api_profiles() -> list[dict[str, str]]:
        return _profile_rows()

    @app.get("/api/reviews")
    async def api_reviews() -> list[dict[str, Any]]:
        return services.list_open_reviews(state.settings)

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

            async def runner() -> None:
                try:
                    result, _session_id = await services.run_task(
                        text, state.settings, profile,
                        frontend_name="web",
                        callbacks=Callbacks(on_token=on_token, on_tool=on_tool,
                                            on_tool_result=on_tool_result, stream=True),
                        repo=state.repo_path,
                        llm_factory=state.llm_factory,
                    )
                    await queue.put({
                        "event": "done",
                        "data": json.dumps({"answer": result.answer, "usage": result.usage}),
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

    @app.post("/api/improve")
    async def api_improve(payload: dict[str, Any]) -> JSONResponse:
        objective = str(payload.get("objective", "")).strip()
        if not objective:
            return JSONResponse({"error": "empty objective"}, status_code=400)
        auto = bool(payload.get("auto_approve", False))

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

        def push_events(ev: Any) -> None:
            """Broadcast each live cycle event to all SSE subscribers."""
            data = {"cycle_id": getattr(ev, "cycle_id", ""), "kind": ev.kind,
                    "message": ev.message, "payload": ev.data}
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
                )
                # Notify any SSE subscribers that the cycle finished.
                for _, events in state.stream_events.items():
                    for q in list(events.values()):
                        q.put_nowait({
                            "event": "cycle_result",
                            "data": json.dumps({
                                "cycle_id": result.cycle_id, "status": result.status,
                                "branch": result.branch, "steps": result.steps,
                            }),
                        })
            except Exception:  # noqa: BLE001 - the gate future is abandoned on failure
                pass

        asyncio.create_task(runner())
        return JSONResponse({"status": "started"})

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
            async def gen() -> AsyncIterator[dict[str, Any]]:
                while not queue.empty():
                    yield queue.get_nowait()
            return EventSourceResponse(gen())

        state.stream_events.setdefault("cycle", {})[sub_id] = queue

        async def gen() -> AsyncIterator[dict[str, Any]]:
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

        return EventSourceResponse(gen())

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
    out: dict[str, Any] = {}
    for k, v in list(args.items())[:2]:
        text = str(v)
        out[k] = text[:60] + "..." if len(text) > 60 else text
    return out


async def _error_event(message: str) -> AsyncIterator[dict[str, Any]]:
    yield {"event": "error", "data": json.dumps({"message": message})}


def launch(host: str | None = None, port: int | None = None) -> None:
    """Run the web server (uvicorn). Host/port from env or args."""
    import uvicorn

    host = host or os.environ.get("NELKE_WEB_HOST", "127.0.0.1")
    port = port or int(os.environ.get("NELKE_WEB_PORT", "8000"))
    app = create_app()
    uvicorn.run(app, host=host, port=port)
