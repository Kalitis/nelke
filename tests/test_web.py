"""Web frontend tests (FastAPI + SSE) against a mocked LLM and fake governance.

No real model, no real subprocess gate. The app is built with an injectable
``AppState`` so tests drive the same core the live server uses.
"""

from __future__ import annotations

import json
from typing import Any

from conftest import FakeGovernance, final_response, tool_response
from starlette.testclient import TestClient

from nelke.core.llm import LLMResponse
from nelke.frontends.web import AppState, create_app


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #
def _scripted_llm_factory(responses: list[LLMResponse], cycle_reviewer=None):
    """Return an llm_factory closure backed by a scripted responder.

    For cycle flows (worker vs reviewer), routes by system-prompt keyword.
    """
    state = {"i": 0, "r": 0}

    def _factory(_profile: str | None):
        class _LLM:
            async def chat(self, messages, *, tools=None, stream=False, on_token=None, **_kw):
                system = next((m["content"] for m in messages if m.get("role") == "system"), "")
                if "review gate" in system or "AI review gate" in system:
                    state["r"] += 1
                    return cycle_reviewer(messages, tools)
                resp = responses[min(state["i"], len(responses) - 1)]
                state["i"] += 1
                if stream and on_token is not None and resp.content:
                    on_token(resp.content)
                return resp

        return _LLM()

    return _factory


def _good_fix_plan():
    return [
        tool_response("self_write", {"path": "memory/facts/web.md", "content": "# Web\n\ndone"}),
        final_response("step done"),
        tool_response("propose_cycle_complete", {}),
        final_response("complete"),
    ]


def _approved_reviewer():
    return lambda m, t: final_response("VERDICT: APPROVE\nSUMMARY: ok\nCOMMENTS: none")


def _client(settings, tmp_repo, llm_factory, governance=None) -> TestClient:
    state = AppState(
        settings=settings, llm_factory=llm_factory,
        governance=governance, repo_path=tmp_repo.repo,
    )
    app = create_app(state)
    return TestClient(app)


def _parse_sse(body: str) -> list[tuple[str, str]]:
    """Parse an SSE byte stream into a list of (event, data) pairs.

    Normalizes CRLF line endings emitted by some ASGI test clients.
    """
    events: list[tuple[str, str]] = []
    body = body.replace("\r\n", "\n")
    for part in body.split("\n\n"):
        event = "message"
        data = ""
        for line in part.split("\n"):
            if line.startswith("event:"):
                event = line[6:].strip()
            elif line.startswith("data:"):
                data += line[5:].strip()
        if event != "message" or data:
            events.append((event, data))
    return events


# --------------------------------------------------------------------------- #
# Pages
# --------------------------------------------------------------------------- #
def test_index_page_returns_chat_ui(settings, tmp_repo, monkeypatch):
    # Force the legacy Jinja2 UI: the SPA (if built under static/dist) takes
    # over `/` by default, but the legacy page must remain reachable.
    monkeypatch.setenv("NELKE_WEB_LEGACY", "1")
    app = create_app(AppState(settings=settings, repo_path=tmp_repo.repo))
    with TestClient(app) as client:
        resp = client.get("/")
    assert resp.status_code == 200
    assert "Nelke" in resp.text
    assert "chat-form" in resp.text


def test_memory_page_lists_files(settings, tmp_repo):
    from nelke.core import services

    services.open_memory(tmp_repo.repo).write("facts/x.md", "# X\n\nbody")
    app = create_app(AppState(settings=settings, repo_path=tmp_repo.repo))
    with TestClient(app) as client:
        resp = client.get("/memory")
    assert resp.status_code == 200
    assert "facts/x.md" in resp.text


def test_health_endpoint(settings, tmp_repo):
    app = create_app(AppState(settings=settings, repo_path=tmp_repo.repo))
    from nelke import __version__

    with TestClient(app) as client:
        resp = client.get("/api/health")
    assert resp.json() == {"ok": True, "version": __version__}


# --------------------------------------------------------------------------- #
# Chat (SSE)
# --------------------------------------------------------------------------- #
def test_chat_streams_tokens_and_done(settings, tmp_repo):
    resp_obj = LLMResponse(content="hello there", usage={"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5})
    factory = _scripted_llm_factory([resp_obj])
    client = _client(settings, tmp_repo, factory)
    with client:
        r = client.post("/api/chat", json={"text": "hi", "profile": None})
    assert r.status_code == 200
    events = _parse_sse(r.text)
    kinds = [e for e, _ in events]
    assert "token" in kinds
    assert "usage" in kinds  # live per-call usage is streamed to the client
    assert "done" in kinds
    usage_ev = json.loads(next(d for e, d in events if e == "usage"))
    assert usage_ev["total_tokens"] == 5
    done_data = json.loads(next(d for e, d in events if e == "done"))
    assert done_data["answer"] == "hello there"
    assert done_data["usage"]["total_tokens"] == 5


def test_chat_streams_tool_events(settings, tmp_repo):
    # Tool write then final answer: the agent loop fires on_tool/on_tool_result.
    (tmp_repo.repo / "workspace").mkdir(exist_ok=True)  # workspace dir is created per-session anyway
    factory = _scripted_llm_factory([
        tool_response("read", {"path": "README.md"}),
        final_response("done reading"),
    ])
    client = _client(settings, tmp_repo, factory)
    with client:
        r = client.post("/api/chat", json={"text": "read the readme", "profile": None})
    events = _parse_sse(r.text)
    kinds = [e for e, _ in events]
    assert "tool" in kinds
    assert "tool_result" in kinds
    assert "done" in kinds


def test_chat_empty_text_returns_error_event(settings, tmp_repo):
    factory = _scripted_llm_factory([final_response("x")])
    client = _client(settings, tmp_repo, factory)
    with client:
        r = client.post("/api/chat", json={"text": "", "profile": None})
    events = _parse_sse(r.text)
    assert any(e == "error" for e, _ in events)


def test_chat_tags_session_as_web(settings, tmp_repo):
    from nelke.core import services

    factory = _scripted_llm_factory([final_response("ok")])
    client = _client(settings, tmp_repo, factory)
    with client:
        client.post("/api/chat", json={"text": "hi", "profile": None})
    db = services.open_db(settings)
    rows = db.connect().execute("SELECT frontend FROM sessions").fetchall()
    assert any(r["frontend"] == "web" for r in rows)


# --------------------------------------------------------------------------- #
# Chats (multiple conversations, each with history + memory)
# --------------------------------------------------------------------------- #
def test_chats_create_list_detail_rename_delete(settings, tmp_repo):
    from nelke.core import services

    factory = _scripted_llm_factory([final_response("hi")])
    client = _client(settings, tmp_repo, factory)
    with client:
        # create
        r = client.post("/api/chats", json={"title": "First"})
        assert r.status_code == 200
        cid = r.json()["id"]

        # list + detail
        lst = client.get("/api/chats").json()
        assert any(c["id"] == cid for c in lst)
        det = client.get("/api/chats/" + cid).json()
        assert det["title"] == "First"
        assert det["messages"] == []

        # rename
        assert client.patch("/api/chats/" + cid, json={"title": "Renamed"}).status_code == 200
        assert client.get("/api/chats/" + cid).json()["title"] == "Renamed"
        assert client.patch("/api/chats/" + cid, json={"title": " "}).status_code == 400

        # delete
        assert client.delete("/api/chats/" + cid).status_code == 200
        assert client.get("/api/chats/" + cid).status_code == 404
        assert client.delete("/api/chats/" + cid).status_code == 404

    assert services.get_chat(settings, cid) is None


def test_chat_message_in_session_persists_history(settings, tmp_repo):
    factory = _scripted_llm_factory([final_response("hello there")])
    client = _client(settings, tmp_repo, factory)
    with client:
        cid = client.post("/api/chats", json={}).json()["id"]
        r = client.post(f"/api/chats/{cid}/messages", json={"text": "hi", "profile": None})
        assert r.status_code == 200
        events = _parse_sse(r.text)
        assert any(e == "done" for e, _ in events)
        det = client.get("/api/chats/" + cid).json()
        roles = [m["role"] for m in det["messages"]]
        assert "user" in roles
        assert "assistant" in roles
    # global memory store is shared by chats (no per-chat memory dir)
    index = tmp_repo.repo / "memory" / "INDEX.md"
    assert index.exists()
    assert not (tmp_repo.repo / "memory" / "chats").exists()


def test_chat_message_unknown_chat_returns_404(settings, tmp_repo):
    factory = _scripted_llm_factory([final_response("x")])
    client = _client(settings, tmp_repo, factory)
    with client:
        r = client.post("/api/chats/doesnotexist/messages", json={"text": "hi"})
    assert r.status_code == 404


# --------------------------------------------------------------------------- #
# Cycles history pages + api
# --------------------------------------------------------------------------- #
def test_cycles_list_and_detail(settings, tmp_repo):
    factory = _scripted_llm_factory(_good_fix_plan(), cycle_reviewer=_approved_reviewer())
    app = create_app(AppState(
        settings=settings, llm_factory=factory,
        governance=FakeGovernance(), repo_path=tmp_repo.repo,
    ))
    TestClient(app).post("/api/improve", json={"objective": "seed cycle", "auto_approve": True})

    with TestClient(app) as c:
        lst = c.get("/api/cycles/list").json()
        assert lst
        cid = lst[0]["id"]
        assert lst[0]["status"] == "merged"
        assert lst[0]["steps"]

        det = c.get("/api/cycles/" + cid)
        assert det.status_code == 200
        body = det.json()
        assert body["id"] == cid
        assert body["events"]

        page = c.get("/cycles")
        assert page.status_code == 200
        assert cid in page.text

        det_page = c.get("/cycles/" + cid)
        assert det_page.status_code == 200
        assert "Steps" in det_page.text
        assert "Timeline" in det_page.text


def test_cycle_detail_unknown_404(settings, tmp_repo):
    factory = _scripted_llm_factory([final_response("x")])
    client = _client(settings, tmp_repo, factory)
    with client:
        assert client.get("/api/cycles/nope").status_code == 404


# --------------------------------------------------------------------------- #
# Improve + review gate
# --------------------------------------------------------------------------- #
def test_improve_auto_approve_merges(settings, tmp_repo):
    factory = _scripted_llm_factory(_good_fix_plan(), cycle_reviewer=_approved_reviewer())
    client = _client(settings, tmp_repo, factory, governance=FakeGovernance())
    with client:
        r = client.post("/api/improve", json={"objective": "add a memory lesson", "auto_approve": True})
        assert r.json() == {"status": "started"}
    # the background task ran to completion within the TestClient context
    assert (tmp_repo.repo / "memory" / "facts" / "web.md").exists()
    assert tmp_repo.current_branch() == "main"


def test_cycles_api_and_stream_report_progress(settings, tmp_repo):
    """The web card reads the persisted trace via /api/cycles and /api/cycles/stream."""

    # Run a full cycle (auto-approve) so events are persisted.
    factory = _scripted_llm_factory(_good_fix_plan(), cycle_reviewer=_approved_reviewer())
    app = create_app(AppState(
        settings=settings, llm_factory=factory,
        governance=FakeGovernance(), repo_path=tmp_repo.repo,
    ))
    TestClient(app).post("/api/improve", json={"objective": "add a memory lesson", "auto_approve": True})

    with TestClient(app) as c:
        r = c.get("/api/cycles")
        assert r.status_code == 200
        cycles = r.json()
        assert cycles, "expected a cycle record"
        kinds = {ev["kind"] for ev in cycles[-1]["events"]}
        assert "agent_tool" in kinds
        assert "merged" in kinds
        # SSE stream replays the trace (replay_only makes the generator return
        # after draining the persisted events so TestClient doesn't hang).
        s = c.get("/api/cycles/stream?replay_only=1")
        assert s.status_code == 200
        body = s.text.replace("\r\n", "\n")
        assert "cycle_event" in body
        assert "self_write" in body


def _wait_for_review(settings, timeout: float = 5.0) -> list[dict[str, Any]]:
    """Poll until the cycle reaches the human gate and creates a review request."""
    import time

    from nelke.core import services

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        reviews = services.list_open_reviews(settings)
        if reviews:
            return reviews
        time.sleep(0.05)
    return services.list_open_reviews(settings)


def test_improve_human_gate_resolve_approve_merges(settings, tmp_repo):
    """Cycle parks on the human gate; /api/review/{id} with approved wakes it."""
    factory = _scripted_llm_factory(_good_fix_plan(), cycle_reviewer=_approved_reviewer())
    state = AppState(
        settings=settings, llm_factory=factory,
        governance=FakeGovernance(), repo_path=tmp_repo.repo,
    )
    app = create_app(state)
    with TestClient(app) as client:
        client.post("/api/improve", json={"objective": "add a memory lesson"})
        reviews = _wait_for_review(settings)
        assert reviews, "expected a pending review after the cycle reached the gate"
        request_id = reviews[0]["id"]
        r = client.post("/api/review/" + request_id, json={"decision": "approved"})
        assert r.status_code == 200
        assert r.json()["status"] == "resolved"
    # merged after the context closed (background task completes)
    assert (tmp_repo.repo / "memory" / "facts" / "web.md").exists()
    assert tmp_repo.current_branch() == "main"


def test_improve_human_gate_resolve_reject_keeps_branch(settings, tmp_repo):
    factory = _scripted_llm_factory(_good_fix_plan(), cycle_reviewer=_approved_reviewer())
    state = AppState(
        settings=settings, llm_factory=factory,
        governance=FakeGovernance(), repo_path=tmp_repo.repo,
    )
    app = create_app(state)
    with TestClient(app) as client:
        client.post("/api/improve", json={"objective": "add a memory lesson"})
        reviews = _wait_for_review(settings)
        request_id = reviews[0]["id"]
        client.post("/api/review/" + request_id, json={"decision": "rejected"})
    assert tmp_repo.current_branch().startswith("improve/")
    main_log = tmp_repo._run("log", "--oneline", "main", "-5").stdout
    assert "memory lesson" not in main_log


def test_review_page_shows_diff(settings, tmp_repo):
    factory = _scripted_llm_factory(_good_fix_plan(), cycle_reviewer=_approved_reviewer())
    state = AppState(
        settings=settings, llm_factory=factory,
        governance=FakeGovernance(), repo_path=tmp_repo.repo,
    )
    app = create_app(state)
    with TestClient(app) as client:
        client.post("/api/improve", json={"objective": "add a memory lesson", "auto_approve": False})
        reviews = _wait_for_review(settings)
        request_id = reviews[0]["id"]
        r = client.get("/review/" + request_id)
    assert r.status_code == 200
    assert "memory/facts/web.md" in r.text
    assert "Approve" in r.text


def test_resolve_review_bad_decision(settings, tmp_repo):
    factory = _scripted_llm_factory(_good_fix_plan(), cycle_reviewer=_approved_reviewer())
    client = _client(settings, tmp_repo, factory, governance=FakeGovernance())
    with client:
        r = client.post("/api/review/whatever", json={"decision": "maybe"})
    assert r.status_code == 400


def test_resolve_review_unknown_id_404(settings, tmp_repo):
    factory = _scripted_llm_factory([final_response("x")])
    client = _client(settings, tmp_repo, factory, governance=FakeGovernance())
    with client:
        r = client.post("/api/review/nonexistent", json={"decision": "approved"})
    assert r.status_code == 404


# --------------------------------------------------------------------------- #
# Message tree endpoints (edit / regenerate / swipe / delete / tree)
# --------------------------------------------------------------------------- #
def _seed_chat_with_two_turns(settings, tmp_repo, factory):
    """Create a chat with user→assistant→user→assistant history; return chat id."""
    from nelke.core import services

    cid = services.create_chat(settings, frontend="web")
    client = _client(settings, tmp_repo, factory)
    with client:
        client.post(f"/api/chats/{cid}/messages", json={"text": "first"})
        client.post(f"/api/chats/{cid}/messages", json={"text": "second"})
    return cid


def test_tree_endpoint_returns_nodes(settings, tmp_repo):
    factory = _scripted_llm_factory([final_response("a1"), final_response("a2")])
    cid = _seed_chat_with_two_turns(settings, tmp_repo, factory)
    client = _client(settings, tmp_repo, factory)
    with client:
        r = client.get(f"/api/chats/{cid}/tree")
    assert r.status_code == 200
    body = r.json()
    assert body["root_id"] is not None
    # 4 nodes: 2 user + 2 assistant in a linear chain.
    assert len(body["nodes"]) == 4
    assert body["active_leaf_id"] is not None


def test_edit_message_creates_sibling_and_sets_active(settings, tmp_repo):
    factory = _scripted_llm_factory([final_response("a1"), final_response("a2")])
    cid = _seed_chat_with_two_turns(settings, tmp_repo, factory)
    from nelke.core import services

    db = services.open_db(settings)
    # Find the first user message to edit.
    first_user = next(
        m for m in db.list_messages(cid, active_only=False) if m["role"] == "user"
    )
    client = _client(settings, tmp_repo, factory)
    with client:
        r = client.patch(
            f"/api/chats/{cid}/messages/{first_user['id']}",
            json={"content": "edited first"},
        )
    assert r.status_code == 200
    body = r.json()
    assert body["parent_id"] is None  # root user message
    new_id = body["message_id"]
    # New sibling is active; old subtree soft-deleted.
    new_row = db.get_message(new_id)
    assert new_row["content"] == "edited first"
    assert new_row["is_active"] == 1
    assert db.get_message(first_user["id"])["is_deleted"] == 1


def test_delete_message_soft_deletes_and_repairs_active_path(settings, tmp_repo):
    factory = _scripted_llm_factory([final_response("a1"), final_response("a2")])
    cid = _seed_chat_with_two_turns(settings, tmp_repo, factory)
    from nelke.core import services

    db = services.open_db(settings)
    messages = db.list_messages(cid, active_only=False)
    leaf = messages[-1]  # last assistant message
    client = _client(settings, tmp_repo, factory)
    with client:
        r = client.delete(f"/api/chats/{cid}/messages/{leaf['id']}")
    assert r.status_code == 200
    body = r.json()
    assert body["deleted_id"] == leaf["id"]
    # Deleted leaf tombstoned; an earlier node becomes the active leaf.
    assert db.get_message(leaf["id"])["is_deleted"] == 1
    assert body["active_leaf_id"] is not None


def test_swipe_activate_switches_active_path(settings, tmp_repo):
    """Two assistant siblings under the same user → activate switches the leaf."""
    from nelke.core import services

    cid = services.create_chat(settings, frontend="web")
    db = services.open_db(settings)
    # Seed a user message with two assistant siblings directly in the DB.
    user_id = db.add_message(cid, "user", "hi")
    first_id = db.add_message(cid, "assistant", "answer one", parent_id=user_id,
                              is_active=True, sibling_order=0)
    second_id = db.add_message(cid, "assistant", "answer two", parent_id=user_id,
                               is_active=False, sibling_order=1)
    assert db.active_leaf(cid)["id"] == first_id

    factory = _scripted_llm_factory([final_response("x")])
    client = _client(settings, tmp_repo, factory)
    with client:
        r = client.post(f"/api/chats/{cid}/messages/{second_id}/activate")
    assert r.status_code == 200
    body = r.json()
    assert body["active_leaf_id"] == second_id
    # The second answer is now on the active path; the first is inactive.
    assert db.get_message(second_id)["is_active"] == 1
    assert db.get_message(first_id)["is_active"] == 0


def test_spa_served_when_dist_built(settings, tmp_repo, monkeypatch):
    """`/` returns the SPA entry when a build is present."""
    from nelke.frontends import web as web_mod

    # Pretend a build exists by stubbing the locator.
    fake_dist = tmp_repo.repo / "static" / "dist"
    fake_dist.mkdir(parents=True)
    (fake_dist / "index.html").write_text("<!doctype html><html>SPA</html>")
    (fake_dist / "assets").mkdir()  # the SPA mount expects an assets subdir
    monkeypatch.setattr(web_mod, "_spa_dist_dir", lambda: fake_dist)
    monkeypatch.delenv("NELKE_WEB_LEGACY", raising=False)
    app = create_app(AppState(settings=settings, repo_path=tmp_repo.repo))
    with TestClient(app) as client:
        r = client.get("/")
    assert r.status_code == 200
    assert "SPA" in r.text
