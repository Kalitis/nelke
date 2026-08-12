"""Tests for the shared frontend service layer (core/services.py).

These verify the wiring every frontend relies on: chat session setup + db
tagging, one-shot task run with usage persistence, cycle delegation, review
resolution (the "first frontend wins" contract) and the memory read helpers.
All use the fakes from conftest, no real LLM.
"""

from __future__ import annotations

import pytest
from conftest import FakeGovernance, final_response, tool_response

from nelke.core import services
from nelke.core.services import Callbacks


# --------------------------------------------------------------------------- #
# build_chat_session / run_task
# --------------------------------------------------------------------------- #
def _llm_factory(responses):
    """Build an llm_factory closure backed by a scripted FakeLLM."""
    state = {"i": 0}

    def _factory(_profile):
        class _LLM:
            async def chat(self, messages, *, tools=None, model=None, temperature=None,
                           max_tokens=None, stream=False, on_token=None):
                resp = responses[min(state["i"], len(responses) - 1)]
                state["i"] += 1
                if stream and on_token is not None and resp.content:
                    on_token(resp.content)
                return resp

        return _LLM()

    return _factory


async def test_build_chat_session_tags_db_and_wires_callbacks(tmp_repo, settings):
    factory = _llm_factory([final_response("hi")])
    session = services.build_chat_session(
        settings, profile=None, frontend_name="web",
        callbacks=Callbacks(stream=True), repo=tmp_repo.repo, llm_factory=factory,
    )
    assert session.session_id
    assert session.db.get_cycle(session.session_id) is None  # it's a session, not a cycle
    row = session.db.connect().execute(
        "SELECT frontend FROM sessions WHERE id=?", (session.session_id,)
    ).fetchone()
    assert row["frontend"] == "web"
    assert session.agent is not None


async def test_run_task_persists_usage_and_ends_session(tmp_repo, settings):
    from nelke.core.llm import LLMResponse

    resp = LLMResponse(content="42", usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15})
    factory = _llm_factory([resp])
    result, session_id = await services.run_task(
        "what is the answer?", settings, profile=None,
        frontend_name="tui", repo=tmp_repo.repo, llm_factory=factory,
    )
    assert result.answer == "42"
    totals = services.open_db(settings).usage_totals(session_id=session_id)
    assert totals["total_tokens"] == 15
    assert totals["calls"] == 1


async def test_run_task_persists_usage_per_call_in_real_time(tmp_repo, settings):
    """Usage is written to the DB per LLM call (real-time), not one aggregate."""
    from nelke.core.llm import LLMResponse, ToolCall

    responses = [
        LLMResponse(content="", tool_calls=[ToolCall("call_1", "read", {"path": "x.txt"})],
                    usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}),
        LLMResponse(content="done", usage={"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}),
    ]
    factory = _llm_factory(responses)
    result, session_id = await services.run_task(
        "go", settings, profile=None, frontend_name="web",
        repo=tmp_repo.repo, llm_factory=factory,
    )
    assert result.answer == "done"
    db = services.open_db(settings)
    events = db.list_usage(session_id=session_id)
    assert len(events) == 2  # one usage_events row per LLM call
    totals = db.usage_totals(session_id=session_id)
    assert totals["total_tokens"] == 7
    assert totals["calls"] == 2


async def test_run_task_streaming_callbacks_fire(tmp_repo, settings):
    tokens: list[str] = []
    factory = _llm_factory([final_response("hello")])
    result, _ = await services.run_task(
        "say hello", settings, profile=None, frontend_name="web",
        callbacks=Callbacks(on_token=tokens.append, stream=True),
        repo=tmp_repo.repo, llm_factory=factory,
    )
    assert result.answer == "hello"
    assert "".join(tokens) == "hello"


async def test_run_task_reset_false_continues_conversation(tmp_repo, settings):
    """A second run_task call is a fresh session (stateless helper); continuity
    requires holding the ChatSession — verify that path instead."""
    factory = _llm_factory([final_response("first"), final_response("second")])
    session = services.build_chat_session(
        settings, profile=None, frontend_name="web", repo=tmp_repo.repo, llm_factory=factory,
    )
    first = await session.agent.run("hi", reset=True)
    second = await session.agent.run("again", reset=False)
    assert first.answer == "first"
    assert second.answer == "second"
    # continuity: the second turn saw the first turn's messages
    roles = [m["role"] for m in session.agent._messages]
    assert roles.count("user") == 2


# --------------------------------------------------------------------------- #
# run_cycle
# --------------------------------------------------------------------------- #
def _good_fix_plan():
    return [
        tool_response("self_write", {"path": "memory/facts/work.md", "content": "# Work\n\ndone"}),
        final_response("step done"),
        tool_response("propose_cycle_complete", {}),
        final_response("complete"),
    ]


def _cycle_llm_factory(worker_responses, reviewer=None):
    reviewer = reviewer or (lambda m, t: final_response("VERDICT: APPROVE\nSUMMARY: ok\nCOMMENTS: none"))
    state = {"i": 0, "r": 0}

    def _factory(_profile):
        class _LLM:
            async def chat(self, messages, *, tools=None, **_kw):
                system = next((m["content"] for m in messages if m.get("role") == "system"), "")
                if "review gate" in system or "AI review gate" in system:
                    state["r"] += 1
                    return reviewer(messages, tools)
                i = state["i"]
                state["i"] += 1
                return worker_responses[min(i, len(worker_responses) - 1)]

        return _LLM()

    return _factory


async def test_run_cycle_delegates_to_engine_and_human_gate(tmp_repo, settings):
    events: list[str] = []
    gate_calls: list[bool] = []

    async def human(_req) -> bool:
        gate_calls.append(True)
        return True

    factory = _cycle_llm_factory(_good_fix_plan())
    result = await services.run_cycle(
        "add a memory lesson", settings, profile=None,
        on_event=lambda e: events.append(e.kind), human_approve=human,
        repo_path=tmp_repo.repo, llm_factory=factory, governance=FakeGovernance(),
    )
    assert result.merged
    assert gate_calls == [True]
    assert "merged" in events
    assert (tmp_repo.repo / "memory" / "facts" / "work.md").exists()


async def test_run_cycle_reject_keeps_branch(tmp_repo, settings):
    async def human(_req) -> bool:
        return False

    factory = _cycle_llm_factory(_good_fix_plan())
    result = await services.run_cycle(
        "add a memory lesson", settings, profile=None,
        human_approve=human, repo_path=tmp_repo.repo, llm_factory=factory, governance=FakeGovernance(),
    )
    assert result.status == "rejected"
    assert tmp_repo.current_branch().startswith("improve/")


async def test_run_cycle_requires_git_repo(tmp_path, settings):
    factory = _cycle_llm_factory([])
    not_a_repo = tmp_path / "nope"
    not_a_repo.mkdir()
    with pytest.raises(RuntimeError, match="not a git repository"):
        await services.run_cycle("x", settings, profile=None, repo_path=not_a_repo, llm_factory=factory)


# --------------------------------------------------------------------------- #
# resolve_review
# --------------------------------------------------------------------------- #
async def _seed_pending_review(tmp_repo, settings) -> str:
    """Run a quick cycle that reaches the human gate, return its review request id.

    Uses the SAME database services writes to (settings.db_path), since the
    ``db`` fixture opens a different temp file.
    """
    factory = _cycle_llm_factory(_good_fix_plan())
    # no human_approve -> cycle stops at "awaiting-human" with an open request
    await services.run_cycle(
        "add a memory lesson", settings, profile=None,
        repo_path=tmp_repo.repo, llm_factory=factory, governance=FakeGovernance(),
    )
    db = services.open_db(settings)
    open_reqs = db.list_review_requests(open_only=True)
    human_reqs = [r for r in open_reqs if r["kind"] == "human"]
    assert human_reqs, "expected a pending human review request"
    return human_reqs[0]["id"]


async def test_resolve_review_approved_merges(tmp_repo, settings):
    request_id = await _seed_pending_review(tmp_repo, settings)
    outcome = services.resolve_review(settings, request_id, "approved", repo_path=tmp_repo.repo)
    assert outcome.status == "merged"
    assert outcome.human_verdict == "approved"
    db = services.open_db(settings)
    cycle = db.get_cycle(outcome.cycle_id)
    assert cycle["status"] == "merged"
    # request resolved in db
    remaining = [r for r in db.list_review_requests(open_only=True) if r["id"] == request_id]
    assert remaining == []
    # merged onto main
    assert (tmp_repo.repo / "memory" / "facts" / "work.md").exists()
    assert tmp_repo.current_branch() == "main"


async def test_resolve_review_rejected_keeps_branch(tmp_repo, settings):
    request_id = await _seed_pending_review(tmp_repo, settings)
    outcome = services.resolve_review(settings, request_id, "rejected", repo_path=tmp_repo.repo)
    assert outcome.status == "rejected"
    db = services.open_db(settings)
    cycle = db.get_cycle(outcome.cycle_id)
    assert cycle["status"] == "rejected"


def test_resolve_review_unknown_id_raises(settings, tmp_path):
    with pytest.raises(RuntimeError, match="review request not found"):
        services.resolve_review(settings, "does-not-exist", "approved", repo_path=tmp_path)


async def test_resolve_review_idempotent_first_wins(tmp_repo, settings):
    """Resolving already-resolved request does not flip the cycle back."""
    request_id = await _seed_pending_review(tmp_repo, settings)
    services.resolve_review(settings, request_id, "approved", repo_path=tmp_repo.repo)
    db = services.open_db(settings)
    cycle_id = db.list_review_requests(open_only=False)[0]["cycle_id"]
    # the merged branch is gone from an improve/ state
    assert tmp_repo.current_branch() == "main"
    assert db.get_cycle(cycle_id)["status"] == "merged"


# --------------------------------------------------------------------------- #
# list_open_reviews / get_review / memory helpers
# --------------------------------------------------------------------------- #
async def test_list_open_reviews_empty(settings, tmp_path, monkeypatch):
    monkeypatch.setenv("NELKE_REPO", str(tmp_path))
    assert services.list_open_reviews(settings) == []


async def test_list_open_reviews_returns_human_only(tmp_repo, settings, monkeypatch):
    monkeypatch.setenv("NELKE_REPO", str(tmp_repo.repo))
    await _seed_pending_review(tmp_repo, settings)
    reviews = services.list_open_reviews(settings)
    assert len(reviews) == 1
    assert reviews[0]["kind"] == "human"
    assert reviews[0]["branch"].startswith("improve/")
    assert "memory lesson" in reviews[0]["objective"]


async def test_get_review_returns_diff(tmp_repo, settings, monkeypatch):
    monkeypatch.setenv("NELKE_REPO", str(tmp_repo.repo))
    request_id = await _seed_pending_review(tmp_repo, settings)
    review = services.get_review(settings, request_id)
    assert review is not None
    assert "memory/facts/work.md" in review["diff"]


def test_get_review_unknown_returns_none(settings, tmp_path):
    assert services.get_review(settings, "nope") is None


async def test_memory_overview_and_recall(tmp_repo, settings):
    store = services.open_memory(tmp_repo.repo)
    store.write("facts/alpha.md", "# Alpha\n\ntags: math\n\nThe answer is 42.\n")
    store.write("notes/beta.md", "# Beta\n\nUnrelated text here.\n")
    overview = services.memory_overview(tmp_repo.repo)
    names = {m["name"] for m in overview}
    assert "facts/alpha.md" in names
    assert "notes/beta.md" in names
    hits = services.recall_memory(tmp_repo.repo, "answer math", top_k=5)
    assert hits, "expected recall hits"
    assert any("alpha" in h.name for h in hits)


# --------------------------------------------------------------------------- #
# find_repo / open_db
# --------------------------------------------------------------------------- #
def test_find_repo_respects_env_override(tmp_path, monkeypatch):
    monkeypatch.setenv("NELKE_REPO", str(tmp_path))
    assert services.find_repo() == tmp_path


def test_open_db_migrates(settings):
    db = services.open_db(settings)
    counts = db.status()
    assert "sessions" in counts


# --------------------------------------------------------------------------- #
# Chats: multiple conversations with history (shared global memory)
# --------------------------------------------------------------------------- #
def test_chat_crud_and_titles(settings, tmp_repo):
    cid = services.create_chat(settings, title="First")
    chats = services.list_chats(settings, frontend="web")
    assert any(c["id"] == cid and c["title"] == "First" for c in chats)

    assert services.rename_chat(settings, cid, "Renamed")
    assert services.get_chat(settings, cid, repo=tmp_repo.repo)["title"] == "Renamed"

    chat = services.get_chat(settings, cid, repo=tmp_repo.repo)
    assert chat["messages"] == []
    assert chat["memory"] == []

    assert services.delete_chat(settings, cid)
    assert services.get_chat(settings, cid) is None
    assert services.rename_chat(settings, cid, "x") is False
    assert services.delete_chat(settings, cid) is False


def test_list_chats_orders_by_activity_and_derives_title(settings):
    a = services.create_chat(settings, frontend="web")
    b = services.create_chat(settings, frontend="web")
    # title derives from the first user message when no meta title is set
    db = services.open_db(settings)
    db.add_message(a, "user", "First user question about math")
    chats = services.list_chats(settings, frontend="web")
    assert chats[0]["id"] == a  # newer activity
    assert any(c["id"] == b for c in chats)
    assert any(c["id"] == a and "First user question" in c["title"] for c in chats)
    assert any(c["id"] == a and c["message_count"] == 1 for c in chats)


def test_list_chats_default_lists_every_frontend(settings, tmp_repo):
    web = services.create_chat(settings, title="Web chat", frontend="web")
    tui = services.create_chat(settings, title="TUI chat", frontend="tui")
    tg = services.create_chat(settings, title="TG chat", frontend="telegram")
    # default view = all chats, shared across frontends (web/TUI/Telegram)
    all_chats = services.list_chats(settings)
    ids = {c["id"] for c in all_chats}
    assert {web, tui, tg} <= ids
    by_id = {c["id"]: c for c in all_chats}
    assert by_id[web]["frontend"] == "web"
    assert by_id[tui]["frontend"] == "tui"
    assert by_id[tg]["frontend"] == "telegram"
    # per-frontend filter is still available for backwards compatibility
    web_only = services.list_chats(settings, frontend="web")
    assert web_only and all(c["frontend"] == "web" for c in web_only)
    assert web in {c["id"] for c in web_only}
    assert tg not in {c["id"] for c in web_only}


def _chat_llm_factory(responses):
    """llm_factory that records every messages list it sees, for continuity checks."""
    seen: list[list[str]] = []
    state = {"i": 0}

    def _factory(_profile):
        class _LLM:
            async def chat(self, messages, *, tools=None, stream=False, on_token=None, **_kw):
                seen.append([m.get("role") for m in messages])
                resp = responses[min(state["i"], len(responses) - 1)]
                state["i"] += 1
                if stream and on_token is not None and resp.content:
                    on_token(resp.content)
                return resp

        return _LLM()

    return _factory, seen


async def test_run_chat_turn_persists_transcript(settings, tmp_repo):
    factory, _ = _chat_llm_factory([final_response("first")])
    cid = services.create_chat(settings, frontend="tui")
    result, chat_id, _msg_id = await services.run_chat_turn(
        "first msg", settings, None, cid,
        frontend_name="tui", repo=tmp_repo.repo, llm_factory=factory,
    )
    assert chat_id == cid
    assert result.answer == "first"
    messages = services.get_chat_messages(settings, cid)
    assert [m["role"] for m in messages] == ["user", "assistant"]
    assert messages[0]["content"] == "first msg"
    assert messages[1]["content"] == "first"


async def test_run_chat_turn_reloads_history_across_calls(settings, tmp_repo):
    factory, seen = _chat_llm_factory([final_response("first"), final_response("second")])
    cid = services.create_chat(settings, frontend="tui")
    await services.run_chat_turn("first msg", settings, None, cid,
                                 frontend_name="tui", repo=tmp_repo.repo, llm_factory=factory)
    await services.run_chat_turn("second msg", settings, None, cid,
                                 frontend_name="tui", repo=tmp_repo.repo, llm_factory=factory)
    assert len(seen) == 2
    assert seen[1].count("user") >= 2  # the reloaded history was passed to the LLM


async def test_run_chat_turn_uses_global_memory(settings, tmp_repo):
    factory, _ = _chat_llm_factory([final_response("ok")])
    cid = services.create_chat(settings, frontend="web")
    await services.run_chat_turn("hi", settings, None, cid,
                                 frontend_name="web", repo=tmp_repo.repo, llm_factory=factory)
    # The chat uses the shared global store (repo/memory), not per-chat memory.
    index = tmp_repo.repo / "memory" / "INDEX.md"
    assert index.exists()
    assert not (tmp_repo.repo / "memory" / "chats").exists()


# --------------------------------------------------------------------------- #
# Cycle history browser
# --------------------------------------------------------------------------- #
async def test_list_cycles_and_detail(tmp_repo, settings):
    async def human(_req) -> bool:
        return True

    factory = _cycle_llm_factory(_good_fix_plan())
    await services.run_cycle(
        "add a memory lesson", settings, None,
        human_approve=human, repo_path=tmp_repo.repo, llm_factory=factory,
        governance=FakeGovernance(),
    )
    cycles = services.list_cycles(settings)
    assert cycles
    first = cycles[0]
    assert first["status"] == "merged"
    assert first["steps"]  # persisted step trace
    detail = services.get_cycle_detail(settings, first["id"])
    assert detail is not None
    assert detail["id"] == first["id"]
    assert detail["events"]  # timeline present
    assert detail["reviews"]
    # Single-worker cycles populate no parallel worker rows.
    assert detail["workers"] == []


async def test_parallel_cycle_detail_exposes_workers(tmp_repo, settings):
    """Parallel cycles expose one worker row per planner slice in the detail DTO."""
    import json as _json

    async def human(_req) -> bool:
        return True

    worker_responses = [
        tool_response("self_write", {"path": "memory/facts/a.md", "content": "# a\ndone"}),
        final_response("done"),
    ]
    planner_payload = _json.dumps({"tasks": [{"title": "all", "detail": "do everything"}]})

    def factory(_profile):
        class _LLM:
            async def chat(self, messages, *, tools=None, **_kw):
                system = next((m["content"] for m in messages if m.get("role") == "system"), "")
                if "task planner" in system:
                    return final_response(planner_payload)
                if "review gate" in system or "AI review gate" in system:
                    return final_response("VERDICT: APPROVE\nSUMMARY: ok\nCOMMENTS: none")
                # Workers are stateless here; reuse the same scripted plan for each.
                return worker_responses[0]

        return _LLM()

    await services.run_cycle(
        "do two things", settings, None,
        human_approve=human, repo_path=tmp_repo.repo, llm_factory=factory,
        governance=FakeGovernance(), mode="parallel",
    )
    cycles = services.list_cycles(settings)
    first = cycles[0]
    detail = services.get_cycle_detail(settings, first["id"])
    assert detail is not None
    # At least one worker row was persisted; the planner returned a single task.
    assert len(detail["workers"]) >= 1
    w = detail["workers"][0]
    assert {"id", "worker_index", "title", "detail", "status"} <= set(w.keys())


def test_reconcile_stale_cycles_marks_empty_branches(tmp_repo, settings):
    """running cycles with no commits on their branch are stuck/failed."""
    db = services.open_db(settings)
    db.create_cycle("ghost branch", "improve/ghost", cycle_id="c-ghost")
    tmp_repo.checkout_new_branch("improve/empty", base="main")
    db.create_cycle("no commits", "improve/empty", cycle_id="c-empty")
    tmp_repo.checkout_new_branch("improve/work", base="main")
    (tmp_repo.repo / "work.txt").write_text("x", encoding="utf-8")
    tmp_repo.add_all()
    tmp_repo.commit("work")
    db.create_cycle("has commits", "improve/work", cycle_id="c-work")

    marked = services.reconcile_stale_cycles(settings, repo_path=tmp_repo.repo)
    assert {m["id"] for m in marked} == {"c-ghost", "c-empty"}
    assert db.get_cycle("c-ghost")["status"] == "stuck"
    assert db.get_cycle("c-empty")["status"] == "stuck"
    # a branch with commits (here also the currently checked-out branch) is kept
    assert db.get_cycle("c-work")["status"] == "running"


# --------------------------------------------------------------------------- #
# Projects: CRUD + per-project memory
# --------------------------------------------------------------------------- #
def test_create_project_seeds_memory_and_returns_stage(tmp_repo, settings):
    pid = services.create_project(
        settings, name="Nelke", description="agent", stage="idea", repo=tmp_repo.repo,
    )
    # memory dir seeded with an INDEX.md
    store = services.open_project_memory(tmp_repo.repo, pid)
    assert "INDEX.md" in {str(p) for p in store.files()}

    detail = services.get_project(settings, pid, repo=tmp_repo.repo)
    assert detail is not None
    assert detail["stage"] == "idea"
    assert detail["chat_count"] == 0
    # detail carries memory files (at least the seeded INDEX)
    assert any(f["name"] == "INDEX.md" for f in detail["memory_files"])


def test_set_project_memory_writes_note_and_detail_lists_it(tmp_repo, settings):
    pid = services.create_project(settings, name="P", repo=tmp_repo.repo)
    ok = services.set_project_memory(
        settings, pid, "notes.md", "first note", append=False, repo=tmp_repo.repo,
    )
    assert ok
    detail = services.get_project(settings, pid, repo=tmp_repo.repo)
    names = {f["name"] for f in detail["memory_files"]}
    assert "notes.md" in names
    # content round-trips through the MemoryStore
    store = services.open_project_memory(tmp_repo.repo, pid)
    assert "first note" in store.read("notes.md")


def test_set_project_memory_rejects_bad_names_and_unknown_project(tmp_repo, settings):
    pid = services.create_project(settings, name="P", repo=tmp_repo.repo)
    with pytest.raises(ValueError):
        services.set_project_memory(settings, pid, "sub/dir.md", "x", repo=tmp_repo.repo)
    with pytest.raises(ValueError):
        services.set_project_memory(settings, pid, "notes.txt", "x", repo=tmp_repo.repo)
    # unknown project → False, not an exception
    assert services.set_project_memory(settings, "ghost", "notes.md", "x", repo=tmp_repo.repo) is False


def test_attach_chat_to_project_and_list(tmp_repo, settings):
    pid = services.create_project(settings, name="P", repo=tmp_repo.repo)
    sid = services.create_chat(settings, frontend="web")
    assert services.attach_chat_to_project(settings, chat_id=sid, project_id=pid)
    detail = services.get_project(settings, pid, repo=tmp_repo.repo)
    assert len(detail["chats"]) == 1
    assert detail["chats"][0]["id"] == sid

    # detach
    assert services.attach_chat_to_project(settings, chat_id=sid, project_id=None)
    assert len(services.get_project(settings, pid, repo=tmp_repo.repo)["chats"]) == 0


def test_update_project_stage_and_delete(tmp_repo, settings):
    pid = services.create_project(settings, name="P", stage="idea", repo=tmp_repo.repo)
    assert services.update_project(settings, pid, stage="active")
    assert services.get_project(settings, pid, repo=tmp_repo.repo)["stage"] == "active"
    assert services.delete_project(settings, pid)
    assert services.get_project(settings, pid, repo=tmp_repo.repo) is None


# --------------------------------------------------------------------------- #
# Kanban boards / columns / cards (service layer)
# --------------------------------------------------------------------------- #
def test_kanban_board_crud_and_default_columns(tmp_repo, settings):
    pid = services.create_project(settings, name="Kanban proj", repo=tmp_repo.repo)
    board_id = services.create_kanban_board(settings, pid, "Board", "desc")
    board = services.get_kanban_board(settings, board_id)
    assert board is not None
    assert board["project_id"] == pid
    assert board["name"] == "Board"
    assert board["description"] == "desc"
    # a fresh board ships default columns
    col_names = [c["name"] for c in board["columns"]]
    assert "Backlog" in col_names and "Done" in col_names

    # list_kanban_boards includes the board
    boards = services.list_kanban_boards(settings, pid)
    assert [b["id"] for b in boards] == [board_id]

    # delete
    assert services.delete_kanban_board(settings, board_id)
    assert services.get_kanban_board(settings, board_id) is None


def test_kanban_board_requires_project_and_name(tmp_repo, settings):
    with pytest.raises(ValueError, match="board name is required"):
        services.create_kanban_board(settings, "ghost", "")
    with pytest.raises(ValueError, match="project not found"):
        services.create_kanban_board(settings, "ghost", "Board")


def test_kanban_add_move_and_reorder_cards(tmp_repo, settings):
    pid = services.create_project(settings, name="P", repo=tmp_repo.repo)
    board_id = services.create_kanban_board(settings, pid, "Board")
    board = services.get_kanban_board(settings, board_id)
    assert board is not None
    col_a = board["columns"][0]["id"]
    col_b = board["columns"][1]["id"]

    c1 = services.add_kanban_card(settings, board_id, col_a, "First", "a desc")
    c2 = services.add_kanban_card(settings, board_id, col_a, "Second")

    board = services.get_kanban_board(settings, board_id)
    col_a_cards = [c for c in board["columns"][0]["cards"]]
    assert sorted([c["title"] for c in col_a_cards]) == ["First", "Second"]

    # move a card to another column
    assert services.move_kanban_card(settings, c1, col_b)
    board = services.get_kanban_board(settings, board_id)
    moved = next(c for c in board["columns"][1]["cards"] if c["id"] == c1)
    assert moved["title"] == "First"

    # update a card's title/description
    assert services.update_kanban_card(settings, c2, "Second!", "new desc")
    board = services.get_kanban_board(settings, board_id)
    updated = next(c for c in board["columns"][0]["cards"] if c["id"] == c2)
    assert updated["title"] == "Second!"
    assert updated["description"] == "new desc"

    # delete a card
    assert services.delete_kanban_card(settings, c2)
    board = services.get_kanban_board(settings, board_id)
    assert all(c["id"] != c2 for col in board["columns"] for c in col["cards"])


def test_kanban_add_card_invalid_inputs(tmp_repo, settings):
    pid = services.create_project(settings, name="P", repo=tmp_repo.repo)
    board_id = services.create_kanban_board(settings, pid, "Board")
    col_id = services.get_kanban_board(settings, board_id)["columns"][0]["id"]

    # empty title
    with pytest.raises(ValueError, match="card title is required"):
        services.add_kanban_card(settings, board_id, col_id, "   ")
    # unknown board
    with pytest.raises(ValueError, match="board not found"):
        services.add_kanban_card(settings, "ghost-board", col_id, "X")
    # column not on the board
    with pytest.raises(ValueError, match="column not found"):
        services.add_kanban_card(settings, board_id, "ghost-column", "X")


def test_kanban_move_card_rejects_foreign_column(tmp_repo, settings):
    pid = services.create_project(settings, name="P", repo=tmp_repo.repo)
    b1 = services.create_kanban_board(settings, pid, "B1")
    b2 = services.create_kanban_board(settings, pid, "B2")
    col_id_b1 = services.get_kanban_board(settings, b1)["columns"][0]["id"]
    col_id_b2 = services.get_kanban_board(settings, b2)["columns"][0]["id"]
    card = services.add_kanban_card(settings, b1, col_id_b1, "X")
    # moving to a column of another board is rejected
    assert services.move_kanban_card(settings, card, col_id_b2) is False
    # moving to a valid column works
    assert services.move_kanban_card(settings, card, col_id_b1)
    # unknown card is rejected
    assert services.move_kanban_card(settings, "ghost-card", col_id_b1) is False


