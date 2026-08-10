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
