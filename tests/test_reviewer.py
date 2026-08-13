"""Reviewer verdict parsing and end-to-end review flow with a fake LLM."""

from __future__ import annotations

from conftest import FakeLLM, final_response

from nelke.core.gitops import GitRepo
from nelke.core.llm import LLMResponse
from nelke.core.reviewer import Reviewer, parse_verdict


def test_parse_approve():
    v = parse_verdict("VERDICT: APPROVE\nSUMMARY: good\nCOMMENTS: none")
    assert v.verdict == "approve"
    assert v.approved
    assert v.comments == "none"


def test_parse_request_changes():
    v = parse_verdict(
        "VERDICT: REQUEST_CHANGES\nSUMMARY: risky\nCOMMENTS:\n- add tests\n- fix import"
    )
    assert v.verdict == "request_changes"
    assert not v.approved
    assert "add tests" in v.comments


def test_parse_defaults_conservative_when_absent():
    v = parse_verdict("I have no strong opinion.")
    assert v.verdict == "request_changes"


def test_parse_approve_without_explicit_verdict():
    v = parse_verdict("Everything looks good, APPROVE the merge.")
    assert v.verdict == "approve"


async def test_reviewer_uses_approve_verdict(tmp_repo: GitRepo):
    llm = FakeLLM(responder=lambda m, t: final_response(
        "VERDICT: APPROVE\nSUMMARY: clean\nCOMMENTS: none"
    ))
    reviewer = Reviewer(tmp_repo, llm, base="main")
    # give the branch something to review
    tmp_repo.checkout_new_branch("improve/x", base="main")
    (tmp_repo.repo / "f.txt").write_text("new line\n", encoding="utf-8")
    tmp_repo.add_all()
    tmp_repo.commit("add line")
    verdict = await reviewer.review("improve f")
    assert verdict.approved
    assert verdict.summary == "clean"
    # diff was injected into the prompt
    assert "new line" in llm.calls[0]["messages"][-1]["content"]


async def test_reviewer_detects_request_changes(tmp_repo: GitRepo):
    llm = FakeLLM(responder=lambda m, t: final_response(
        "VERDICT: REQUEST_CHANGES\nSUMMARY: needs coverage\nCOMMENTS:\n- add a test"
    ))
    reviewer = Reviewer(tmp_repo, llm, base="main")
    tmp_repo.checkout_new_branch("improve/y", base="main")
    (tmp_repo.repo / "f.txt").write_text("more\n", encoding="utf-8")
    tmp_repo.add_all()
    tmp_repo.commit("more")
    verdict = await reviewer.review("improve f")
    assert not verdict.approved
    assert "test" in verdict.comments


def test_parse_empty_answer_is_request_changes_with_visible_comment():
    """An empty/tool-only reviewer answer must not silently count as approval,
    and must carry a visible comment so the cycle log explains the failure."""
    v = parse_verdict("")
    assert not v.approved
    assert v.verdict == "request_changes"
    assert v.comments  # non-empty — no silent failure
    assert "no verdict" in v.comments.lower()


async def test_review_forces_verdict_when_agent_returns_none(tmp_repo: GitRepo):
    """When the reviewer agent runs out of tool iterations and returns no answer,
    review() makes one final tool-less LLM call that must yield a verdict — so a
    review never silently fails as request_changes with empty comments."""
    from nelke.core.llm import LLMResponse, ToolCall

    call = {"n": 0}

    class _LLM:
        async def chat(self, messages, *, tools=None, stream=False, on_token=None,
                       temperature=None, **_kw):
            call["n"] += 1
            # First call (agent.run): model returns a tool call only, no answer.
            if call["n"] == 1:
                return LLMResponse(content="", tool_calls=[
                    ToolCall(id="c1", name="git_diff", arguments={})
                ])
            # Final forced call (no tools offered): model commits to a verdict.
            return LLMResponse(content="VERDICT: APPROVE\nSUMMARY: fine\nCOMMENTS:\n- none")

    reviewer = Reviewer(tmp_repo, _LLM(), iteration_cap=1)
    verdict = await reviewer.review("the objective", diff="diff --git ...")
    assert verdict.approved
    assert call["n"] >= 2  # the forced fallback fired


async def test_review_does_not_force_when_verdict_present(tmp_repo: GitRepo):
    """If the agent already returned a VERDICT line, review() does NOT make a
    redundant forced call."""
    call = {"n": 0}

    class _LLM:
        async def chat(self, messages, **_kw):
            call["n"] += 1
            return LLMResponse(
                content="VERDICT: REQUEST_CHANGES\nSUMMARY: x\nCOMMENTS:\n- bug")

    reviewer = Reviewer(tmp_repo, _LLM())
    verdict = await reviewer.review("obj", diff="diff")
    assert not verdict.approved
    assert "bug" in verdict.comments
    assert call["n"] == 1  # no forced fallback
