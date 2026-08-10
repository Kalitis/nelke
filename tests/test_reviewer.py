"""Reviewer verdict parsing and end-to-end review flow with a fake LLM."""

from __future__ import annotations

from conftest import FakeLLM, final_response

from nelke.core.gitops import GitRepo
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
