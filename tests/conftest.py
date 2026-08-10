"""Shared fakes and fixtures for the Nelke seed test suite."""

from __future__ import annotations

from typing import Any, Callable

import pytest

from nelke.core.db import Database
from nelke.core.gitops import GitRepo
from nelke.core.governance import CheckResult, GateResult
from nelke.core.llm import LLMResponse, ToolCall
from nelke.core.memory import MemoryStore


# --------------------------------------------------------------------------- #
# Fake LLM
# --------------------------------------------------------------------------- #
class FakeLLM:
    """Deterministic LLM driven by a responder: ``(messages, tools) -> LLMResponse``."""

    def __init__(self, responder: Callable[[list[dict], list[dict] | None], LLMResponse] | None = None) -> None:
        self.responder = responder or (lambda messages, tools: LLMResponse(content="OK"))
        self.calls: list[dict[str, Any]] = []

    async def chat(self, messages, *, tools=None, model=None, temperature=None,
                   max_tokens=None, stream=False, on_token=None):
        self.calls.append({"messages": list(messages), "tools": tools})
        resp = self.responder(messages, tools)
        if stream and on_token is not None and resp.content:
            on_token(resp.content)
        return resp


def tool_response(name: str, arguments: dict[str, Any], tool_id: str = "call_1") -> LLMResponse:
    return LLMResponse(content="", tool_calls=[ToolCall(id=tool_id, name=name, arguments=arguments)])


def final_response(text: str = "done") -> LLMResponse:
    return LLMResponse(content=text)


def scripted(responses: list[LLMResponse]) -> Callable[[list[dict], list[dict] | None], LLMResponse]:
    state = {"i": 0}

    def _respond(messages, tools):
        i = state["i"]
        state["i"] += 1
        return responses[min(i, len(responses) - 1)]

    return _respond


def llm_with_script(responses: list[LLMResponse]) -> FakeLLM:
    return FakeLLM(responder=scripted(responses))


def driver_fake(
    worker: Callable[[list[dict], list[dict] | None], LLMResponse] | None = None,
    reviewer: Callable[[list[dict], list[dict] | None], LLMResponse] | None = None,
) -> FakeLLM:
    """Runs a distinct responder for worker vs reviewer agents (detected by system prompt)."""
    reviewer = reviewer or (lambda messages, tools: final_response(
        "VERDICT: APPROVE\nSUMMARY: looks good\nCOMMENTS: none"
    ))
    worker = worker or scripted([final_response("done")])

    def _respond(messages, tools):
        system = next((m["content"] for m in messages if m.get("role") == "system"), "")
        if "review gate" in system or "AI review gate" in system:
            return reviewer(messages, tools)
        return worker(messages, tools)

    return FakeLLM(responder=_respond)


# --------------------------------------------------------------------------- #
# Fake governance
# --------------------------------------------------------------------------- #
class FakeGovernance:
    """Programmable gate/boot-check. Defaults to always green."""

    def __init__(self) -> None:
        self.gates: list[GateResult] = []
        self.boots: list[CheckResult] = []

    async def gate(self) -> GateResult:
        return self.gates.pop(0) if self.gates else GateResult(passed=True, checks=[])

    async def boot_check(self) -> CheckResult:
        return self.boots.pop(0) if self.boots else CheckResult("boot-check", ok=True)

    async def run_lint(self) -> CheckResult:
        return CheckResult("lint", ok=True)

    async def run_typecheck(self) -> CheckResult:
        return CheckResult("typecheck", ok=True)

    async def run_tests(self) -> CheckResult:
        return CheckResult("tests", ok=True)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture
def tmp_repo(tmp_path) -> GitRepo:
    """A real git repo with identity and one seed commit on main."""
    root = tmp_path / "repo"
    root.mkdir()
    repo = GitRepo(root)
    repo.init(default_branch="main")
    repo.configure_local_identity("Nelke Test", "nelke-test@example.com")
    (root / "README.md").write_text("# Nelke test repo\n", encoding="utf-8")
    repo.add_all()
    repo.commit("initial")
    return repo


@pytest.fixture
def db(tmp_path) -> Database:
    database = Database(tmp_path / "nelke.db")
    database.migrate()
    return database


@pytest.fixture
def memory_store(tmp_path) -> MemoryStore:
    return MemoryStore(tmp_path / "memory")


@pytest.fixture
def settings(tmp_path, monkeypatch):
    monkeypatch.setenv("NELKE_NELKE_HOME", str(tmp_path / "home"))
    from nelke.config import Settings

    return Settings()


def approved_reviewer():
    """A reviewer that always approves."""
    return lambda messages, tools: final_response("VERDICT: APPROVE\nSUMMARY: ok\nCOMMENTS: none")


def simple_choice(message, finish_reason="stop"):
    from types import SimpleNamespace as NS

    return NS(choices=[NS(message=message, finish_reason=finish_reason)], usage=None)
