"""Nelke - self-improving general-purpose agent."""

from __future__ import annotations

__version__ = "0.6.0"


def boot_check() -> None:
    """Import-time smoke check used as the rollback gate after self-commits.

    Must be fast and must NOT require a network connection or a configured LLM.
    It imports every core module (surfacing SyntaxError/ImportError) and runs a
    trivial agent loop against a stub LLM.
    """
    import asyncio

    # Import all core modules so a broken import in any of them fails the check.
    import nelke.core.agent  # noqa: F401
    import nelke.core.cycle  # noqa: F401
    import nelke.core.db  # noqa: F401
    import nelke.core.gitops  # noqa: F401
    import nelke.core.governance  # noqa: F401
    import nelke.core.llm  # noqa: F401
    import nelke.core.memory  # noqa: F401
    import nelke.core.reviewer  # noqa: F401
    import nelke.core.services  # noqa: F401
    import nelke.core.session_analyzer  # noqa: F401
    import nelke.core.tools.base  # noqa: F401
    import nelke.core.tools.fs  # noqa: F401
    import nelke.core.tools.memory  # noqa: F401
    import nelke.core.tools.registry  # noqa: F401
    import nelke.core.tools.selfedit  # noqa: F401
    import nelke.core.tools.shell  # noqa: F401
    import nelke.core.tools.subagent  # noqa: F401
    import nelke.core.tools.web  # noqa: F401
    import nelke.frontends.telegram_bot  # noqa: F401
    import nelke.frontends.tui  # noqa: F401
    import nelke.frontends.web  # noqa: F401
    from nelke.core.agent import Agent
    from nelke.core.llm import StubLLM

    stub = StubLLM(answer="OK")
    agent = Agent(
        name="boot-check",
        system_prompt="Reply with the single word OK. Never call tools.",
        tools=[],
        llm=stub,
    )
    result = asyncio.run(agent.run("Say OK."))
    if "OK" not in (result.answer or ""):
        raise RuntimeError(f"boot_check smoke failed: {result.answer!r}")
