"""Prompt-cache tests for the main ``dslab`` profile.

Unit coverage for cache-metric normalization lives in ``test_llm.py``; this
file checks the real behaviour the ``agent_temperature=0`` change was meant to
unlock: second and later requests with an *unchanged long prefix* should be
served from the provider's prompt cache instead of re-billing the whole prompt.

These tests contact the real dslab endpoint and are skipped unless
``NELKE_TEST_REAL=1`` is set (same convention as ``test_integration.py``).
Run locally with::

    $env:NELKE_TEST_REAL = "1"
    uv run pytest tests/test_cache_hit.py -q

If the provider does not report any cache metrics, the tests fail loudly — that
is the point: it tells us whether prompt caching is actually engaging on dslab.
"""

from __future__ import annotations

import os
from typing import Any

import pytest

from nelke.config import load_env_files, load_profiles
from nelke.core.llm import build_llm

PROFILE = "dslab"

load_env_files()

_PREFIX_SEED = (
    "cache diagnostics filler text with the quick brown fox jumping over the "
    "lazy dog every single time the sun rises in the east and sets in the west."
)
_PREFIX = "You are a terse assistant. " + (_PREFIX_SEED + " ") * 200


def _dslab_ready() -> bool:
    profile = load_profiles().get(PROFILE)
    if profile is None:
        return False
    return bool(profile.resolved_api_key())


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("NELKE_TEST_REAL") != "1",
        reason="set NELKE_TEST_REAL=1 to run real-model cache tests",
    ),
    pytest.mark.skipif(
        not _dslab_ready(),
        reason=f"profile {PROFILE!r} is not configured with an api key",
    ),
]


@pytest.fixture(scope="module")
def llm():
    return build_llm(PROFILE)


async def _chat(llm, tale: str, *, stream: bool = False) -> dict[str, Any]:
    resp = await llm.chat(
        [
            {"role": "system", "content": _PREFIX},
            {"role": "user", "content": "Reply with exactly " + tale},
        ],
        stream=stream,
        temperature=0.0,
    )
    assert resp.usage is not None, "dslab must report usage (usage = None)"
    return dict(resp.usage)


async def test_dslab_reports_cache_metric_fields(llm):
    """The provider must expose cache fields at all (read/write), not just totals."""
    usage = await _chat(llm, "one")
    assert "cache_read_tokens" in usage
    assert "cache_write_tokens" in usage
    assert usage["prompt_tokens"] > 0


async def test_dslab_second_call_served_from_prompt_cache(llm):
    """A repeated identical prefix must be served from cache on non-streaming calls."""
    first = await _chat(llm, "two")
    second = await _chat(llm, "three")
    third = await _chat(llm, "four")
    # The first call may only write the cache; the follow-ups must read it.
    assert first["cache_read_tokens"] >= 0
    for label, usage in (("second", second), ("third", third)):
        assert usage["cache_read_tokens"] > 0, (
            f"no cache read on {label} identical-prefix call (temperature=0): {usage}"
        )
        assert usage["cache_write_tokens"] < usage["prompt_tokens"], (
            f"{label} call re-billed the full prompt (cache not engaged): {usage}"
        )


async def test_dslab_streaming_call_served_from_prompt_cache(llm):
    """The agent-loop path (streaming) must also hit the cache on repeats."""
    await _chat(llm, "five", stream=True)
    usage = await _chat(llm, "six", stream=True)
    assert usage["cache_read_tokens"] > 0, (
        f"no cache read on streaming identical-prefix call: {usage}"
    )
