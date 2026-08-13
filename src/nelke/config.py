"""Nelke configuration: pydantic-settings + provider profiles.

Settings come from the environment / `.env` files under the ``NELKE_`` prefix.
Provider profiles live in ``~/.nelke/config.toml`` (see ``config.example.toml``).
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_HOME = Path.home()
_ENV_FILES = (".env", str(_HOME / ".nelke" / ".env"))


def load_env_files() -> None:
    """Load ``.env`` files into :data:`os.environ`.

    ``pydantic-settings`` reads ``.env`` only for its own ``NELKE_``-prefixed
    fields, so secrets referenced by profiles via ``api_key_ref`` (e.g.
    ``OPENAI_API_KEY``) never reach :data:`os.environ` and
    :meth:`Profile.resolved_api_key` returns ``None``. Calling this early at
    startup mirrors a shell ``source .env`` so non-prefixed variables are
    visible to the whole process. ``override=False`` keeps real environment
    variables authoritative over file values.
    """
    for env_path in _ENV_FILES:
        p = Path(env_path)
        if p.exists():
            load_dotenv(p, override=False)


def default_nelke_home() -> Path:
    return Path.home() / ".nelke"


class Settings(BaseSettings):
    """Runtime settings loaded from env / .env (``NELKE_`` prefix)."""

    model_config = SettingsConfigDict(
        env_prefix="NELKE_",
        env_file=_ENV_FILES,
        extra="ignore",
    )

    nelke_home: Path = Field(default_factory=default_nelke_home)
    default_profile: str = "openai"
    max_agent_iterations: int = 20
    max_cycle_steps: int = 30
    max_review_rounds: int = 3
    max_step_attempts: int = 3
    # Per-worker cap on read-only tool calls (self_read/self_glob/self_grep/
    # recall/memory_show/memory_list) in a single round. A worker that exceeds
    # it is stopped mid-run and re-prompted to make edits instead of looping on
    # exploration. 0 disables the cap (legacy behaviour).
    explore_budget: int = 6
    # How many times a cycle sends the agent back to fix governance-gate
    # failures (tests/lint/typecheck/test-gap) before giving up. A single agent
    # slip should bounce back for rework, not kill the whole cycle.
    max_gate_attempts: int = 5
    # When True, the governance gate rejects any change that touches src code
    # without a matching tests/test_<module>.py, forcing the agent to write
    # tests for new code it introduces.
    require_code_tests: bool = True
    code_timeout: int = 120
    web_timeout: int = 30
    recall_top_k: int = 8
    index_max_tokens: int = 2000
    # Use real (dense, multilingual) embeddings from an OpenAI-compatible
    # endpoint (`[embeddings]` in ~/.nelke/config.toml, e.g. LM Studio) for
    # memory recall/auto-link. When off — or when the endpoint is unreachable /
    # has no embedding model loaded — the local offline hashing embedder is used.
    embeddings_enabled: bool = True
    # Plan-first mode: sketch an explicit plan before the tool loop on every
    # turn. Saves iterations/tool errors on multi-step tasks, at the cost of
    # one extra (cheap, non-tool) LLM call per turn. Off by default.
    plan_first: bool = False

    # Sampling temperature for agent/tool/subagent calls. Defaults to 1 (the
    # OpenAI-compatible default) for natural responses. Prompt caching is NOT
    # gated by temperature on the main dslab profile, so a non-zero default is
    # safe — verified: identical long prefixes hit the cache at T=1.0 and T=0.0
    # alike. Lower it per-profile only if you want more deterministic sampling.
    agent_temperature: float = 1.0

    @field_validator("agent_temperature", mode="before")
    @classmethod
    def _clamp_temperature(cls, value: object) -> object:
        """Keep temperature in [0, 2] so invalid values can't disable caching or break sampling."""
        try:
            return max(0.0, min(2.0, float(str(value))))
        except (TypeError, ValueError):
            return 0.0

    @field_validator("nelke_home", mode="before")
    @classmethod
    def _expand_tilde(cls, value: object) -> object:
        if isinstance(value, str):
            return os.path.expanduser(value)
        return value

    @property
    def db_path(self) -> Path:
        return self.nelke_home / "nelke.db"

    @property
    def config_file(self) -> Path:
        return self.nelke_home / "config.toml"

    @property
    def workspaces_dir(self) -> Path:
        return self.nelke_home / "workspaces"


@dataclass
class Profile:
    """A named LLM provider profile (OpenAI-compatible endpoint)."""

    name: str
    base_url: str
    model: str
    api_key: str | None = None
    api_key_ref: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def resolved_api_key(self, env: dict[str, str] | None = None) -> str | None:
        env = dict(os.environ) if env is None else env
        if self.api_key:
            return self.api_key
        if self.api_key_ref:
            return env.get(self.api_key_ref)
        return None


def load_profiles(path: Path | None = None) -> dict[str, Profile]:
    """Load provider profiles from a ``config.toml`` (e.g. ``~/.nelke/config.toml``)."""
    path = path or default_nelke_home() / "config.toml"
    if not path.exists():
        return {}
    with path.open("rb") as fh:
        data = tomllib.load(fh)
    profiles: dict[str, Profile] = {}
    for name, cfg in (data.get("profiles") or {}).items():
        if not isinstance(cfg, dict):
            continue
        profiles[name] = Profile(
            name=str(name),
            base_url=str(cfg.get("base_url", "")),
            model=str(cfg.get("model", "")),
            api_key=cfg.get("api_key"),
            api_key_ref=cfg.get("api_key_ref"),
            extra=cfg.get("extra") or {},
        )
    return profiles


def get_profile(
    name: str | None = None, profiles: dict[str, Profile] | None = None
) -> Profile:
    """Return the active profile or raise a helpful error."""
    settings = Settings()
    wanted = name or settings.default_profile
    profiles = profiles if profiles is not None else load_profiles()
    if not profiles:
        raise ProfileError(
            "No provider profiles found. Run `nelke config init` to create "
            f"{settings.config_file}, then add an OpenAI-compatible provider."
        )
    if wanted not in profiles:
        raise ProfileError(
            f"Profile {wanted!r} not found. Available: {', '.join(sorted(profiles))}"
        )
    return profiles[wanted]


class ProfileError(RuntimeError):
    pass
