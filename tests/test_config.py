"""Config tests: env loading, provider profiles from TOML."""

from __future__ import annotations

from pathlib import Path

import pytest

from nelke.config import ProfileError, Settings, get_profile, load_profiles


def test_settings_env_prefix(monkeypatch, tmp_path):
    monkeypatch.setenv("NELKE_NELKE_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("NELKE_DEFAULT_PROFILE", "lmstudio")
    monkeypatch.setenv("NELKE_MAX_AGENT_ITERATIONS", "7")
    s = Settings()
    assert s.nelke_home == tmp_path / "home"
    assert s.default_profile == "lmstudio"
    assert s.max_agent_iterations == 7


def test_settings_defaults(monkeypatch):
    monkeypatch.delenv("NELKE_NELKE_HOME", raising=False)
    s = Settings()
    assert s.nelke_home == Path.home() / ".nelke"
    assert s.max_cycle_steps == 30
    # agent temperature defaults to 0 so provider prompt caching engages
    assert s.agent_temperature == 0.0


def test_agent_temperature_env_and_clamp(monkeypatch, tmp_path):
    monkeypatch.setenv("NELKE_NELKE_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("NELKE_AGENT_TEMPERATURE", "1.2")
    assert Settings().agent_temperature == 1.2
    # out-of-range / invalid values are clamped to a safe range (never >2,
    # because a temperature>0 is what disables prompt caching on some providers)
    monkeypatch.setenv("NELKE_AGENT_TEMPERATURE", "9")
    assert Settings().agent_temperature == 2.0
    monkeypatch.setenv("NELKE_AGENT_TEMPERATURE", "not-a-number")
    assert Settings().agent_temperature == 0.0


def test_load_profiles(tmp_path):
    cfg = tmp_path / "config.toml"
    cfg.write_text(
        """
[profiles.openai]
base_url = "https://api.openai.com/v1"
model = "gpt-4o-mini"
api_key_ref = "OPENAI_API_KEY"

[profiles.ollama]
base_url = "http://localhost:11434/v1"
model = "qwen3:8b"
api_key = "ollama"
""",
        encoding="utf-8",
    )
    profiles = load_profiles(cfg)
    assert set(profiles) == {"openai", "ollama"}
    assert profiles["openai"].resolved_api_key({"OPENAI_API_KEY": "sk-1"}) == "sk-1"
    assert profiles["ollama"].resolved_api_key({}) == "ollama"


def test_get_profile_missing_raises(tmp_path):
    with pytest.raises(ProfileError):
        get_profile("nope", profiles={})


def test_get_profile_unknown_name():
    from nelke.config import Profile

    profiles = {"other": Profile(name="other", base_url="x", model="m")}
    with pytest.raises(ProfileError):
        get_profile("openai", profiles=profiles)
