"""Tests for the LLM source registry, factory, and message shaping."""
from __future__ import annotations

import warnings

import pytest

from deepresearch.llm import (
    SOURCES,
    LLMSource,
    build_chat,
    build_system_message,
    get_source,
    resolve,
)

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def test_known_sources_registered():
    expected = {
        "opus", "sonnet", "haiku",
        "gpt-4o", "gpt-4o-mini",
        "gemini-2-flash",
        "gemma4-e4b", "gemma4-e2b", "qwen-7b",
    }
    missing = expected - SOURCES.keys()
    assert not missing, f"missing: {missing}"


def test_anthropic_sources_marked_for_cache():
    for name in ("opus", "sonnet"):
        assert SOURCES[name].supports_prompt_cache, f"{name} should support prompt cache"
    # Haiku is not flagged — judge use case keeps it cheap and Anthropic's
    # cost saving on small judge requests is negligible.
    assert SOURCES["haiku"].supports_prompt_cache is False


def test_get_source_unknown_raises():
    with pytest.raises(ValueError, match="unknown LLM source"):
        get_source("not-a-real-source")


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------

def test_resolve_uses_override():
    assert resolve("agent", override="haiku").name == "haiku"


def test_resolve_uses_env_for_agent(monkeypatch):
    monkeypatch.setenv("AGENT_LLM", "gemma4-e4b")
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    assert resolve("agent").name == "gemma4-e4b"


def test_resolve_uses_env_for_judge(monkeypatch):
    monkeypatch.setenv("JUDGE_LLM", "sonnet")
    monkeypatch.delenv("JUDGE_MODEL", raising=False)
    assert resolve("judge").name == "sonnet"


def test_resolve_default_agent(monkeypatch):
    monkeypatch.delenv("AGENT_LLM", raising=False)
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("ANTHROPIC_MODEL", raising=False)
    monkeypatch.delenv("OLLAMA_MODEL", raising=False)
    assert resolve("agent").name == "opus"


def test_resolve_default_judge(monkeypatch):
    monkeypatch.delenv("JUDGE_LLM", raising=False)
    monkeypatch.delenv("JUDGE_MODEL", raising=False)
    assert resolve("judge").name == "haiku"


def test_resolve_invalid_role_raises():
    with pytest.raises(ValueError, match="unknown role"):
        resolve("supervisor")


def test_resolve_legacy_ollama_env(monkeypatch):
    monkeypatch.delenv("AGENT_LLM", raising=False)
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("OLLAMA_MODEL", "gemma4:e4b")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        src = resolve("agent")
    assert src.provider == "ollama"
    assert src.model == "gemma4:e4b"
    # Should match the registered gemma4-e4b entry, not a derived one
    assert src.name == "gemma4-e4b"
    assert any(issubclass(w.category, DeprecationWarning) for w in caught)


def test_resolve_legacy_anthropic_env(monkeypatch):
    monkeypatch.delenv("AGENT_LLM", raising=False)
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_MODEL", "claude-opus-4-7")
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        src = resolve("agent")
    assert src.name == "opus"


def test_resolve_legacy_judge_model(monkeypatch):
    monkeypatch.delenv("JUDGE_LLM", raising=False)
    monkeypatch.setenv("JUDGE_MODEL", "claude-haiku-4-5-20251001")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        src = resolve("judge")
    assert src.name == "haiku"
    assert any(issubclass(w.category, DeprecationWarning) for w in caught)


# ---------------------------------------------------------------------------
# System-message shaping
# ---------------------------------------------------------------------------

def test_system_message_anthropic_with_cache_uses_blocks():
    src = SOURCES["opus"]
    msg = build_system_message("hello world", src)
    assert isinstance(msg.content, list)
    assert msg.content[0]["text"] == "hello world"
    assert msg.content[0]["cache_control"] == {"type": "ephemeral"}


def test_system_message_anthropic_uncached_is_plain_string():
    src = SOURCES["haiku"]
    msg = build_system_message("hello world", src)
    assert msg.content == "hello world"


def test_system_message_ollama_is_plain_string():
    src = SOURCES["gemma4-e4b"]
    msg = build_system_message("hello world", src)
    assert msg.content == "hello world"


def test_system_message_openai_is_plain_string():
    src = SOURCES["gpt-4o"]
    msg = build_system_message("hello world", src)
    assert msg.content == "hello world"


# ---------------------------------------------------------------------------
# Factory dispatch (lazy-import safety)
# ---------------------------------------------------------------------------

def test_build_chat_unknown_provider_raises():
    bogus = LLMSource(name="x", provider="not-a-provider", model="x")
    with pytest.raises(ValueError, match="unknown provider"):
        build_chat(bogus)


def test_build_chat_anthropic_returns_chat_anthropic():
    # langchain-anthropic is a default dep, so this should work without extras.
    src = SOURCES["haiku"]  # uncached → smaller config to construct
    chat = build_chat(src, max_tokens=10, streaming=False)
    # langchain wraps; check we got something with the right model attribute
    assert getattr(chat, "model", None) == src.model or getattr(chat, "model_name", None) == src.model


def test_build_chat_ollama_returns_chat_ollama():
    src = SOURCES["gemma4-e4b"]
    chat = build_chat(src, max_tokens=10, streaming=False)
    assert getattr(chat, "model", None) == src.model
