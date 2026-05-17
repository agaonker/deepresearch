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
    prepare_tools_for_caching,
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


# ---------------------------------------------------------------------------
# prepare_tools_for_caching
# ---------------------------------------------------------------------------

def _sample_tools():
    """A small subset of real catalog tools (real BaseTool objects)."""
    from deepresearch.tools.catalog import ALL_DATA_TOOLS
    from deepresearch.tools.render import ALL_RENDER_TOOLS

    return ALL_DATA_TOOLS[:2] + ALL_RENDER_TOOLS[:1]


def test_prepare_tools_caching_anthropic_marks_last_tool():
    tools = _sample_tools()
    out = prepare_tools_for_caching(tools, SOURCES["opus"])
    assert len(out) == len(tools)
    # All entries are dicts with Anthropic tool shape.
    for entry in out:
        assert isinstance(entry, dict)
        assert "name" in entry and "input_schema" in entry
    # Only the LAST tool carries cache_control.
    assert "cache_control" not in out[0]
    assert "cache_control" not in out[1]
    assert out[-1]["cache_control"] == {"type": "ephemeral"}


def test_prepare_tools_caching_anthropic_preserves_order():
    tools = _sample_tools()
    out = prepare_tools_for_caching(tools, SOURCES["sonnet"])
    expected_names = [t.name for t in tools]
    got_names = [entry["name"] for entry in out]
    assert got_names == expected_names


def test_prepare_tools_caching_ollama_passthrough():
    tools = _sample_tools()
    out = prepare_tools_for_caching(tools, SOURCES["gemma4-e4b"])
    # Ollama provider: returned unchanged (still BaseTool objects, not dicts).
    assert list(out) == list(tools)
    for entry in out:
        assert hasattr(entry, "name")  # BaseTool attr, not dict["name"]


def test_prepare_tools_caching_openai_passthrough():
    tools = _sample_tools()
    out = prepare_tools_for_caching(tools, SOURCES["gpt-4o"])
    # OpenAI uses server-side prefix caching; no client-side cache_control needed.
    assert list(out) == list(tools)


def test_prepare_tools_caching_empty_list_returned_unchanged():
    out = prepare_tools_for_caching([], SOURCES["opus"])
    assert list(out) == []


def test_prepare_tools_caching_anthropic_no_cache_provider_passes_through():
    # Build an Anthropic source variant with caching disabled.
    src = LLMSource(
        name="anthropic-nocache",
        provider="anthropic",
        model="claude-haiku-4-5-20251001",
        supports_prompt_cache=False,
    )
    tools = _sample_tools()
    out = prepare_tools_for_caching(tools, src)
    # Even on Anthropic, caching off => no transformation.
    assert list(out) == list(tools)


def test_prepare_tools_caching_survives_bind_tools():
    """End-to-end: the cache_control breakpoint must reach ChatAnthropic.bind_tools
    and stay attached to the final tool in the bound runnable's kwargs.

    Guards against the failure mode where LangChain re-formats dicts and drops
    keys it doesn't recognize."""
    tools = _sample_tools()
    cacheable = prepare_tools_for_caching(tools, SOURCES["opus"])
    llm = build_chat(SOURCES["haiku"], max_tokens=10, streaming=False)
    bound = llm.bind_tools(cacheable)
    bound_tools = getattr(bound, "kwargs", {}).get("tools", [])
    assert len(bound_tools) == len(tools)
    # First N-1 tools have no cache_control; last one does.
    for entry in bound_tools[:-1]:
        assert "cache_control" not in entry
    assert bound_tools[-1].get("cache_control") == {"type": "ephemeral"}
