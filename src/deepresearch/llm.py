"""LLM source registry, factory, and provider-aware system message shaping.

The agent and the eval judge both pick an LLM through this module, so swapping
between Anthropic Opus, local Gemma via Ollama, OpenAI gpt-4o, or Google Gemini
is a one-name decision rather than a fork in the call site.

Public surface:

- `LLMSource` — frozen dataclass: `(name, provider, model, supports_prompt_cache, notes)`.
- `SOURCES` — registry keyed by short name.
- `resolve(role, override=None)` — pick a source for "agent" or "judge".
- `build_chat(source)` — return the right LangChain chat model, lazy-imported.
- `build_system_message(text, source)` — provider-aware shaping. Anthropic with
  caching enabled gets a content block carrying `cache_control: ephemeral`;
  everyone else gets a plain string.

Design notes:

- Provider packages (`langchain-openai`, `langchain-google-genai`,
  `langchain-ollama`) are optional extras. Each `_build_*` lazy-imports its
  package and raises a clear install hint if missing.
- Caching is provider-specific by design. Anthropic needs the explicit
  `cache_control` block; OpenAI auto-caches identical prefixes ≥1024 tokens;
  Google's Gemini 2.5+ caches implicitly; Ollama's KV cache lives in the
  model server (governed by `OLLAMA_KEEP_ALIVE`). We don't try to fake a
  unified hit-count metric — each provider's stats land in its own metadata.
"""
from __future__ import annotations

import os
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage


@dataclass(frozen=True)
class LLMSource:
    """A named LLM configuration. Add new sources by appending to `SOURCES`."""

    name: str
    provider: str  # "anthropic" | "openai" | "google" | "ollama"
    model: str
    supports_prompt_cache: bool = False
    notes: str = ""


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

SOURCES: dict[str, LLMSource] = {
    # Anthropic — cloud, prompt-cache eligible
    "opus": LLMSource(
        name="opus", provider="anthropic", model="claude-opus-4-7",
        supports_prompt_cache=True,
        notes="frontier; best tool calling and reasoning",
    ),
    "sonnet": LLMSource(
        name="sonnet", provider="anthropic", model="claude-sonnet-4-6",
        supports_prompt_cache=True,
        notes="cheaper iteration; near-Opus quality",
    ),
    "haiku": LLMSource(
        name="haiku", provider="anthropic", model="claude-haiku-4-5-20251001",
        notes="fast/cheap; default judge",
    ),

    # OpenAI — cloud, automatic prefix caching server-side
    "gpt-4o": LLMSource(
        name="gpt-4o", provider="openai", model="gpt-4o",
        notes="OpenAI flagship multimodal",
    ),
    "gpt-4o-mini": LLMSource(
        name="gpt-4o-mini", provider="openai", model="gpt-4o-mini",
        notes="OpenAI cheap/fast",
    ),

    # Google — cloud (cloud-hosted Gemma / Gemini)
    "gemini-2-flash": LLMSource(
        name="gemini-2-flash", provider="google", model="gemini-2.0-flash",
        notes="Google Gemini 2 flash; implicit cache on 2.5+ tier",
    ),

    # Ollama — local
    "gemma4-e4b": LLMSource(
        name="gemma4-e4b", provider="ollama", model="gemma4:e4b",
        notes="local; 4B effective; tool-call quality lower than Opus",
    ),
    "gemma4-e2b": LLMSource(
        name="gemma4-e2b", provider="ollama", model="gemma4:e2b",
        notes="local; 2B effective; smaller/faster than e4b",
    ),
    "qwen-7b": LLMSource(
        name="qwen-7b", provider="ollama", model="qwen2.5:7b",
        notes="local; reliable tool calling on local hardware",
    ),
}


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------

_DEFAULTS = {"agent": "opus", "judge": "haiku"}
_ENV_VARS = {"agent": "AGENT_LLM", "judge": "JUDGE_LLM"}


def get_source(name: str) -> LLMSource:
    """Look up a source by name. Raises ValueError if unknown."""
    if name not in SOURCES:
        known = ", ".join(sorted(SOURCES))
        raise ValueError(f"unknown LLM source {name!r}. Known: {known}")
    return SOURCES[name]


def resolve(role: str, *, override: str | None = None) -> LLMSource:
    """Pick the LLM source for `role` ("agent" or "judge").

    Resolution order:
      1. explicit `override` (e.g. CLI flag)
      2. role-specific env var (`AGENT_LLM`, `JUDGE_LLM`)
      3. legacy env vars (`LLM_PROVIDER` + `ANTHROPIC_MODEL`/`OLLAMA_MODEL`/`JUDGE_MODEL`)
      4. role default (agent → opus, judge → haiku)
    """
    if role not in _DEFAULTS:
        raise ValueError(f"unknown role {role!r}; expected 'agent' or 'judge'")
    if override:
        return get_source(override)
    if env := os.getenv(_ENV_VARS[role]):
        return get_source(env)
    if legacy := _legacy_from_env(role):
        return legacy
    return get_source(_DEFAULTS[role])


def _legacy_from_env(role: str) -> LLMSource | None:
    """Honor pre-registry env vars for one release, with a deprecation warning.

    Map:
      role=agent + LLM_PROVIDER=ollama + OLLAMA_MODEL=...     → derived ollama source
      role=agent + LLM_PROVIDER=anthropic + ANTHROPIC_MODEL=… → derived anthropic source
      role=judge + JUDGE_MODEL=…                              → derived anthropic source
    """
    if role == "judge":
        if model := os.getenv("JUDGE_MODEL"):
            warnings.warn(
                "JUDGE_MODEL is deprecated; set JUDGE_LLM=<source-name> instead "
                "(e.g. JUDGE_LLM=haiku).",
                DeprecationWarning,
                stacklevel=3,
            )
            return _matching_source("anthropic", model) or LLMSource(
                name=f"legacy-judge:{model}", provider="anthropic", model=model,
            )
        return None

    provider = os.getenv("LLM_PROVIDER")
    if not provider:
        return None
    provider = provider.lower()
    if provider == "ollama":
        model = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
        warnings.warn(
            "LLM_PROVIDER + OLLAMA_MODEL are deprecated; set AGENT_LLM=<source-name> "
            "(e.g. AGENT_LLM=gemma4-e4b).",
            DeprecationWarning,
            stacklevel=3,
        )
        return _matching_source("ollama", model) or LLMSource(
            name=f"legacy-ollama:{model}", provider="ollama", model=model,
        )
    if provider == "anthropic":
        model = os.getenv("ANTHROPIC_MODEL", "claude-opus-4-7")
        warnings.warn(
            "LLM_PROVIDER + ANTHROPIC_MODEL are deprecated; set AGENT_LLM=<source-name> "
            "(e.g. AGENT_LLM=opus).",
            DeprecationWarning,
            stacklevel=3,
        )
        return _matching_source("anthropic", model) or LLMSource(
            name=f"legacy-anthropic:{model}", provider="anthropic", model=model,
            supports_prompt_cache=True,
        )
    return None


def _matching_source(provider: str, model: str) -> LLMSource | None:
    for src in SOURCES.values():
        if src.provider == provider and src.model == model:
            return src
    return None


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def _build_anthropic(source: LLMSource, *, max_tokens: int, streaming: bool) -> BaseChatModel:
    try:
        from langchain_anthropic import ChatAnthropic
    except ImportError as e:
        raise ImportError(
            "Anthropic provider requires `langchain-anthropic` (already a default dep). "
            "Try `uv sync`."
        ) from e
    kwargs: dict[str, Any] = {
        "model": source.model,
        "streaming": streaming,
        "max_tokens": max_tokens,
    }
    if source.supports_prompt_cache:
        kwargs["default_headers"] = {"anthropic-beta": "prompt-caching-2024-07-31"}
    return ChatAnthropic(**kwargs)


def _build_openai(source: LLMSource, *, max_tokens: int, streaming: bool) -> BaseChatModel:
    try:
        from langchain_openai import ChatOpenAI
    except ImportError as e:
        raise ImportError(
            "OpenAI provider requires the `openai` extra. Install with "
            "`uv sync --extra openai` (or `--extra all-llms`)."
        ) from e
    # langchain-openai 1.x dropped `max_tokens` from the constructor signature.
    # Pass it through `model_kwargs` (typed Dict[str, Any]) so it reaches the
    # request body without tripping mypy on the call-arg.
    return ChatOpenAI(
        model=source.model,
        streaming=streaming,
        model_kwargs={"max_tokens": max_tokens},
    )


def _build_google(source: LLMSource, *, max_tokens: int, streaming: bool) -> BaseChatModel:
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
    except ImportError as e:
        raise ImportError(
            "Google provider requires the `google` extra. Install with "
            "`uv sync --extra google` (or `--extra all-llms`)."
        ) from e
    # ChatGoogleGenerativeAI uses max_output_tokens, not max_tokens
    return ChatGoogleGenerativeAI(model=source.model, max_output_tokens=max_tokens)


def _build_ollama(source: LLMSource, *, max_tokens: int, streaming: bool) -> BaseChatModel:
    try:
        from langchain_ollama import ChatOllama
    except ImportError as e:
        raise ImportError(
            "Ollama provider requires the `ollama` extra. Install with "
            "`uv sync --extra ollama` (or `--extra all-llms`)."
        ) from e
    return ChatOllama(
        model=source.model,
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        temperature=0,
    )


_BUILDERS: dict[str, Callable[..., BaseChatModel]] = {
    "anthropic": _build_anthropic,
    "openai": _build_openai,
    "google": _build_google,
    "ollama": _build_ollama,
}


def build_chat(source: LLMSource, *, max_tokens: int = 4096, streaming: bool = True) -> BaseChatModel:
    """Construct the right LangChain chat model for `source`."""
    fn = _BUILDERS.get(source.provider)
    if fn is None:
        raise ValueError(
            f"unknown provider {source.provider!r}. Add a `_build_{source.provider}` "
            "function or fix the registry entry."
        )
    return fn(source, max_tokens=max_tokens, streaming=streaming)


# ---------------------------------------------------------------------------
# System-message shaping
# ---------------------------------------------------------------------------

def build_system_message(text: str, source: LLMSource) -> SystemMessage:
    """Wrap the rendered system prompt for the source's expected message shape.

    Anthropic + caching: content block carrying `cache_control: ephemeral`.
    Everything else: plain string. See module docstring for caching nuances.
    """
    if source.provider == "anthropic" and source.supports_prompt_cache:
        return SystemMessage(content=[
            {
                "type": "text",
                "text": text,
                "cache_control": {"type": "ephemeral"},
            }
        ])
    return SystemMessage(content=text)


__all__ = [
    "LLMSource",
    "SOURCES",
    "build_chat",
    "build_system_message",
    "get_source",
    "resolve",
]
