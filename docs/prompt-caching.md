# Prompt Caching — Current State and Path Forward

Honest summary: **prompt caching is not currently active at the wire level**, despite the codebase having the right intent wired up at the LangChain layer. This document captures what works, what doesn't, and what would actually unlock the cost savings.

## What we tried

We added `prepare_tools_for_caching` (`src/deepresearch/llm.py`) and wired it into `agent_node` (`src/deepresearch/graph/nodes.py:40`). The helper converts the BM25 top-K tools into Anthropic tool dicts and attaches `cache_control: {"type": "ephemeral"}` to the last one — a cache breakpoint covering the system prompt + tool defs.

Unit tests in `tests/test_llm.py` verify the helper produces the right shape and that the cache_control survives `bind_tools(...).kwargs["tools"]`.

## What actually goes out the door

Running `scripts/verify_tool_cache.py` against haiku-4-5 with two back-to-back identical calls:

```
=== Call 1 (cold cache — should populate) ===
{'input_tokens': 4488, 'cache_creation_input_tokens': 0, 'cache_read_input_tokens': 0, 'output_tokens': 42}

=== Call 2 (warm cache — should hit) ===
{'input_tokens': 4488, 'cache_creation_input_tokens': 0, 'cache_read_input_tokens': 0, 'output_tokens': 48}
```

`cache_creation_input_tokens` and `cache_read_input_tokens` are both **0** on both calls. No cache is being created and none is being read.

## Why not — two bugs surfaced

### Bug 1: System prompt cache_control is dropped at serialization

`build_system_message` produces:
```python
SystemMessage(content=[
    {"type": "text", "text": ..., "cache_control": {"type": "ephemeral"}}
])
```

Captured HTTP body (via httpx interception):
```
system: plain string (7748 chars)
```

`langchain-anthropic` flattens the content-block list back to a single string for `SystemMessage` before sending. The cache_control field is gone.

This means the system-prompt caching that the codebase has claimed for some time has **never actually worked at the wire level**. Tests that asserted the SystemMessage shape passed, but the shape doesn't survive the request-build path.

### Bug 2: Tool cache_control is dropped at serialization

Our `prepare_tools_for_caching` produces tool dicts with `cache_control` on the last entry. `bind_tools(cacheable)` stores them in `bound.kwargs["tools"]` with cache_control intact. But the actual HTTP request body shows:

```
TOOLS in request body: 13
  [0] coingecko_price       cache_control=(missing)
  ...
  [12] calculate              cache_control=(missing)
```

All cache_control fields stripped. The langchain-anthropic chat model's `_get_request_payload` re-serializes tools and drops unknown keys.

(Side observation: there are 13 tools in the body, not 8, because `ToolRetriever.search` returns `top-K + ALWAYS_INCLUDE` deduplicated. That's by design.)

### Diagnostic: direct Anthropic SDK call works fine

```
big_text = "..." * 500  # ~3500 tokens, above Haiku's 2048-token minimum
client.messages.create(
    system=[{"type": "text", "text": big_text, "cache_control": {"type": "ephemeral"}}],
    ...
)
# Call 1: input=8  cache_creation=4502  cache_read=0
# Call 2: input=8  cache_creation=0     cache_read=4502   ← cache hit
```

Anthropic's API supports caching; langchain-anthropic 1.4.2's serialization layer is what's eating the fields.

## Per-model minimums (additional gotcha)

Even with cache_control flowing correctly to the wire, caching only activates above per-model thresholds:

| Model               | Minimum cached tokens |
|---------------------|-----------------------|
| Opus / Sonnet       | 1024                  |
| Haiku               | 2048                  |

Our system prompt is ~1500 tokens — below Haiku's minimum. So even a fixed system-prompt-only cache wouldn't fire on haiku. The tool-def breakpoint matters because (system + tools) lands ~3500 tokens, comfortably over the 2048 threshold.

## Path forward — three options

### Option A: AnthropicPromptCachingMiddleware (official, biggest refactor)

`langchain-anthropic` ships `AnthropicPromptCachingMiddleware` in `langchain_anthropic/middleware/prompt_caching.py`. It does exactly what we want — tags system, tools, and last cacheable block, plus sets a top-level `cache_control` kwarg on model_settings.

**Cost:** requires migrating our agent from hand-rolled langgraph nodes to `langchain.agents.create_agent()`. That's a significant restructure — `agent_node`, `tool_node`, `should_continue`, custom state, render-tool conventions, the system-prompt requirement — all of it would need to fit the agents framework or be re-engineered around it.

### Option B: Custom request-payload override (smaller refactor)

Subclass `ChatAnthropic` and override `_get_request_payload` to inject cache_control into the outgoing body before it's serialized. Targeted, no framework migration, but fragile — any langchain-anthropic version bump could rename or restructure that method.

**Cost:** ~half-day to write + test, ongoing maintenance risk.

### Option C: Bypass LangChain for the Anthropic call

Build the request body directly with the `anthropic` SDK. Loses LangChain features like streaming integration with LangGraph's `stream_mode="messages"` and tracing. Too invasive given how much we lean on LangChain elsewhere.

### Option D: Upstream PR to langchain-anthropic

File an issue + PR adding cache_control passthrough for dict-typed tool entries and content-block SystemMessages in `_get_request_payload`. Right long-term fix, but slow.

## Cost math when this actually works

Assuming we unblock both system + tool caching:

| Provider | Without caching | With caching (S+T) | Savings |
|----------|-----------------|---------------------|---------|
| Opus     | $0.27 / query   | ~$0.13 / query      | ~50%    |
| Sonnet   | $0.054 / query  | ~$0.027 / query     | ~50%    |
| Haiku    | $0.018 / query  | ~$0.009 / query     | ~50%    |

(System prompt + tool defs ~ 3500 tokens. Cached portion served at 0.1× input rate. Remaining ~1000 user/history tokens charged normally.)

## What this PR delivers

- `prepare_tools_for_caching` helper + tests — produces the **correct shape** even though it doesn't reach the wire yet. Forward-compatible: when option B or D lands, caching auto-activates.
- `scripts/verify_tool_cache.py` — the diagnostic that surfaced both bugs. Run after any langchain-anthropic upgrade or wire-format fix to confirm caching activated.
- This document — so the next person to look at "why aren't we caching" doesn't have to re-derive the findings.

What it does NOT deliver: actual cached requests at the wire level. That requires option A or B above.

## Recommended next action

If saving ~50% on Anthropic input cost matters: pick **Option B** (subclass + override `_get_request_payload`). Half-day of work, scoped change. Verify with `verify_tool_cache.py`.

If it's not worth the maintenance risk right now: ship this PR as-is (honest sketch + diagnostic), file an upstream issue at `langchain-ai/langchain-anthropic`, wait for the passthrough fix.
