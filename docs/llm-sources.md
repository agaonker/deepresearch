# LLM sources

The agent and the eval judge each pick a named LLM source from a registry in [`src/deepresearch/llm.py`](../src/deepresearch/llm.py). Adding a new model is one line in the registry, not a new branch in the call sites.

## Registered sources

```bash
uv run research --list-llms       # print the table below at any time
```

| Source | Provider | Model | Cache | Notes |
|---|---|---|---|---|
| `opus` | anthropic | `claude-opus-4-7` | yes | frontier; best tool calling and reasoning |
| `sonnet` | anthropic | `claude-sonnet-4-6` | yes | cheaper iteration; near-Opus quality |
| `haiku` | anthropic | `claude-haiku-4-5-20251001` | — | fast/cheap; default judge |
| `gpt-4o` | openai | `gpt-4o` | auto* | OpenAI flagship multimodal |
| `gpt-4o-mini` | openai | `gpt-4o-mini` | auto* | OpenAI cheap/fast |
| `gemini-2-flash` | google | `gemini-2.0-flash` | auto* | Google Gemini 2 flash |
| `gemma4-e4b` | ollama | `gemma4:e4b` | local | local; tool-call quality lower than Opus |
| `gemma4-e2b` | ollama | `gemma4:e2b` | local | local; smaller/faster than e4b |
| `qwen-7b` | ollama | `qwen2.5:7b` | local | reliable tool calling on local hardware |

\* auto = provider does prefix caching server-side without code action; see "Caching" below.

## Picking a source

```bash
uv run research --llm gemma4-e4b "What is 2+2?"   # CLI flag
AGENT_LLM=sonnet uv run research "your query"     # env var
```

Defaults: agent → `opus`, judge → `haiku`. The judge has its own selector — `JUDGE_LLM=sonnet uv run python scripts/run_experiment.py --limit 5` runs the agent on Opus and the eval judge on Sonnet.

> Want to compare providers head-to-head on the same dataset? See [**testing-providers.md**](testing-providers.md) for the full runbook — commands, costs, and reference scores for Anthropic Claude, OpenAI ChatGPT, and local Gemma via Ollama.

## Provider extras

Anthropic and the core deps are installed by default. Other providers ship as optional extras so you only install what you need:

```bash
uv sync --extra ollama       # adds langchain-ollama
uv sync --extra openai       # adds langchain-openai
uv sync --extra google       # adds langchain-google-genai
uv sync --extra all-llms     # all three
```

If you pick a source whose extra isn't installed, you'll get a clear `ImportError` with the install command.

## Caching across providers

Each provider handles caching differently — we don't try to fake a unified API:

| Provider | Mechanism | What this codebase does |
|---|---|---|
| Anthropic | Explicit `cache_control: ephemeral` block (5-min TTL) | Wired automatically when `supports_prompt_cache=True` (Opus, Sonnet) |
| OpenAI | Auto-caches identical prefixes ≥1024 tokens server-side | Nothing — savings appear in `usage.prompt_tokens_details.cached_tokens` |
| Google | Implicit caching on Gemini 2.5+; explicit `cachedContents` API exists | Plain string for v1; relies on implicit |
| Ollama | Transformer KV cache in the model server's memory | Plain string; set `OLLAMA_KEEP_ALIVE=24h` to keep models warm between sessions |

## Adding a new source

1. Append a one-line entry to `SOURCES` in [`src/deepresearch/llm.py`](../src/deepresearch/llm.py).
2. If it's a new provider, add an `_build_<provider>` function (lazy-import the LangChain package) and register it in `_BUILDERS`.
3. If the provider needs an optional extra, add it under `[project.optional-dependencies]` in `pyproject.toml`.

## Local Ollama setup

Requires `ollama serve` running and the model pulled:

```bash
ollama pull gemma4:e4b
OLLAMA_KEEP_ALIVE=24h uv run research --llm gemma4-e4b "your query"
```

Tool-calling quality on local models is lower than Opus — fine for offline iteration, weaker for demos. The 12-iteration cap (`MAX_ITERATIONS`) protects you from runaway loops.

## Legacy env vars

`LLM_PROVIDER`, `ANTHROPIC_MODEL`, `OLLAMA_MODEL`, and `JUDGE_MODEL` still work but emit a `DeprecationWarning`. Migrate to the new `AGENT_LLM` / `JUDGE_LLM` names.
