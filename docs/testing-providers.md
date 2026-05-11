# Testing across LLM providers

This guide shows how to put the same agent through the same dataset on three providers — Anthropic Claude, OpenAI ChatGPT, and local Gemma via Ollama — and compare the results in LangSmith.

## Findings (TL;DR)

A 5-example smoke comparison run on 2026-05-10, agent code unchanged across runs (same PR #5 prompts + caching wiring, same BM25 retrieval, same dataset).

### Headline numbers

| Scorer | Opus + Haiku judge | Gemma 4 e4b + Haiku judge | Gemma 4 e4b + Qwen 7B judge |
|---|---|---|---|
| `tool_recall` | **1.00** | 0.80 | 1.00 |
| `render_match` | **0.80** | 0.20 | 0.40 |
| `iterations_used` | 0.76 | 0.69 | **0.84** |
| `answer_correctness` | 0.68 | 0.44 | 0.68 |
| Latency / example | ~10–30 s | ~110 s | ~110 s |
| LLM cost / 5 examples | ~$0.30 | ~$0.03 (judge only) | **$0** |

### What worked

- **Tool selection survives even at 4B params.** BM25 narrows 42 tools to 8 before binding; Gemma picks correctly from that short menu most of the time.
- **The new abstraction is provider-neutral.** Anthropic, OpenAI, Google, and Ollama all flow through the same `build_chat` / `build_system_message` path with no extra branching at call sites.
- **Going fully local works.** `AGENT_LLM=gemma4-e4b JUDGE_LLM=qwen-7b` runs end-to-end on the local machine, $0 LLM cost, results still upload to LangSmith.

### Where local lags

- **Render-shape discipline collapses (0.20 vs 0.80).** Gemma understands "call a render" but picks the wrong one most of the time — e.g. emits free text where `render_table` was expected.
- **Multi-step robustness is weakest.** One Gemma run hit the 12-iteration ceiling and never finished; Opus never does this on the same examples.
- **Answer completeness suffers.** Even when tools succeed, Gemma's prose answers more often skip cited URLs or stop short on multi-fact questions.
- **Wall time is ~3.5× slower.** Each Gemma turn is 20–60 s vs Opus's 2–10 s; with 3–5 turns per query, a 50-example pass projects to ~90 min local vs ~30 min on Opus.

### Why the gap exists

- **Model size.** Gemma 4 e4b is ~4 B effective params; frontier cloud models are in the hundreds of billions. Tool-calling and structured-output skills scale strongly with size.
- **Training mix.** Frontier cloud models are RLHF-tuned heavily on tool-calling and JSON-shape output; general-purpose local models get less of this.
- **Prompt length sensitivity.** The 1900-token system prompt + 8 bound tool definitions is digestible for Opus; smaller models get distracted by the same input.

### Judge calibration is real

The same Gemma agent outputs scored **0.44 from Haiku** and **0.68 from Qwen 7B** — a 24-point gap. Different judges, different rubric interpretation, not directly comparable. Pin a single judge across all experiments you want to compare.

### Recommendation matrix

| Use case | Pick |
|---|---|
| Cheap dev iteration, willing to accept quality dip | Local Gemma (free, ~90 min/50-eval) |
| Production demos or interview talking points | **Opus or Sonnet** |
| Best price/quality compromise | **gpt-4o-mini or Sonnet** (~$0.55 + ~15 min for 50 examples) |
| Air-gapped / offline / privacy-critical | Local Gemma + local Qwen judge |

### Reproduce these numbers

```bash
# Opus baseline
AGENT_LLM=opus uv run python scripts/run_experiment.py --limit 5 --prefix exp-opus

# Local Gemma + cloud Haiku judge
AGENT_LLM=gemma4-e4b OLLAMA_KEEP_ALIVE=24h \
  uv run python scripts/run_experiment.py --limit 5 --prefix exp-gemma4-e4b

# Fully local (agent + judge)
AGENT_LLM=gemma4-e4b JUDGE_LLM=qwen-7b OLLAMA_KEEP_ALIVE=24h \
  uv run python scripts/run_experiment.py --limit 5 --prefix exp-gemma-local-judge
```

LangSmith experiment IDs from this run:
- `exp-prompts-v2-689b30e1` — Opus baseline
- `exp-gemma4-e4b-1ac47263` — Gemma agent, Haiku judge
- `exp-gemma-local-judge-a5ce86d3` — Gemma agent, Qwen judge

---


The eval runner (`scripts/run_experiment.py`) takes the agent's [10/50-example golden dataset](../src/deepresearch/eval/dataset.py) and scores each run with four metrics:

| Scorer | Type | Cost | What it measures |
|---|---|---|---|
| `tool_recall` | programmatic | free | Fraction of `must_include_tools` actually called |
| `render_match` | programmatic | free | 1.0 if the run finished with the expected render tool |
| `iterations_used` | programmatic | free | Inverted ReAct-loop count (1.0 = single iteration, 0.0 = MAX) |
| `answer_correctness` | Haiku-as-judge | ~$0.005/run | 1–5 rubric rating, normalized to 0–1 |

## Prerequisites

```bash
# Project + Anthropic (default)
git clone https://github.com/agaonker/deepresearch.git && cd deepresearch
uv sync
cp .env.example .env   # then fill in keys
```

Required env vars in `.env`:
```bash
ANTHROPIC_API_KEY=sk-ant-...
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=lsv2_pt_...
LANGCHAIN_PROJECT=deepresearch-agent
```

## 1. Anthropic Claude (cloud)

Three Anthropic models registered: `opus`, `sonnet`, `haiku`.

```bash
# Single query — sanity check the path
uv run research --llm opus "What is the current stock price of NVDA?"

# 5-example eval (~$0.30 on opus, ~$0.05 on sonnet, ~$0.01 on haiku)
AGENT_LLM=opus uv run python scripts/run_experiment.py --limit 5 --prefix exp-opus

# Full 50-example eval (~$2.75 on opus)
AGENT_LLM=opus uv run python scripts/run_experiment.py --limit 50 --prefix exp-opus-full
```

Reference scores from PR #5 / PR #6 (5-example smoke):

| Scorer | Opus |
|---|---|
| `tool_recall` | 1.00 |
| `render_match` | 0.80 |
| `iterations_used` | 0.76 |
| `answer_correctness` | 0.68 |

Prompt caching is on automatically for `opus` and `sonnet` (per [the registry](../src/deepresearch/llm.py)). The `anthropic-beta: prompt-caching-2024-07-31` header and the `cache_control: ephemeral` block on the system message land your system prompt in the 5-minute cache — confirmed live with `cache_read_input_tokens ≈ 4878` on every call after the first.

## 2. OpenAI ChatGPT (cloud)

Two OpenAI models registered: `gpt-4o`, `gpt-4o-mini`.

```bash
# Install the OpenAI extra (langchain-openai)
uv sync --extra openai

# Add to .env
echo 'OPENAI_API_KEY=sk-...' >> .env
```

```bash
# Single query
uv run research --llm gpt-4o-mini "What is the current stock price of NVDA?"

# 5-example eval (~$0.05 on gpt-4o-mini, ~$0.30 on gpt-4o)
AGENT_LLM=gpt-4o-mini uv run python scripts/run_experiment.py --limit 5 --prefix exp-gpt-4o-mini
```

OpenAI auto-caches identical prefixes ≥1024 tokens server-side — no code action, no header — and surfaces the hit count under `usage.prompt_tokens_details.cached_tokens`.

## 3. Local Gemma via Ollama (free)

Three local models registered: `gemma4-e4b`, `gemma4-e2b`, `qwen-7b`.

```bash
# Install the Ollama extra
uv sync --extra ollama

# Start Ollama (in another shell) and pull the model
ollama serve &
ollama pull gemma4:e4b
```

```bash
# Single query — agent runs entirely on your machine
OLLAMA_KEEP_ALIVE=24h uv run research --llm gemma4-e4b \
  "What is the current stock price of NVDA?"

# 5-example eval — agent free, Haiku judge ~$0.03 total
AGENT_LLM=gemma4-e4b OLLAMA_KEEP_ALIVE=24h \
  uv run python scripts/run_experiment.py --limit 5 --prefix exp-gemma4-e4b

# Or completely free — programmatic scorers only, skip Haiku
AGENT_LLM=gemma4-e4b OLLAMA_KEEP_ALIVE=24h \
  uv run python scripts/run_experiment.py --limit 5 --no-llm-judge \
  --prefix exp-gemma4-free
```

`OLLAMA_KEEP_ALIVE=24h` keeps the model loaded between calls (default is 5 minutes), preserving the transformer KV cache so prefix-repeated calls are fast.

Measured Gemma 4 e4b scores (5-example smoke, run 2026-05-10):

| Scorer | Opus | Gemma 4 e4b | Δ |
|---|---|---|---|
| `tool_recall` | 1.00 | 0.80 | –0.20 |
| `render_match` | 0.80 | 0.20 | –0.60 |
| `iterations_used` | 0.76 | 0.69 | –0.07 |
| `answer_correctness` | 0.68 | 0.44 | –0.24 |

Observed Gemma behavior:

- **Tool calls**: usually right (BM25 narrows the catalog enough that even a smaller model picks correctly).
- **Render selection**: weak. Gemma often emits free text where a `render_table` or `render_card` was expected — it understands "use a render tool" but picks the wrong one.
- **Multi-tool / parallel calls**: shaky. The first example (weather comparison Tokyo + Paris) hit the 12-iteration cap and never finished — Gemma got stuck in a retry loop.
- **Citations**: tends to omit `sources=[]` even when explicitly instructed.

For local-first iteration this is fine. For demos, stay on Opus.

## Comparing experiments

Once you've run multiple experiments, open the LangSmith dataset and pick the "Experiments" tab:

```
https://smith.langchain.com → Datasets → deepresearch-golden-v1 → Experiments
```

Each `--prefix` you used becomes one column in the comparison view. Tip: prefix with the model name (`exp-opus`, `exp-gpt-4o-mini`, `exp-gemma4-e4b`) so the columns are self-labeling.

## Cost summary

| Pass | Provider | Approx. cost | Latency |
|---|---|---|---|
| 5 examples | Opus | $0.30 | 10–30 s/run |
| 5 examples | Sonnet | $0.05 | 5–15 s/run |
| 5 examples | Haiku | $0.01 | 3–10 s/run |
| 5 examples | gpt-4o | $0.30 | 10–30 s/run |
| 5 examples | gpt-4o-mini | $0.05 | 5–15 s/run |
| 5 examples | gemma4-e4b (agent) + Haiku (judge) | $0.03 | 60+ s/run on CPU, faster with GPU |
| 5 examples | gemma4-e4b (agent only, `--no-llm-judge`) | $0.00 | same |
| 50 examples | Opus | $2.75 | — |

Set a $50/month spend cap at <https://console.anthropic.com/settings/limits> as a safety net.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `ImportError: ... requires the 'openai' extra` | OpenAI package not installed | `uv sync --extra openai` |
| `httpx.ConnectError` on `--llm gemma4-e4b` | Ollama daemon not running | `ollama serve` |
| Gemma takes 60s+ per turn | Model unloads between calls | `OLLAMA_KEEP_ALIVE=24h` |
| `DeprecationWarning: LLM_PROVIDER + OLLAMA_MODEL are deprecated` | Using old env vars | Switch to `AGENT_LLM=<source-name>` |
| `unknown LLM source 'gemma3'` | Source not in registry | `uv run research --list-llms` to see options; add a new entry in `src/deepresearch/llm.py` |
