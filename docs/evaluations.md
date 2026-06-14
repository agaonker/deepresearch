# Evaluations

Two layers of eval, both using the same 50-example golden dataset (`src/deepresearch/eval/dataset.py`, mirrored to LangSmith as `deepresearch-golden-v1`):

1. **BM25 recall** (free, deterministic, runs in CI) — `tests/test_eval.py` asserts that for every golden query, the BM25 retriever's top-K includes the expected tools. Currently 100% recall at K=8.
2. **Agent behavior** (LangSmith experiment, paid) — `scripts/run_experiment.py` runs the compiled graph against each example and uploads scores to LangSmith.

```bash
uv run python scripts/run_experiment.py                 # 5 examples, ~$0.30 (smoke test)
uv run python scripts/run_experiment.py --limit 50      # full pass, ~$2.75
uv run python scripts/run_experiment.py --no-llm-judge  # skip Haiku, free programmatic only
```

## Scorers

| Scorer | Type | What it measures |
|---|---|---|
| `tool_recall` | programmatic | Fraction of `must_include_tools` actually called |
| `render_match` | programmatic | 1.0 if the run finished with the expected render tool |
| `iterations_used` | programmatic | Inverted cost — 1.0 for a single iteration, 0.0 at the 12-iteration cap |
| `answer_correctness` | Haiku-as-judge (~$0.005/run) | 1–5 rating from a strict evaluator, normalized to 0–1 |

## Sample experiment results

A 5-example smoke test (experiment `exp-20260509-052214-6c1aa708`):

![LangSmith experiment overview](screenshots/langsmith-experiment-overview.png)

Per-row scores:

![LangSmith experiment scores](screenshots/langsmith-experiment-scores.png)

| Scorer | Avg on 5 runs | Read |
|---|---|---|
| `tool_recall` | **1.00** | Perfect — agent always reaches the BM25-expected tool. |
| `iterations_used` | **0.76** | Healthy — most runs finish in 3–4 ReAct loops. |
| `render_match` | **0.40** | 2 of 5 hit the expected render. The agent often picks `render_card` when the dataset prescribes `render_table` — the LLM uses its own judgment for render shape. |
| `answer_correctness` | **0.32** | Haiku is grading 1–2/5 on most answers. The judge prompt is strict and lacks ground truth — first-cut signal, needs prompt tuning. |

Cost on the smoke pass: ~60K tokens, latency 9–30s per run.
