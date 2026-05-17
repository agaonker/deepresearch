# Reviewer Agent — Design Sketch

> **🅿 STATUS: PARKED**
> Implementation paused pending compute sponsorship. The pre-flight gate (Phase 0) costs ~$3-4 in API calls; the full experiment is ~$25-65. Picking this back up requires either a sponsor for the eval spend OR a willingness to absorb the cost out of pocket. All architectural decisions below are still load-bearing; resume by running Phase 0 first.

Directional validation of Apple's "Reinforced Agent: Inference-time Feedback for Tool-Calling LLMs" ([paper](https://machinelearning.apple.com/research/reinforced-agent-inference-feedback)) inside this repo's LangGraph ReAct loop.

---

## What the paper claims

A secondary **reviewer agent** inspects each proposed tool call before it executes. The reviewer can correct misroutes the base agent makes, but it can also degrade calls that were already correct — so the paper introduces two metrics:

- **Helpfulness** — fraction of base-agent errors the reviewer fixes.
- **Harmfulness** — fraction of correct base-agent calls the reviewer breaks.
- **Ratio** — helpfulness / harmfulness. The paper's best (o3-mini reviewer + GPT-4o agent) hits 3.0.

Headline numbers: +5.5% on irrelevance detection (BFCL), +7.1% on multi-turn (τ²-Bench), +1.5–2.8% additional from GEPA prompt optimization.

**Key insight:** the reviewer is *decoupled* from the agent. It can be a different (often smaller, more specialized) model, optimized independently.

---

## Where errors can come from — the BM25 / LLM-pick decomposition

This was the load-bearing question that shaped the experiment design. Tool-call errors in our pipeline come from two distinct stages:

```
query → BM25(catalog, k=8) → LLM picks 1 from 8 → ToolNode runs it
         ┌──────────────┐    ┌──────────────────┐
         │ retrieval     │    │ selection         │
         │ accuracy      │    │ accuracy          │
         └──────────────┘    └──────────────────┘
```

1. **BM25 miss** — the right tool wasn't in the top-8. Neither LLM nor reviewer can recover; the right tool was never on the table.
2. **LLM mis-pick** — the right tool was in the top-8, but the LLM picked a different one.

**The reviewer can only fix #2.** It inspects what the LLM picked from a set BM25 already filtered. So the reviewer's *maximum possible* helpfulness is bounded by the LLM mis-pick rate. If our agent already picks correctly from the BM25 top-8 ~95% of the time, the addressable error surface is ~5% — small, and any harmfulness easily eats it.

**Two metrics that matter before building anything:**

```
P(BM25 includes right tool)        ← test_bm25_includes_expected_tools already measures this
P(LLM picks right tool | BM25 hit) ← Phase 0 below measures this
```

The product of these is the end-to-end accuracy ceiling for an agent-only pipeline. The reviewer can only attack errors where `BM25 hit AND LLM miss` — the "LLM had the right answer in its candidate set and still picked wrong" cases.

## Phase 0 — Pre-flight gate (haiku-only, ~$3-4)

**Run this before any reviewer code lands.** It tells you whether the entire experiment is worth running.

1. Run all 50 golden queries through `haiku-4-5` via the existing CLI, 1 rep each (`temperature=0` keeps it near-deterministic). Use `--llm haiku`.
2. For each query: was the expected `render_*` tool actually called? Was BM25's top-8 hit?
3. Compute `P(haiku picks right | BM25 hit)`.

**Decision rule:**

| Result for haiku                       | Action                                                                   |
|----------------------------------------|--------------------------------------------------------------------------|
| ≥ 0.95                                 | Stop. Smarter models are at least this high → no addressable error surface → scrap or pivot to BM25 reranker (different class of error). |
| 0.85 – 0.95                            | Borderline. Run sonnet too (~$10 more) before deciding.                  |
| < 0.85                                 | Real error surface exists. Proceed to Phase 1 with confidence.           |

**Why haiku-only for the pre-flight:** haiku has the lowest tool-call accuracy of the three Anthropic-registered models, so it's the **lower bound** on LLM-pick accuracy across our lineup. If even haiku is ≥ 0.95, opus and sonnet are definitely ≥ 0.95 too — no error surface anywhere. This is a $3 binary "is the experiment worth running" check.

**Caveat:** haiku-only is for the *pre-flight*, not the reviewer experiment itself. The reviewer experiment needs the asymmetric pairings in Phase 1 (see model-pairing matrix below) — self-review by the same model (haiku reviewing haiku) is the least informative cell in the matrix.

## How it maps to this repo

### Current flow

```
agent_node ──► should_continue ──► tool_node ──► agent_node ──► …
   (LLM)         (router)          (parallel)      (LLM)
                                   ToolNode
```

### Proposed flow

```
agent_node ──► should_review ──► reviewer_node ──► tool_node ──► agent_node ──► …
   (LLM)         (router)          (LLM, smaller)    (parallel)     (LLM)
                                        │
                                        ├─ ALLOW ─► tool_node
                                        └─ BLOCK ─► agent_node (with feedback message)
```

`reviewer_node` sits between agent and tools. Reads the last agent message's `tool_calls`, decides per-call, attaches feedback to state.

### Pieces we already have

| Piece                           | Where                                           | Reused for                                |
|---------------------------------|-------------------------------------------------|-------------------------------------------|
| Multi-model registry            | `src/deepresearch/llm.py`                       | Reviewer can be any registered model      |
| Eval harness                    | `tests/test_eval.py`, `scripts/run_experiment.py` | Compute helpfulness / harmfulness        |
| Scorers                         | `src/deepresearch/eval/scorers.py`              | Add `helpfulness`, `harmfulness` scorers  |
| LangSmith tracing               | `@traceable` on agent + retriever               | Inspect reviewer decisions per call       |
| ToolRetriever                   | `src/deepresearch/tools/retriever.py`           | Reviewer can see same candidate set       |

### Pieces we need to add

- `src/deepresearch/graph/reviewer.py` — the reviewer node + LLM call
- Edges in `src/deepresearch/graph/builder.py` — agent → reviewer → tool, BLOCK loop
- A `review_with: str | None` field in `AgentState` (graph state)
- A `--review-with <model-name>` CLI flag in `cli.py`
- Two scorers in `eval/scorers.py`: `helpfulness`, `harmfulness`
- Labeled ground truth in the golden set: per-query, the expected tool call (we already have `expected_tools` per `test_eval.py` — extend it)

---

## Reviewer contract

```python
# Input: the agent's last message + its proposed tool_calls + the agent's reasoning text
# Output: a per-call decision, with reasoning

class ReviewDecision(TypedDict):
    tool_call_id: str
    verdict: Literal["allow", "block"]   # "edit" is v2
    reason: str                          # one sentence; surfaces in trace + feedback loop

# Reviewer prompt shape:
#   - User query (original)
#   - Agent's reasoning that led to this call
#   - Proposed tool name + args
#   - Optional: top-K candidate tools from BM25 (so reviewer can see what wasn't picked)
#   - Task: is this call appropriate? If not, why?
```

### Block behavior

When `verdict == "block"`, the reviewer's `reason` becomes a `ToolMessage` with `name=<tool>` and content `[reviewer blocked: <reason>]`. Agent re-runs with this feedback in context. Same iteration counter — counts against `MAX_ITERATIONS=12`. If the agent proposes the same blocked call twice, force-allow on the third attempt to prevent loops.

### Edit behavior (v2, not in prototype)

Reviewer rewrites args. Riskier — reviewer is now a co-author, not just a gate. Defer until we've validated the ALLOW/BLOCK case.

---

## Metrics

For a labeled query `q` with expected tool call `t*` (name + args), and the agent's actual final tool call `t̂`:

```
correct(q) := t̂.name == t*.name AND t̂.args ⊇ t*.required_args
```

Run each query under both regimes:
- **Baseline:** agent only (no reviewer)
- **Reviewed:** agent + reviewer

Per query, four outcomes:

|                    | baseline correct | baseline wrong |
|--------------------|------------------|----------------|
| reviewed correct   | TP (no change)   | FIX (helpful)  |
| reviewed wrong     | BREAK (harmful)  | TN (no change) |

```
helpfulness = FIX / (FIX + TN)           # of wrongs, how many got fixed
harmfulness = BREAK / (BREAK + TP)       # of corrects, how many got broken
ratio       = helpfulness / harmfulness  # paper's headline number
net_lift    = (FIX - BREAK) / |Q|        # % of queries net-improved
```

Both metrics need ground-truth `t*`. The existing eval set's `expected_tools` field (a set of acceptable tool names per query) is a starting point — we'd extend it to a single canonical name + required args.

---

## Model-pairing matrix

The paper's interesting finding is that a **smaller/different** reviewer can guide a larger agent. The specific pairing the paper used was **o3-mini reviewer + GPT-4o agent** — a small *reasoning* model reviewing a larger general model. That's the load-bearing asymmetry, not just "small vs large."

### Mapping the paper's pairing to Anthropic

| Paper            | Anthropic equivalent                                   |
|------------------|--------------------------------------------------------|
| GPT-4o agent     | **sonnet-4-6** (closest cost+capability tier) or opus-4-7 (pure capability) |
| o3-mini reviewer | **haiku-4-5 with extended thinking enabled** — Anthropic ships thinking as a feature flag, not a separate SKU |

**Anthropic doesn't have an o3-mini-equivalent SKU.** It ships `thinking={"type": "enabled", "budget_tokens": N}` as a per-call parameter on any 4.x model. So the closest analog to "small reasoning model" is `haiku-4-5 + thinking_on`. Without thinking, you're testing a small *general* reviewer, not a small *reasoning* reviewer — and the paper's headline 3:1 ratio came from the reasoning asymmetry.

**Project-state caveat:** `grep -rn "thinking|reasoning_effort|extended_thinking" src/deepresearch/` currently returns zero hits. Enabling thinking requires ~30 minutes of work:
- Add `supports_thinking: bool` field to `LLMSource` in `llm.py`
- Pass `thinking={"type": "enabled", "budget_tokens": N}` to `ChatAnthropic` in `_build_anthropic` when enabled
- Add a `--reviewer-thinking-budget <N>` CLI flag

Without this, the experiment doesn't faithfully replicate the paper.

### Pairings to run (5 cells)

| agent                | reviewer    | hypothesis                                      |
|----------------------|-------------|-------------------------------------------------|
| opus                 | (none)      | Baseline ceiling. What's the agent's solo score? |
| opus                 | haiku       | Cheap reviewer guarding strong agent — paper's headline pattern |
| opus                 | sonnet      | Peer reviewer. Helpfulness up, harmfulness up?  |
| haiku                | haiku       | Weak agent + weak reviewer. Net win?            |
| gemma4-e4b           | haiku       | Local agent + cloud reviewer — practical hybrid for the "local agent, cloud sanity check" pattern |

Cost note: every cell except `(opus, none)` doubles inference *per turn that has a tool call*. For Gemma agent + Haiku reviewer this is ~free in dollars (Haiku is cheap) but adds ~300-500ms reviewer latency.

---

## Eval shape

### Phase 1 — full matrix on existing labeled set (only if Phase 0 shows error surface)

- Existing **50-query** golden set in `src/deepresearch/eval/dataset.py` (asserted by `tests/test_eval.py:25`).
- Extend each query's `expected_tools` from a set of names to `{name: str, required_args: dict}`.
- Wire extended thinking on the reviewer (haiku-with-thinking — see Anthropic mapping above).
- Run the 5-cell matrix with **K=3 replications per cell** under the paired design (Section 1 above). K=3 is enough at `temperature=0`; drop to K=1 if a 10-query sample shows < 5% disagreement across reps.
- Compute helpfulness, harmfulness, ratio, net_lift, overhead×, plus McNemar p-values and bootstrap 95% CIs per pairing.
- Publish the table from Section 7 to `docs/reviewer-agent-results.md`.
- **Decision gate:** apply the pre-registered rule (Section 6) — any pairing hitting `ratio ≥ 2.0`, `p < 0.10`, `overhead ≤ 1.5×` triggers Phase 2. Below that, we either drop the project or move to Phase 1b (judge-graded scaling).
- **n=50 power:** detects ratio ≥ 2.0 with p < 0.10 reliably, ratio ~1.5 borderline. Small absolute deltas (1-3%) like the paper's GEPA increment still need Phase 3 (BFCL, n≈2000).

### Phase 2 — local benchmark (if Phase 1 is promising)

- Expand golden set to ~50 queries across categories: definition lookup (`render_qa`), comparison (`render_table`), numeric (`render_chart`), hierarchy (`render_tree`), dated (`render_timeline`), single-concept (`render_card`).
- Adversarially label 10-15 queries known to misroute (e.g., "compare X" that the agent answers as `render_card` instead of `render_table`).
- Statistical power: with n=50 and reasonable effect sizes (paper saw ~5-7% absolute deltas), confidence intervals get tight enough to publish.

### Phase 3 — public benchmark (aspirational)

- Add a BFCL adapter at `src/deepresearch/eval/bfcl_adapter.py`. BFCL ships JSON cases; map them onto our render-tool catalog where possible, skip where not.
- Now directly comparable to the paper. This is real work — BFCL has ~2000 cases across multiple categories, and our tool catalog is research-flavored, not function-call-flavored, so coverage will be partial.

---

## Quantitative measurement methodology

The point of the prototype is to **measure** whether the reviewer pattern lifts the agent — not to vibe-check it. The measurement design has to support a statistical comparison, not just a side-by-side print.

### 1. Paired design

Every query runs under both regimes (baseline + reviewed) with the **same seed, same retriever output, same model temperature**. The unit of analysis is the *(query, regime)* pair, so we control for query difficulty. This eliminates "the reviewed run got the easy questions" as a confound.

Concretely, the experiment runner emits one row per `(query × regime × replication)` tuple to a JSONL file:

```jsonl
{"query_id":"q01","regime":"baseline","rep":1,"agent":"opus","reviewer":null,"tool_call":"render_table","args":{...},"correct":true,"latency_ms":2840,"input_tokens":1820,"output_tokens":210}
{"query_id":"q01","regime":"reviewed","rep":1,"agent":"opus","reviewer":"haiku","tool_call":"render_table","args":{...},"correct":true,"latency_ms":3210,"input_tokens":1820,"output_tokens":210,"reviewer_input_tokens":540,"reviewer_output_tokens":40,"reviewer_verdict":"allow","reviewer_reason":"appropriate for comparison query"}
```

### 2. Replications

LLM calls are nondeterministic even at `temperature=0` (especially with tool-calling). Each `(query × regime)` cell gets **K=5 replications**. We take the mode of `correct` across reps as the cell's outcome (majority vote), and report variance separately. K=5 gives a clean tie-break and bounds the per-cell compute at 5× the baseline.

### 3. Significance test

The right test for paired binary outcomes is **McNemar's exact test** on the 2×2 contingency table of baseline-correct vs reviewed-correct (after collapsing replications per cell). It tests whether the off-diagonal counts (`FIX` and `BREAK` from the table above) are symmetric.

```python
# Pseudocode for the per-pairing statistical readout
from statsmodels.stats.contingency_tables import mcnemar

# After collapsing K reps per cell to a single correct/wrong outcome:
#                    reviewed_correct  reviewed_wrong
#  baseline_correct        TP              BREAK
#  baseline_wrong         FIX               TN
table = [[TP, BREAK], [FIX, TN]]
result = mcnemar(table, exact=True)   # exact binomial when n<25
# result.pvalue, result.statistic
```

Pair this with **bootstrap 95% CIs** on helpfulness, harmfulness, ratio, and net_lift (10,000 resamples of the query set, stratified by baseline-correct vs baseline-wrong).

### 4. Power realism — what we can and can't detect

With **n=50** paired queries (the actual golden set size in `src/deepresearch/eval/dataset.py`):

| Effect size                                       | Detectable at p<0.10? | Confidence interval width |
|---------------------------------------------------|-----------------------|---------------------------|
| Reviewer fixes 10/15 wrong queries (huge effect)  | yes                   | ±10% on helpfulness       |
| Reviewer fixes 6/15 wrong queries (medium)        | yes                   | ±15%                      |
| Reviewer fixes 2/15 wrong queries (small)         | borderline            | ±18%                      |
| Reviewer fixes 1/15 wrong queries (~paper's GEPA delta) | no             | ±22%                      |

**Implication:** n=50 detects medium-to-large effects (ratio ≥ 2.0) with reasonable confidence. Detecting the paper's 5.5% irrelevance delta or its 1.5-2.8% GEPA delta still needs Phase 3 (BFCL, n≈2000) — those are small-effect-size tests our golden set isn't large enough to nail. We're answering "does the reviewer pattern produce a meaningful lift in this codebase," not "do Apple's specific numbers replicate."

### 4b. Realistic cost estimate (resume-time budget)

| stage             | model lineup                                  | rough cost | wall clock |
|-------------------|-----------------------------------------------|-----------|------------|
| Phase 0 preflight | haiku only, 50 queries, 1 rep                 | ~$3-4     | ~15 min    |
| Phase 1 full      | opus + sonnet + haiku, 50 queries, K=3 reps   | ~$25-55   | ~75 min    |
| Phase 1 minimal   | sonnet + haiku only (drop opus), K=1 rep      | ~$5-10    | ~25 min    |

**Caching nuance:** Anthropic prompt caching is wired up only for the **system prompt** (`llm.py:281-295`, `build_system_message` adds `cache_control: {"type": "ephemeral"}`). Tool definitions (~2000 tokens per call, re-sent every call) are NOT cached — `bind_tools` in `graph/nodes.py:40` doesn't pass `cache_control`. So caching saves ~30% off input cost, not the 80% you'd see if tool defs were also cached. The corrected numbers above assume that real ~30% savings.

**Cost optimization TODO:** wrapping the bound tool definitions in `cache_control: ephemeral` would cut another ~30-50% off the Phase 1 budget. Requires either a custom binding or switching to the raw Anthropic SDK for the tool block (LangChain's `bind_tools` doesn't expose `cache_control` directly). One-line change at minimum, half-day if it needs a clean abstraction. Logged as a P3 follow-up.

### 5. Cost & latency as first-class metrics

Helpfulness alone is meaningless without cost. Every regime reports:

| Metric                       | Why it matters                                            |
|------------------------------|-----------------------------------------------------------|
| `latency_p50_ms`             | What the user actually waits                              |
| `latency_p95_ms`             | Tail experience                                           |
| `tokens_in_per_query`        | Cloud reviewer cost driver                                |
| `tokens_out_per_query`       | Output bandwidth                                          |
| `usd_per_query`              | Combined cost using current model prices                  |
| `reviewer_overhead_ratio`    | `latency_reviewed / latency_baseline` — the "tax"         |

Decision-relevant view: **helpfulness gain per dollar of reviewer overhead.** If a haiku reviewer adds $0.0003/query and fixes 20% of errors on an opus agent, that's almost free. If a sonnet reviewer adds $0.005/query and fixes 25%, the per-fix cost is much higher.

### 6. Pre-registered decision rule

Before running anything, commit to thresholds so the result isn't post-hoc rationalized:

- **Ship the reviewer pattern** if: any pairing achieves **ratio ≥ 2.0** with **McNemar p < 0.10** AND **reviewer_overhead_ratio ≤ 1.5×**.
- **Investigate further** if: 1.0 ≤ ratio < 2.0 (signal but not strong enough).
- **Abandon** if: ratio < 1.0 (reviewer is net negative) OR overhead > 2.5× regardless of ratio.

The p-value bar is loose (0.10, not 0.05) because at n=50 we're using it as a sanity filter on direction, not a confirmatory test on a specific effect magnitude.

### 7. Reporting format

Every prototype run produces `docs/reviewer-agent-results.md` with this table at the top:

```
Reviewer Agent — Phase 1 Results (n=50, K=3 reps)
┌────────────┬────────────┬─────────────┬─────────────┬───────┬─────────┬───────────┬──────────┐
│ agent      │ reviewer   │ helpfulness │ harmfulness │ ratio │ net_lift│ overhead× │ p (Mcnem)│
├────────────┼────────────┼─────────────┼─────────────┼───────┼─────────┼───────────┼──────────┤
│ opus       │ (none)     │       —     │       —     │   —   │    —    │   1.00    │    —     │
│ opus       │ haiku      │ 0.60 ±0.31  │ 0.13 ±0.18  │  4.6  │  +0.30  │   1.18    │   0.07   │
│ opus       │ sonnet     │ 0.80 ±0.24  │ 0.25 ±0.21  │  3.2  │  +0.40  │   1.45    │   0.04   │
│ haiku      │ haiku      │ 0.40 ±0.27  │ 0.40 ±0.27  │  1.0  │   0.00  │   1.92    │   0.50   │
│ gemma4-e4b │ haiku      │ 0.50 ±0.30  │ 0.20 ±0.22  │  2.5  │  +0.20  │   1.34    │   0.13   │
└────────────┴────────────┴─────────────┴─────────────┴───────┴─────────┴───────────┴──────────┘
```
(Numbers above are illustrative — actual results TBD.)

Below the table, list every disagreement: `q07: baseline picked render_card, reviewed picked render_table (correct). q03: baseline picked render_chart (correct), reviewed blocked, agent retried with render_card (wrong).` That qualitative trace is where the design insights actually live.

### 8. Optional Phase 1b — LLM judge for unlabeled scaling

If Phase 1 results are promising but power-limited, run a **judge-graded** parallel evaluation:

- Generate 100+ queries unlabeled (template-driven or human-curated).
- Run both regimes.
- Have a strong LLM (Opus, separate from agent/reviewer) judge each tool call's appropriateness. We already have `JUDGE_LLM` infrastructure from the eval suite.
- Validate judge accuracy on the 10-labeled set first (target: ≥90% agreement with human labels). If judge accuracy is high, scale up.
- Caveat: judge bias is real. Report agent-vs-judge agreement, not just regime deltas. If the judge favors a particular tool, all reviewer/agent comparisons under that judge inherit the bias.

This gets us to n≈100 without expanding the human-labeled set, at the cost of a softer ground truth.

---

## Open questions to resolve before building

1. **Does the reviewer see candidate alternatives?** Pure agent-call review (just "is this call OK?") vs. retrieval-aware review (here are the BM25 top-K, here's what was picked, was the right one chosen?). The second is strictly more powerful but ties the reviewer to our retrieval system.

2. **Parallel tool calls.** When agent proposes 3 calls in one turn (parallel via `ToolNode`), does the reviewer judge each independently or as a set? Paper assumes single call. We'd default to per-call review, but if all 3 are blocked the agent gets 3 feedback messages in one turn — could overwhelm.

3. **Render tool review.** Should the *final* `render_*` call be reviewed? The agent is required by system prompt to end with one, and review-and-block on the final render means the agent loops forever. Default: don't review render tools, only data tools.

4. **Reviewer cost-aware skip.** Should we skip review for low-stakes / cheap tools (e.g., `render_qa`, free local computation)? Adds complexity, may not be worth it for the prototype.

5. **Ground truth.** "Correct tool call" is sometimes ambiguous — a query like "what's BM25?" can be answered by `wikipedia_search` OR by the agent's own knowledge via `render_qa`. Do we accept either? The paper sidesteps this by using BFCL where ground truth is explicit. We'll need a judging policy.

---

## Risks

- **Reviewer adds latency without value on easy queries.** ~80% of our golden set is "obviously right" calls. Reviewer pays cost on all of them; helps on ~20%. Total wall-clock per session ↑.
- **Reviewer-blocks-correct (harmfulness) is the killer failure mode.** A 1.5:1 ratio is borderline; below 1:1 the system is worse than nothing. Need to surface this clearly in eval.
- **n=50 power ceiling.** Detects medium-to-large effects (ratio ≥ 2.0). Does NOT detect the paper's small-delta findings (5.5% irrelevance, 1.5-2.8% GEPA) — those need BFCL-scale n.
- **Render-tool exclusion is load-bearing.** If review fires on the final `render_*` call and blocks it, the CLI prints raw text instead of a painted box. Test coverage must catch this.
- **Reviewer prompt drift.** The reviewer's prompt is the whole product. GEPA-style optimization (the paper's +2%) is a separate workstream we're explicitly deferring.

---

## Phased plan

1. **Sketch (this doc)** — done.
2. **Phase 0 — pre-flight** — haiku-only, no reviewer, measure `P(LLM picks right | BM25 hit)` on the 50-query set. ~$3-4 / ~15 min wall clock. Decision gate: if haiku ≥ 0.95, scrap or pivot to BM25 reranker; else proceed.
3. **Phase 1 — full matrix prototype** — `reviewer.py` node + 5-cell matrix + 2 scorers on the 50-query set, with extended thinking wired for the haiku reviewer. ~$25-55. Half a day with CC. Run, publish results table, decision gate: if any pairing hits ratio ≥ 2.0 / p < 0.10 / overhead ≤ 1.5×, proceed; else stop or run Phase 1b.
4. **Phase 1b — judge-graded scaling** (optional) — only if Phase 1 shows borderline signal. Generate 100+ unlabeled queries, run both regimes, use opus-as-judge to grade. ~$50-80. 1 day.
5. **Phase 2 — BFCL adapter** (aspirational) — only if Phase 1 (or 1b) is conclusive. Pull Berkeley Function Call Leaderboard, build adapter for our render-tool catalog. 2-3 days. Direct comparability with the paper.

Each phase has a gate: continue only if the prior phase produces signal. Don't sink time into BFCL before the prototype shows that reviewer agents help *at all* in our setting.

---

## Files that would change (preview)

```
src/deepresearch/graph/
  builder.py            # Add reviewer edge + conditional router
  reviewer.py           # NEW — the reviewer node + LLM call
  state.py              # Add review_with: str | None
  nodes.py              # Maybe — touch _build_llm if reviewer LLM needs different config

src/deepresearch/eval/
  scorers.py            # NEW scorers: helpfulness, harmfulness, ratio
  golden_queries.py     # Extend expected_tools schema

src/deepresearch/cli.py # --review-with flag

scripts/run_experiment.py  # Pass review_with through to runs

docs/reviewer-agent-results.md  # NEW — generated table (Phase 2 output)

tests/test_reviewer.py # NEW — unit tests for the node + scorers
tests/test_eval.py     # Extend with reviewer regimes
```

No changes to `streaming/`, `tools/`, or `prompts/` are expected.

---

## What this doc is and isn't

It **is**: an architectural sketch with a pre-registered quantitative measurement plan — paired design, K=3 replications, McNemar p-values, bootstrap CIs on helpfulness/harmfulness/ratio, plus cost & latency overhead. A Phase 0 pre-flight gate ($3-4) that decides whether the full experiment is worth running. A faithful Anthropic-side mapping of the paper's o3-mini + GPT-4o pairing (haiku-with-thinking + sonnet).

It **isn't**: a replication of the paper's headline numbers. Our n=50 labeled set detects medium-to-large effects (ratio ≥ 2.0) reliably, but not the paper's small-delta findings (5.5% irrelevance, 1.5-2.8% GEPA). For that you need Phase 2 (BFCL, n≈2000). The framing for Phase 1 results: "we measured the reviewer pattern with the following ratio, CI, p-value, and cost overhead in a small LangGraph ReAct agent." Not "we replicated Apple's numbers."

## Resume checklist

When sponsorship lands, work through this in order:

- [ ] Read this doc top to bottom; confirm the BM25-vs-LLM-pick decomposition still holds (i.e., that the agent + retriever code in `src/deepresearch/graph/` and `src/deepresearch/tools/retriever.py` hasn't structurally changed).
- [ ] Run Phase 0 first. Hard gate. Spend $3-4 before spending $50.
- [ ] If Phase 0 passes the gate, wire extended thinking into `llm.py:LLMSource` (add `supports_thinking: bool` field + pass `thinking={"type":"enabled","budget_tokens":N}` to `ChatAnthropic` in `_build_anthropic`).
- [ ] Build the `reviewer_node` per the contract in the "Reviewer contract" section. ALLOW/BLOCK only; defer EDIT to v2.
- [ ] Resolve the 5 open questions in the "Open questions" section *before* writing the reviewer prompt (especially #1: does the reviewer see BM25 candidates? #3: don't review render tools).
- [ ] Run Phase 1 with the pre-registered decision rule. No post-hoc threshold massaging.
- [ ] Publish the results table to `docs/reviewer-agent-results.md` regardless of outcome — a clean negative result is valuable.
