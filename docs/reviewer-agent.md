# Reviewer Agent — Design Sketch

Directional validation of Apple's "Reinforced Agent: Inference-time Feedback for Tool-Calling LLMs" ([paper](https://machinelearning.apple.com/research/reinforced-agent-inference-feedback)) inside this repo's LangGraph ReAct loop.

**Status:** sketch only. No code yet. Implementation gated on user approval.

---

## What the paper claims

A secondary **reviewer agent** inspects each proposed tool call before it executes. The reviewer can correct misroutes the base agent makes, but it can also degrade calls that were already correct — so the paper introduces two metrics:

- **Helpfulness** — fraction of base-agent errors the reviewer fixes.
- **Harmfulness** — fraction of correct base-agent calls the reviewer breaks.
- **Ratio** — helpfulness / harmfulness. The paper's best (o3-mini reviewer + GPT-4o agent) hits 3.0.

Headline numbers: +5.5% on irrelevance detection (BFCL), +7.1% on multi-turn (τ²-Bench), +1.5–2.8% additional from GEPA prompt optimization.

**Key insight:** the reviewer is *decoupled* from the agent. It can be a different (often smaller, more specialized) model, optimized independently.

---

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

The paper's interesting finding is that a **smaller/different** reviewer can guide a larger agent. We'd test that asymmetry directly. Initial matrix (5 cells, not all 25):

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

### Phase 1 — measurable on existing labeled set

- Existing 10-query golden set in `tests/test_eval.py`.
- Extend each query's `expected_tools` from a set of names to `{name: str, required_args: dict}`.
- Run the 5-cell matrix with **K=5 replications per cell** under the paired design (Section 1 above).
- Compute helpfulness, harmfulness, ratio, net_lift, overhead×, plus McNemar p-values and bootstrap 95% CIs per pairing (Sections 2-4).
- Publish the table from Section 7 to `docs/reviewer-agent-results.md`.
- **Decision gate:** apply the pre-registered rule (Section 6) — any pairing hitting `ratio ≥ 2.0`, `p < 0.10`, `overhead ≤ 1.5×` triggers Phase 2. Below that, we either drop the project or move to Phase 1b (judge-graded scaling).
- **Honest caveat on n=10:** we'll detect large effects (ratio ≥ 3) reliably, medium effects borderline, small effects not at all. We're answering "does the pattern *work at all* in this codebase," and "what's the cost/benefit shape" — not "do Apple's specific numbers replicate."

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

With n=10 paired queries:

| Effect size                                      | Detectable at p<0.05? | Confidence interval width |
|--------------------------------------------------|-----------------------|---------------------------|
| Reviewer fixes 5/5 wrong queries (huge effect)   | yes                   | ±20% on helpfulness       |
| Reviewer fixes 3/5 wrong queries (medium)        | borderline            | ±30%                      |
| Reviewer fixes 1/5 wrong queries (small)         | no                    | ±35%                      |

**Implication:** n=10 detects "the reviewer pattern fundamentally helps or hurts." It does not detect 5-10% absolute deltas like the paper reports. To match the paper's CI tightness we need Phase 2 (n≈50) or Phase 3 (BFCL, n≈2000).

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

The p-value bar is loose (0.10, not 0.05) because n=10 power is low — we're using it as a sanity filter, not a confirmatory test.

### 7. Reporting format

Every prototype run produces `docs/reviewer-agent-results.md` with this table at the top:

```
Reviewer Agent — Phase 1 Results (n=10, K=5 reps)
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
- **n=10 noise floor.** Any helpfulness number ± noise from 10 queries has a 95% CI of roughly ±20-30 percentage points. We can publish the direction but not the magnitude.
- **Render-tool exclusion is load-bearing.** If review fires on the final `render_*` call and blocks it, the CLI prints raw text instead of a painted box. Test coverage must catch this.
- **Reviewer prompt drift.** The reviewer's prompt is the whole product. GEPA-style optimization (the paper's +2%) is a separate workstream we're explicitly deferring.

---

## Phased plan

1. **Sketch (this doc)** — done.
2. **Prototype** — `reviewer.py` node + 5-cell matrix + 2 scorers on the existing 10-query set. Half a day with CC. Run, publish results table, decide whether to continue.
3. **Expand to 50 queries** — if Phase 2 shows ratio > 1.5 on at least one pairing, grow the golden set. 1-2 days. This gets us a defensible blog post.
4. **BFCL adapter** — only if 3 is conclusive. 2-3 days. Direct comparability with the paper.

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

It **is**: an architectural sketch with a pre-registered quantitative measurement plan — paired design, K=5 replications, McNemar p-values, bootstrap CIs on helpfulness/harmfulness/ratio, plus cost & latency overhead. Sufficient to drive a 5-cell prototype that produces a defensible measurement (not a vibe-check).

It **isn't**: a replication of the paper's headline numbers. Our n=10 labeled set caps detectable effect sizes to medium-or-larger. To match the paper's CI tightness we need Phase 2 (n≈50) or Phase 3 (BFCL, n≈2000). The framing for Phase 1 results: "we measured the reviewer pattern with the following ratio, CI, p-value, and cost overhead in a small LangGraph ReAct agent." Not "we replicated Apple's numbers."
