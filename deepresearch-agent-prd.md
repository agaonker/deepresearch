# Product Requirements Document
## DeepResearch Agent

> A ReAct tool-calling agent on LangGraph + LangChain. Local-only for v1.0.

| | |
|---|---|

| **Status** | Draft v1.1 |
| **Last updated** | May 2, 2026 |
| **Owner** | Ashish (solo) |
| **Scope** | Local development & demo. No deployment. |
| **Stack** | Python 3.11+ · LangGraph OSS · LangChain · LangSmith · Anthropic Claude |

---

## 1. Summary

DeepResearch Agent is a single-agent ReAct loop, run locally, that takes a natural-language research question and produces a cited report. It demonstrates six capabilities:

1. Parallel tool execution
2. BM25 dynamic tool selection
3. Token-level streaming
4. Mid-execution cancellation
5. Slash commands (deterministic intent expansion)
6. Agent-driven UI via render tools

Two ways to run it locally:

- **CLI** (`python -m deepresearch.cli`) — terminal-only, ASCII rendering, fastest to iterate.
- **`langgraph dev`** — local LangGraph Platform dev server with Studio UI. Same code, visual graph debugger.

Both observed via **LangSmith Developer tier** (free).

---

## 2. Goals & non-goals

### 2.1 Goals
- Build a runnable reference implementation of an agentic ReAct system that mirrors the architecture from the LiHA presentation (single agent role, loops until done).
- Showcase six concrete capabilities end-to-end.
- Keep total cost at **$0/month** subscription + Anthropic API token spend.
- Make every agent run inspectable in LangSmith.
- Make patterns transferable to ZenOnCall later.
- Serve as an interview talking point with live local demo.

### 2.2 Non-goals (v1.0)
- **Any kind of deployment** — no cloud, no Docker push, no public URL, no team sharing.
- Multi-user, multi-tenant scenarios.
- Persistent vector storage of research outputs — `MemorySaver` checkpointer is enough.
- Web frontend — render-tool payloads are JSON; CLI paints ASCII; that's all.
- Custom-trained or fine-tuned models — Claude Opus 4.7 via API.
- Production secrets management — local `.env` is fine.

---

## 3. Personas & primary user

**Primary user:** the developer (you). Solo, local machine, debugging and iterating.

**Secondary user:** interviewers and demo audiences sitting next to you (or watching a screen-share) while you run it live.

That's the entire user list. No remote users.

---

## 4. Third-party services & costs

| Service | Required? | Tier | Cost |
|---|---|---|---|
| **Anthropic API** | ✅ Yes | Pay-as-you-go | Per-token (~$5–$20 for dev; cents per demo run) |
| **LangChain** (framework) | ✅ Yes | OSS | $0 |
| **LangGraph** (OSS library) | ✅ Yes | OSS | $0 |
| **LangGraph CLI** (`langgraph dev`) | ✅ Yes | OSS | $0 — runs entirely on localhost |
| **LangSmith** (observability) | ✅ Yes | Developer | $0 (5K traces/mo, 14-day retention, 1 seat) |
| **LangGraph Platform Cloud** | ❌ No | n/a | n/a — we never deploy |
| **Vector DB / RAG store** | ❌ No | n/a | n/a |

**Total monthly subscription cost: $0.** Just Anthropic token usage.

### 4.1 LangSmith free-tier capacity sanity check

- Trace = one full agent invocation (whole ReAct loop, all tool calls).
- Solo dev: ~50 runs/day × 22 days ≈ 1,100 traces/month.
- Free tier: 5,000 traces/month → ~22% utilization. Plenty of headroom.

### 4.2 When to upgrade (informational only)

Not relevant for local-only v1.0, but for reference:
- **LangSmith Plus ($39/mo):** unlocks longer retention, multiple seats, and access to LangGraph Platform Cloud — none of which we need.
- **Anthropic spend:** set a $50/month limit in the console as a safety net.

---

## 5. Functional requirements

| ID | Requirement | Priority |
|---|---|---|
| **F-1** | Accept a free-text research question and produce a cited markdown report. | P0 |
| **F-2** | Run multiple independent tool calls from one LLM turn in parallel. | P0 |
| **F-3** | Bind only top-K tools to each LLM call, ranked by BM25 against the user query. | P0 |
| **F-4** | Stream model output token-by-token to the terminal. | P0 |
| **F-5** | Cancel a run cleanly on Ctrl-C; preserve checkpointed state for resume in same process. | P0 |
| **F-6** | Parse slash commands deterministically; route `/help`, `/tools`, `/why` as pure handlers and `/compare`, `/research`, `/summarize` as expanding handlers. | P0 |
| **F-7** | Provide four render-tool primitives: `render_card`, `render_table`, `render_chart`, `render_qa`. | P0 |
| **F-8** | Emit every agent run to LangSmith as a single trace with nested spans. | P0 |
| **F-9** | Tag traces with metadata: `command_used`, `iterations`, `tool_count`, `cancelled`. | P0 |
| **F-10** | REPL mode in the CLI (no positional arg = REPL; positional arg = single-shot). | P1 |
| **F-11** | `--explain-tools` flag to inspect BM25 ranking without running the agent. | P1 |
| **F-12** | Run via `langgraph dev` to get LangGraph Studio UI on localhost. | P1 |
| **F-13** | ≥80% test coverage on `commands/` and `tools/render.py`. | P1 |

---

## 6. Non-functional requirements

| Area | Requirement |
|---|---|
| **Latency** | First token within 3s of user input on a typical query. |
| **Cancellation** | Ctrl-C propagates within 500ms; in-flight tool calls drain naturally. |
| **Reliability** | Single tool failure does not crash the run — error surfaces inline. |
| **Observability** | 100% of runs visible in LangSmith. |
| **Cost ceiling** | Anthropic spend capped at $50/month via console. |
| **Setup time** | Clone-to-running ≤15 minutes on a clean Python machine. |

---

## 7. Architecture

```
            ┌──────────────────────┐
   input ──▶│ slash command parser │
            └──────┬───────────────┘
                   │ pure /help, /tools, /why → printed
                   │ /compare, /research, /summarize → (query, hint)
                   ▼
              ┌──────────────────────────────────┐
              │  Tool Catalog (~28 tools)        │
              │  data tools + render tools       │
              └────────────┬─────────────────────┘
                           │  BM25 (top-K) + ALWAYS_INCLUDE
                           ▼
   ┌──────────────────────────────────────────────────┐
   │                    agent                         │
   │   LLM with bind_tools(top_k_tools)               │
   │   - Picks data tools + render tools directly     │
   │   - Streams tokens as it reasons & writes        │
   └──────────────┬─────────────────────────┬─────────┘
                  │ tool_calls              │ final answer
                  ▼                         ▼
        ┌──────────────────┐          ┌──────────┐
        │   tool_executor  │          │   END    │
        │  (parallel via   │          └──────────┘
        │   ToolNode)      │
        └────────┬─────────┘
                 │ ToolMessages — render outputs
                 │ tagged with _render sentinel
                 └──────────────► back to agent (loop)

   ┌─────────────────────── observability ────────────────────────┐
   │  LANGCHAIN_TRACING_V2=true → every run streams to LangSmith. │
   │  One trace per run, nested spans for nodes and tool calls.   │
   └──────────────────────────────────────────────────────────────┘
```

The graph itself is identical whether you run it via `cli.py` or `langgraph dev`. Only the entry point differs.

---

## 8. LangSmith integration

### 8.1 What we trace

| Surface | Captured |
|---|---|
| Top-level run | Input query, system hint, final answer, latency, tokens. |
| `agent_node` | Top-K tools selected, system prompt, LLM I/O, tool calls emitted. |
| `ToolNode` | Each tool — args, result, latency, errors. |
| Render tools | Tagged `tool_kind=render` for filtering. |
| Cancellation | Marked `status=cancelled` in metadata. |

### 8.2 Enabling tracing (zero-code)

```bash
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_API_KEY=lsv2_pt_...
export LANGCHAIN_PROJECT=deepresearch-agent
```

That's it for built-ins (`ChatAnthropic`, `ToolNode`, `StateGraph` auto-trace).

### 8.3 Explicit instrumentation

Add `@traceable` only where LangChain doesn't auto-trace:

```python
from langsmith import traceable

@traceable(run_type="retriever", name="bm25_tool_selection")
def search(self, query: str, k: int = 8, *, always_include=None):
    ...
```

Modules with explicit decorators:
- `tools/retriever.py::ToolRetriever.search` — BM25 selection appears as a span.
- `commands/registry.py::Registry.dispatch` — slash command parsing visible.

### 8.4 Run metadata

```python
config = {
    "configurable": {"thread_id": thread_id},
    "metadata": {
        "command_used": "/compare" if was_slash else "free_text",
        "iterations": state["iterations"],
        "tool_count": len(top_tools),
        "cancelled": False,
    },
    "tags": ["deepresearch", "v1.0"],
}
```

---

## 9. Local setup guide

### 9.1 Prerequisites
- Python 3.11 or 3.12
- `pip` (or `uv`/`poetry`)
- Anthropic API key
- LangSmith account (free Developer plan)

### 9.2 One-time accounts

**Anthropic:**
1. Sign up at https://console.anthropic.com.
2. Add payment method.
3. Set monthly spend limit at $50.
4. Create API key. Save it.

**LangSmith:**
1. Sign up at https://smith.langchain.com — pick **Developer** plan.
2. Skip credit card.
3. Create project `deepresearch-agent`.
4. Settings → API Keys → Create. Save it.

### 9.3 Repo setup

```bash
git clone <repo-url>
cd deepresearch-agent
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 9.4 Environment variables

Create `.env` (gitignored):

```bash
ANTHROPIC_API_KEY=sk-ant-...
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=lsv2_pt_...
LANGCHAIN_PROJECT=deepresearch-agent
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
```

`cli.py` calls `load_dotenv()` automatically.

### 9.5 Run it (two ways)

**CLI mode — fastest:**
```bash
python -m deepresearch.cli                                    # REPL
python -m deepresearch.cli "Compare NVDA vs AMD last 3 quarters"
python -m deepresearch.cli --explain-tools "transformer papers"
```

**LangGraph Studio mode — for visual debugging:**

Add `langgraph.json` to repo root:

```json
{
  "dependencies": ["."],
  "graphs": {
    "deepresearch": "src/deepresearch/graph/builder.py:graph"
  },
  "env": ".env"
}
```

Expose a top-level `graph` symbol in `builder.py`:

```python
graph = build_graph()  # add at module scope
```

Then:

```bash
pip install langgraph-cli
langgraph dev
```

Opens Studio UI at `http://localhost:8123`. Click your graph, send a query, watch the loop run with full state visibility.

### 9.6 Verify tracing

After your first real run:
- Open https://smith.langchain.com → project `deepresearch-agent`.
- A trace should appear within seconds.
- Click in → you should see nested spans: `agent_node`, `bm25_tool_selection`, individual tool calls.

If nothing shows up: check that `LANGCHAIN_TRACING_V2=true` is exported and `LANGCHAIN_API_KEY` is correct.

### 9.7 Smoke test

```bash
pip install pytest
pytest tests/ -v
```

Expected: ~15 tests pass across `test_retriever.py`, `test_commands.py`, `test_render.py`.

---

## 10. Dependencies

`requirements.txt`:

```
langgraph>=0.2.50
langchain>=0.3.0
langchain-anthropic>=0.3.0
langchain-core>=0.3.0
langsmith>=0.1.140
rank-bm25>=0.2.2
pydantic>=2.0
httpx>=0.27
python-dotenv>=1.0.0
```

`requirements-dev.txt`:

```
pytest>=8.0
pytest-asyncio>=0.23
langgraph-cli>=0.1.50      # only needed for `langgraph dev` / Studio
```

---

## 11. Project layout

```
deepresearch-agent/
├── README.md
├── requirements.txt
├── requirements-dev.txt
├── .env.example
├── .gitignore
├── langgraph.json              # for `langgraph dev` / Studio
├── src/deepresearch/
│   ├── __init__.py
│   ├── cli.py                  # REPL + single-shot + cancellation
│   ├── tools/
│   │   ├── catalog.py          # ~24 data tools
│   │   ├── render.py           # 4 render tools
│   │   └── retriever.py        # BM25 (with @traceable)
│   ├── commands/
│   │   └── registry.py         # slash commands (with @traceable)
│   ├── graph/
│   │   ├── state.py
│   │   ├── nodes.py
│   │   └── builder.py          # exports `graph` for langgraph.json
│   └── streaming/
│       ├── events.py
│       └── render_cli.py
└── tests/
    ├── test_retriever.py
    ├── test_commands.py
    └── test_render.py
```

---

## 12. Milestones

| Milestone | Scope | Status |
|---|---|---|
| **M0: Skeleton** | Layout, BM25, agent graph, streaming, cancellation | ✅ Done |
| **M1: Slash commands + render tools** | Registry, 4 render tools, ASCII painter, REPL | ✅ Done |
| **M2: LangSmith wiring** | Env config, `@traceable` on retriever and dispatch, run metadata | ✅ Done |
| **M3: `langgraph dev` integration** | `langgraph.json`, top-level `graph`, Studio walkthrough | ✅ Done |
| **M3.5: Code-quality pass** | Type hints, docstrings, ruff + mypy clean | ✅ Done |
| **M4: Eval dataset** | 10 golden queries, pytest harness | 🔜 Post-v1.0 |
| **M5: mem0 cross-session memory** | OSS mem0 + Chroma, `memory_node`, `/memories` + `/forget`, PRD §2.2 amendment | 🔜 Parked (see plan) |

---

## 13. Risks & mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| LangSmith free quota exhausted during heavy iteration | Low | Low | Solo usage burns ~22% of quota; sampling env var ready if needed. |
| Anthropic spend spikes from a runaway loop | Low | Medium | `_MAX_ITERATIONS=12` hard cap; $50 console limit. |
| BM25 misses a critical tool for an unusual query | Medium | Low | `ALWAYS_INCLUDE` for utilities; `--explain-tools` to debug ranking. |
| LangChain/LangGraph minor-version API drift | Medium | Low | Pinned minimums; tests catch regressions. |
| `langgraph dev` port collision (8123 in use) | Low | Low | `langgraph dev --port <other>` flag. |

---

## 14. Open questions

1. Should `/why <query>` results be traced? They don't hit the LLM — leaning skip.
2. M4 evals: hosted LangSmith evaluators or local + upload? Local saves traces, hosted is simpler.
3. Do we want a `--no-trace` flag on the CLI for offline iteration? Probably yes — cheap to add.

---

## 15. Appendix: useful URLs

- LangSmith dashboard: https://smith.langchain.com
- LangSmith pricing: https://www.langchain.com/pricing
- LangGraph CLI docs: https://langchain-ai.github.io/langgraph/cloud/reference/cli/
- LangGraph Studio: https://studio.langchain.com
- Tracing setup: https://docs.smith.langchain.com/observability/how_to_guides/setup
