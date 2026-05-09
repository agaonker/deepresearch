# DeepResearch Agent

A local-only ReAct tool-calling research agent built on **LangGraph + LangChain**, observed via **LangSmith**, optionally visualised in **LangGraph Studio**.

Single-agent loop, runs on your machine, $0 monthly subscription cost beyond Anthropic API token spend.

---

## What it does

Takes a natural-language research question, picks the most relevant tools via BM25, calls them (in parallel where possible), streams reasoning to the terminal, and finishes with a structured render tool (`render_qa`, `render_card`, `render_table`, `render_chart`, `render_timeline`, or `render_tree`).

Six demonstrated capabilities:
1. **Parallel tool execution** via LangGraph's `ToolNode`.
2. **BM25 dynamic tool selection** — only the top-K tools are bound to each LLM call.
3. **Token-level streaming** to the terminal.
4. **Mid-execution cancellation** — Ctrl-C cleanly stops the loop.
5. **Slash commands** — `/help`, `/tools`, `/why`, `/research`, `/compare`, `/summarize`.
6. **Agent-driven UI** — render tools emit a `_render::` sentinel that the CLI paints as ASCII.

---

## Project status

| Milestone | Scope | Status |
|---|---|---|
| **M0** | Skeleton, agent graph, BM25, streaming, cancellation | ✅ Done |
| **M1** | Slash commands + render tools | ✅ Done |
| **M2** | LangSmith wiring (`@traceable`, run metadata) | ✅ Done |
| **M3** | `langgraph dev` Studio integration | ✅ Done |
| **M3.5** | Code-quality pass (type hints, ruff + mypy clean) | ✅ Done |
| **M4** | Eval dataset (10 golden queries, pytest harness) | ✅ Done |
| **M5** | mem0 cross-session memory | 🔜 Parked (see plan file) |

**36 data tools + 6 render tools** in the catalog today. Tests: **50/50 passing** (35 unit + 15 eval).

---

## Quick start

### 1. Prerequisites
- Python 3.11+
- [`uv`](https://docs.astral.sh/uv/) (used for dependency and venv management)
- Anthropic API key (with funded credits — $5 minimum)
- LangSmith account (free Developer plan)

### 2. Clone & install
```bash
git clone https://github.com/agaonker/deepresearch.git
cd deepresearch
uv sync
```

### 3. Configure environment
```bash
cp .env.example .env
# edit .env with your real keys
```

Minimum required:
```bash
ANTHROPIC_API_KEY=sk-ant-...
ANTHROPIC_MODEL=claude-opus-4-5         # or claude-sonnet-4-6 for cheaper iteration
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=lsv2_pt_...           # Personal Access Token from smith.langchain.com
LANGCHAIN_PROJECT=deepresearch-agent
```

### 4. Run it (three ways)

**REPL** — interactive, fastest iteration:
```bash
uv run research
```

**Single-shot** — one query and exit:
```bash
uv run research "Compare NVDA vs AMD last quarter revenue"
```

**LangGraph Studio** — visual graph debugger in browser:
```bash
uv run langgraph dev
# opens https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024
```

### 5. Verify tracing
After your first run, open https://smith.langchain.com → project `deepresearch-agent`. Each query appears as a trace with nested spans (agent_node, ChatAnthropic, ToolNode, individual tools, `bm25_tool_selection`, `slash_command_dispatch`).

---

## CLI options

```bash
uv run research                              # REPL
uv run research "your query"                 # single-shot
uv run research --explain-tools "query"      # show BM25 ranking, no LLM call
uv run research --no-trace "query"           # disable LangSmith tracing
```

> Tip: activate the venv once (`source .venv/bin/activate`) and the prefix drops entirely — just `research "your query"`.

In the REPL, slash commands work as typed:
- `/help` — list commands
- `/tools` — list all tools in the catalog
- `/why <query>` — show top-K BM25 tools for a query (no LLM call)
- `/research <topic>` — deep research with citations
- `/compare <a> vs <b>` — side-by-side comparison
- `/summarize <text or url>` — summarize and render as a card
- `/exit` or Ctrl-D — quit

---

## Demo & screenshots

A scripted demo runs five queries chosen to exercise each render tool — handy for smoke-testing a fresh checkout, generating LangSmith traces, and capturing example output.

```bash
./scripts/demo.sh                 # all 5 queries with LangSmith tracing on
./scripts/demo.sh --no-trace      # skip LangSmith
./scripts/demo.sh --explain-only  # BM25 ranking only, no LLM call (free)
./scripts/demo.sh 3               # only run query #3
```

| # | Expected render | Query |
|---|---|---|
| 1 | `render_qa` | Current stock price + P/E for NVDA |
| 2 | `render_card` | 5-sentence Wikipedia summary of the French Revolution |
| 3 | `render_table` | NVDA vs AMD: market cap, P/E, 52-week range |
| 4 | `render_chart` | India GDP, World Bank, 2015–2023 |
| 5 | `render_timeline` | Major SpaceX launches, 2008–2024 |

### Example terminal output

Demo #1 captured live — the agent picks `stock_price`, then renders a card:

```text
[agent → tools] stock_price(ticker=NVDA)
[tool:stock_price] **NVIDIA Corporation** (NVDA)
Price: $215.2
Market Cap: $5,230,436,024,320
P/E Ratio: 43.828922
52W Range: 115.21 – 217.8
Volume: 134,128,204
[agent → tools] render_card(title=NVIDIA Corporation (NVDA) Stock Overview, ...)
+----------------------------------------------------------------------------+
| NVIDIA Corporation (NVDA) Stock Overview                                   |
+----------------------------------------------------------------------------+
| NVIDIA's current stock price is **$215.20** with a P/E ratio of **43.83**. |
+----------------------------------------------------------------------------+
| Current Price: $215.20                                                     |
| P/E Ratio: 43.83                                                           |
| Market Cap: $5.23 Trillion                                                 |
| 52-Week Range: $115.21 – $217.80                                           |
| Volume: 134,128,204                                                        |
+----------------------------------------------------------------------------+
```

### CLI in action

`--explain-tools` prints the BM25 ranking for a query without calling the LLM — useful for debugging tool selection:

![--explain-tools output for an SFT/RLHF query](docs/screenshots/cli-explain-tools.png)

A real run on the same topic — agent calls `arxiv_search`, then renders a comparison table of papers:

![Rendered table of SFT/RLHF arxiv papers](docs/screenshots/cli-arxiv-rlhf-table.png)

The agent then writes a narrative summary with cited source URLs:

![Narrative summary with arxiv source citations](docs/screenshots/cli-arxiv-rlhf-summary.png)

### LangSmith traces

Every run with `LANGCHAIN_TRACING_V2=true` shows up at https://smith.langchain.com → project `deepresearch-agent`.

The waterfall view shows the full ReAct loop end-to-end — `agent_node` → `bm25_tool_selection` → `ChatAnthropic` → `tools` (`stock_price`) → back to `agent` → render:

![LangSmith waterfall trace](docs/screenshots/langsmith-trace-waterfall.png)

The vertical tree view of the same trace shows per-node timing, token counts, and the model in use:

![LangSmith trace tree with token counts](docs/screenshots/langsmith-trace-tree.png)

### LangGraph Studio

Run the same compiled graph in the visual debugger:

```bash
uv run langgraph dev
```

![LangGraph Studio graph](docs/screenshots/studio-graph.png)

> See [`docs/screenshots/README.md`](docs/screenshots/README.md) for a capture checklist (which views to shoot, file naming, sizing).

---

## Tool catalog (36 + 6 = 42 total)

### Data tools by category

**Web search & news** (5)
DuckDuckGo: `web_search`, `ddg_news`, `ddg_image_search` · Hacker News: `hackernews_search` · Reddit: `reddit_search`

**Encyclopedic & factual** (4)
Wikipedia: `wiki_summary`, `wiki_search` · Wikidata SPARQL: `wikidata_query` · Country info: `get_country_info`

**Academic literature** (4)
arXiv: `arxiv_search`, `arxiv_paper_details` · PubMed: `pubmed_search` · Crossref: `crossref_search`

**Finance & markets** (6)
Stocks (yfinance): `stock_price`, `stock_financials`, `stock_earnings`, `stock_news`, `market_summary` · Crypto: `coingecko_price`

**Geography & weather** (3)
Geocoding: `osm_geocode` · Weather: `weather_now`, `open_meteo_forecast`

**Macro & demographics** (1)
World Bank indicators: `world_bank_indicator`

**Web fetch & document extraction** (3)
HTTP: `fetch_url`, `fetch_url_headers` · PDF: `pdf_extract_text`

**Multimedia** (1)
YouTube transcripts: `youtube_transcript`

**Developer ecosystem** (2)
GitHub: `get_github_repo` · PyPI: `search_pypi`

**Utilities** (7)
Math: `calculate` · Time: `get_current_datetime` · Units: `convert_units` · Currency: `currency_convert` · Dictionary: `define_word` · Network: `get_public_ip_info` · Text: `summarize_text`

### Render tools (6)
`render_card`, `render_table`, `render_chart`, `render_qa`, `render_timeline`, `render_tree` — each emits a `_render::<kind>\n<json>` sentinel that the CLI parses and paints as ASCII.

---

## Architecture

```
                ┌──────────────────────┐
   input ─────▶│ slash command parser │
                └──────┬───────────────┘
                       │ pure /help, /tools, /why → printed
                       │ /compare, /research, /summarize → expanded query
                       ▼
                ┌──────────────────────────────────┐
                │  Tool Catalog (~42 tools)        │
                │  data tools + render tools       │
                └────────────┬─────────────────────┘
                             │  BM25 (top-K) + ALWAYS_INCLUDE
                             ▼
   ┌──────────────────────────────────────────────────┐
   │                    agent_node                    │
   │   ChatAnthropic.bind_tools(top_k_tools)          │
   │   - emits tool_calls or final text               │
   └──────────────┬─────────────────────────┬─────────┘
                  │ tool_calls              │ no tool_calls
                  ▼                         ▼
        ┌──────────────────┐          ┌──────────┐
        │   ToolNode       │          │   END    │
        │   (parallel)     │          └──────────┘
        └────────┬─────────┘
                 │ ToolMessages — render outputs tagged _render::
                 └──────────────► back to agent_node (loop)

   ┌─────────────── observability ─────────────────────┐
   │ LANGCHAIN_TRACING_V2=true → all runs to LangSmith │
   │ One trace per run, nested spans for nodes & tools │
   └───────────────────────────────────────────────────┘
```

Same compiled graph runs in both the CLI and Studio — only the entry point differs.

---

## Project layout

```
deepresearch/
├── README.md                          # this file
├── .env.example                       # copy to .env and fill in
├── langgraph.json                     # for `langgraph dev` / Studio
├── pyproject.toml                     # deps, ruff, mypy, pytest config
├── deepresearch-agent-prd.md          # product requirements doc
├── scripts/demo.sh                    # 5-query demo runner
├── docs/screenshots/                  # README assets (drop PNGs here)
├── src/deepresearch/
│   ├── cli.py                         # REPL + single-shot + cancellation
│   ├── tools/
│   │   ├── catalog.py                 # 36 data tools
│   │   ├── render.py                  # 6 render tools
│   │   └── retriever.py               # BM25 selector (with @traceable)
│   ├── commands/registry.py           # slash commands (with @traceable dispatch)
│   ├── graph/
│   │   ├── state.py                   # AgentState TypedDict
│   │   ├── nodes.py                   # agent_node, tool_node, MAX_ITERATIONS=12
│   │   └── builder.py                 # StateGraph + MemorySaver, exports `graph`
│   └── streaming/
│       ├── events.py                  # parses _render:: sentinel
│       └── render_cli.py              # ASCII painters
├── src/deepresearch/eval/
│   └── dataset.py                    # 10 golden queries (M4)
└── tests/
    ├── test_retriever.py              # 6 tests
    ├── test_commands.py               # 11 tests
    ├── test_render.py                 # 18 tests
    └── test_eval.py                   # 15 tests — golden BM25 harness
```

---

## Switching LLM providers

The agent reads `LLM_PROVIDER` from the environment.

### Anthropic (default)
```bash
LLM_PROVIDER=anthropic
ANTHROPIC_MODEL=claude-opus-4-5         # or claude-sonnet-4-6, claude-haiku-4-5
```

### Local Ollama (no API cost)
```bash
LLM_PROVIDER=ollama
OLLAMA_MODEL=qwen2.5:7b                 # or llama3.1:8b — must support tool calls
OLLAMA_BASE_URL=http://localhost:11434
```

Requires `ollama serve` running and the model pulled (`ollama pull qwen2.5:7b`). Tool-calling quality on local models is much lower than Opus — fine for offline iteration, weaker for demos. The 12-iteration cap (`MAX_ITERATIONS`) protects you from runaway loops.

---

## Development

### Run tests
```bash
uv run pytest tests/ -v
```

### Lint and type-check
```bash
uv run ruff check src/ tests/
uv run mypy src/deepresearch
```

### Dependencies
- Add a runtime dep: `uv add <package>`
- Add a dev dep: `uv add --dev <package>`

### Adding a new tool
1. Write a `@tool def my_tool(...) -> str:` function in [src/deepresearch/tools/catalog.py](src/deepresearch/tools/catalog.py).
2. Append it to `ALL_DATA_TOOLS` at the bottom of the file.
3. Wrap external calls in `try/except Exception as e: return f"[my_tool error: {e}]"` (matches existing pattern).
4. Add a unit test in `tests/`.

The BM25 retriever picks new tools up automatically — no other registration needed. Tools listed in `ALWAYS_INCLUDE` (in [retriever.py](src/deepresearch/tools/retriever.py)) are bound on every call regardless of relevance.

### Adding a new render tool
1. Write `@tool def render_xxx(...) -> str:` in [src/deepresearch/tools/render.py](src/deepresearch/tools/render.py), returning `f"{_SENTINEL}xxx\n{json.dumps(payload)}"`.
2. Append it to `ALL_RENDER_TOOLS`.
3. Add a `_paint_xxx(d: dict) -> str:` painter in [src/deepresearch/streaming/render_cli.py](src/deepresearch/streaming/render_cli.py) and register it in `_PAINTERS`.
4. Add tests covering the payload + painter.

---

## Costs

| Item | Cost |
|---|---|
| LangChain (OSS framework) | $0 |
| LangGraph (OSS library) | $0 |
| LangGraph CLI / `langgraph dev` | $0 (local server) |
| LangSmith Developer plan | $0 (5K traces/mo, 14-day retention) |
| All 36 data tools | $0 (no API keys, no OAuth) |
| **Anthropic API** | per-token pay-as-you-go |

Typical query costs: **$0.01–0.10 on Opus**, less on Sonnet/Haiku. Set a $50/mo spend cap at https://console.anthropic.com/settings/limits.

---

## Observability

Every run streams to LangSmith automatically when `LANGCHAIN_TRACING_V2=true`:

- **Auto-traced** (no code): `ChatAnthropic` calls, `ToolNode` invocations, every node transition.
- **Custom spans** (via `@traceable`): `bm25_tool_selection` (in `ToolRetriever.search`), `slash_command_dispatch` (in `commands.registry.dispatch`).
- **Run metadata** attached via `config["metadata"]`: `command_used`, `iterations`, `tool_count`, `cancelled`.
- **Tags**: `deepresearch`, `v1.0`.

Filter, search, and replay any run from https://smith.langchain.com.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `anthropic.BadRequestError: credit balance is too low` | Account has $0 in credits | Add funds at console.anthropic.com/settings/billing |
| `anthropic.NotFoundError: model: ...` | `ANTHROPIC_MODEL` is wrong | Try `claude-opus-4-5` or `claude-sonnet-4-6` |
| `anthropic.AuthenticationError` | `ANTHROPIC_API_KEY` missing or wrong | Check `.env`, regenerate key if needed |
| No LangSmith trace appears | `LANGCHAIN_TRACING_V2` not `true`, or wrong key | Verify all `LANGCHAIN_*` env vars |
| `langgraph: Required package 'langgraph-api' is not installed` | Missing extra | `uv add --dev "langgraph-cli[inmem]"` |
| Studio shows no graph | `langgraph.json` path wrong | Confirm `src/deepresearch/graph/builder.py:graph` exists |
| `http://127.0.0.1:2024/` is blank in browser | Working as designed — server has no `/` route | Open the Studio URL the CLI prints, or visit `/docs` |
| REPL hangs after Ctrl-C | Caught by SIGINT handler, finishing current step | Hit Ctrl-D to force exit |

---

## Reference

- **PRD**: [deepresearch-agent-prd.md](deepresearch-agent-prd.md)
- **LangGraph docs**: https://langchain-ai.github.io/langgraph/
- **LangSmith dashboard**: https://smith.langchain.com
- **LangGraph Studio**: https://studio.langchain.com
- **Anthropic console**: https://console.anthropic.com
