# Demo & screenshots

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

## Example terminal output

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

## CLI in action

`--explain-tools` prints the BM25 ranking for a query without calling the LLM — useful for debugging tool selection:

![--explain-tools output for an SFT/RLHF query](screenshots/cli-explain-tools.png)

A real run on the same topic — agent calls `arxiv_search`, then renders a comparison table of papers:

![Rendered table of SFT/RLHF arxiv papers](screenshots/cli-arxiv-rlhf-table.png)

The agent then writes a narrative summary with cited source URLs:

![Narrative summary with arxiv source citations](screenshots/cli-arxiv-rlhf-summary.png)

## LangSmith traces

Every run with `LANGCHAIN_TRACING_V2=true` shows up at https://smith.langchain.com → project `deepresearch-agent`.

The waterfall view shows the full ReAct loop end-to-end — `agent_node` → `bm25_tool_selection` → `ChatAnthropic` → `tools` (`stock_price`) → back to `agent` → render:

![LangSmith waterfall trace](screenshots/langsmith-trace-waterfall.png)

The vertical tree view of the same trace shows per-node timing, token counts, and the model in use:

![LangSmith trace tree with token counts](screenshots/langsmith-trace-tree.png)

## LangGraph Studio

Run the same compiled graph in the visual debugger:

```bash
uv run langgraph dev
```

![LangGraph Studio graph](screenshots/studio-graph.png)

> See [`screenshots/README.md`](screenshots/README.md) for a capture checklist (which views to shoot, file naming, sizing).
