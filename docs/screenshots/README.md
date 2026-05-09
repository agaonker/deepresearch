# Screenshots

Drop PNGs here using these exact filenames so the README picks them up automatically.

## What to capture

Run `./scripts/demo.sh` first so LangSmith has fresh traces and your terminal has fresh output to screenshot.

### LangSmith (https://smith.langchain.com → project `deepresearch-agent`)

| Filename | View | Crop tip | Status |
|---|---|---|---|
| `langsmith-trace-waterfall.png` | One trace in **Waterfall** mode — full ReAct loop with timing bars (`agent` → `bm25_tool_selection` → `ChatAnthropic` → `tools` → render). | Show timing bars on the left and the Input/Output panel on the right. | ✅ Captured |
| `langsmith-trace-tree.png` | Same trace in vertical **Tree** mode — shows per-node duration, token counts, and the model in use. | Include token-count badges and model labels. | ✅ Captured |
| `langsmith-experiment-overview.png` | Experiments view of an eval run — full table with all scorer columns + cost/latency. | Show the experiment header (`exp-...`) and at least 3 example rows. | ✅ Captured |
| `langsmith-experiment-scores.png` | Same experiment, zoomed to the per-row scorer cells with red/green color coding. | Crop to the scorer columns; keep the column averages visible at the top. | ✅ Captured |
| `langsmith-project.png` (optional) | Project dashboard listing recent runs (after running `demo.sh`). | Crop to the runs table — show the metadata columns (`command_used`, `iterations`, `tool_count`). | ☐ |
| `langsmith-tool-detail.png` (optional) | Click into one tool call (e.g. `stock_price`) and screenshot the input/output panel. | Show args, return value, and latency. | ☐ |

### LangGraph Studio (`uv run langgraph dev`)

| Filename | View | Crop tip | Status |
|---|---|---|---|
| `studio-graph.png` | The graph visualization — `__start__` → `agent ⇄ tools` → `__end__`. | Show the whole graph. | ✅ Captured |
| `studio-run.png` (optional) | A run mid-execution, with the message thread on the right. Send demo #1 from inside Studio so you have a known query to point at. | Show the graph, the messages panel, and at least one highlighted node. | ☐ |
| `studio-state.png` (optional) | The state inspector showing `iterations`, `cancelled`, `messages`. | Just the state panel. | ☐ |

### Terminal (CLI)

| Filename | View | Crop tip | Status |
|---|---|---|---|
| `cli-explain-tools.png` | `research --explain-tools "..."` output — BM25 ranking with no LLM call. | Show the prompt + the top tools with their scores. | ✅ Captured |
| `cli-arxiv-rlhf-table.png` | A real run that ends in `render_table` — paints a comparison of SFT/RLHF arxiv papers. | Crop to the painted table. | ✅ Captured |
| `cli-arxiv-rlhf-summary.png` | The agent's narrative summary + cited source URLs that follow the table. | Show the source list and at least one paragraph of the summary. | ✅ Captured |

## Sizing

- Aim for **1600×1000 max** so the README pages load fast on GitHub.
- PNG, not JPG (sharper text).
- If a panel needs annotation, use red rectangles or arrows — keep the original screenshot too.
