# Screenshots

Drop PNGs here using these exact filenames so the README picks them up automatically.

## What to capture

Run `./scripts/demo.sh` first so LangSmith has fresh traces and your terminal has fresh output to screenshot.

### LangSmith (https://smith.langchain.com → project `deepresearch-agent`)

| Filename | View | Crop tip |
|---|---|---|
| `langsmith-project.png` | Project dashboard listing recent runs (after running `demo.sh`). | Crop to the runs table — show the metadata columns (`command_used`, `iterations`, `tool_count`). |
| `langsmith-trace-waterfall.png` | One trace fully expanded: `agent_node` → `bm25_tool_selection` → parallel tool calls → render. Pick demo #3 (the compare query) — it has the most parallelism. | Show the timing waterfall on the left and the run tree expanded. |
| `langsmith-tool-detail.png` | Click into one tool call (e.g. `stock_price`) and screenshot the input/output panel. | Show args, return value, and latency. |
| `langsmith-metadata.png` | The metadata sidebar of any run, showing `tags=[deepresearch, v1.0]` and our custom keys. | Just the metadata panel — don't include the full trace tree. |

### LangGraph Studio (`uv run langgraph dev`)

| Filename | View | Crop tip |
|---|---|---|
| `studio-graph.png` | The graph visualization — `agent ⇄ tools` with the conditional edge. | Show the whole graph + the legend. |
| `studio-run.png` | A run mid-execution, with the message thread on the right. Send demo #1 from inside Studio so you have a known query to point at. | Show the graph, the messages panel, and at least one highlighted node. |
| `studio-state.png` | The state inspector showing `iterations`, `cancelled`, `messages`. | Just the state panel. |

### Terminal

| Filename | View | Crop tip |
|---|---|---|
| `cli-render-qa.png` | Demo #1 final output (rendered Q/A card). | Crop to the painted box only. |
| `cli-render-table.png` | Demo #3 final output (NVDA/AMD comparison table). | Crop to the painted table only. |
| `cli-render-chart.png` | Demo #4 final output (India GDP chart). | Crop to the chart only. |

## Sizing

- Aim for **1600×1000 max** so the README pages load fast on GitHub.
- PNG, not JPG (sharper text).
- If a panel needs annotation, use red rectangles or arrows — keep the original screenshot too.
