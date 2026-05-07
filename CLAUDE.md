# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

This project uses `uv` for dependency and venv management. **All Python commands must be prefixed with `uv run`** so they execute inside the project venv.

```bash
# Run the agent
uv run python -m deepresearch.cli                              # REPL
uv run python -m deepresearch.cli "your query"                 # single-shot
uv run python -m deepresearch.cli --explain-tools "query"      # show BM25 ranking, no LLM call
uv run python -m deepresearch.cli --no-trace "query"           # disable LangSmith tracing
uv run langgraph dev                                           # LangGraph Studio (visual debugger)

# Tests / lint / types
uv run pytest tests/ -v
uv run pytest tests/test_retriever.py::test_name -v            # single test
uv run ruff check src/ tests/
uv run mypy src/deepresearch

# Deps
uv add <pkg>                                                   # runtime
uv add --dev <pkg>                                             # dev
```

`.env` must exist with `ANTHROPIC_API_KEY` (and ideally `LANGCHAIN_*`) — copy from `.env.example`. The CLI calls `load_dotenv()` at startup; tests do not, so tests that need keys must either mock or set env vars explicitly.

## Architecture

Single-agent ReAct loop on LangGraph. The same compiled graph (`src/deepresearch/graph/builder.py:graph`) is the entry point for both the CLI and `langgraph dev`/Studio — `langgraph.json` references it directly. Don't fork the graph for one entry point; change the shared definition.

**Request flow:**

1. `cli.py` reads input → `commands.registry.dispatch()` classifies it as `pure` (handled, print and exit), `expand` (slash command rewrites the query), or `passthrough` (free text).
2. The graph runs `agent_node` → conditional edge (`should_continue`) → `tool_node` → back to `agent_node`, capped at `MAX_ITERATIONS=12` in `graph/nodes.py`.
3. `agent_node` calls `ToolRetriever.search(query, k=8)` to BM25-rank the catalog, binds only the top-K tools (plus `ALWAYS_INCLUDE`) to the LLM, then invokes it with the system prompt.
4. The LLM either emits `tool_calls` (routed to `ToolNode`, executed in parallel) or finishes. The system prompt requires the agent to end with exactly one `render_*` call.
5. Render tools return a string starting with `_render::<kind>\n<json>`. `streaming/events.parse_render` recognizes the sentinel and `streaming/render_cli.maybe_paint` paints it as ASCII in the terminal. Other tool outputs are printed as truncated text.

**Two parallel concepts to keep in sync:**

- **Tool registration:** data tools live in `tools/catalog.py:ALL_DATA_TOOLS`; render tools in `tools/render.py:ALL_RENDER_TOOLS`. Both lists are concatenated into `_ALL_TOOLS` in `graph/nodes.py` (passed to `ToolNode`) AND in `commands/registry.py` (used by `/why` and `/tools`). A new tool only needs appending to one of the two lists — BM25 picks it up via name + docstring. Tools that should bypass BM25 ranking go in `tools/retriever.ALWAYS_INCLUDE`.
- **Render kinds:** adding a render tool requires three coordinated edits — the `@tool` in `render.py` (must return `f"{_SENTINEL}{kind}\n{json.dumps(payload)}"`), a `_paint_<kind>` painter in `streaming/render_cli.py`, and registration of that painter in `_PAINTERS`. The sentinel string `_render::` is duplicated in `render.py` and `streaming/events.py` — keep them identical.

**State and cancellation:** `graph/state.AgentState` is a `TypedDict` with `messages` (using `add_messages` reducer), `iterations`, `cancelled`, `metadata`. `cli._install_sigint_handler` flips a local flag on Ctrl-C; the stream loop then calls `graph.update_state(config, {"cancelled": True})` so `agent_node` and `should_continue` short-circuit on the next tick. Hard exit is Ctrl-D.

**LLM provider switch:** `graph/nodes._build_llm` reads `LLM_PROVIDER` (default `anthropic`, alternative `ollama`). Anthropic uses `streaming=True` and `max_tokens=4096`. Ollama needs a tool-calling-capable model (e.g. `qwen2.5:7b`) and `ollama serve` running locally. Tool-call quality on local models is much lower — fine for offline iteration, weaker for demos.

**Observability:** `LANGCHAIN_TRACING_V2=true` enables auto-tracing of `ChatAnthropic`, `ToolNode`, and node transitions. Custom spans are added via `@traceable` on `ToolRetriever.search` (`bm25_tool_selection`) and `commands.registry.dispatch` (`slash_command_dispatch`). Per-run metadata (`command_used`, `iterations`, `tool_count`, `cancelled`) and tags (`deepresearch`, `v1.0`) are attached in `cli._run_query`.

## Conventions

- **Tool error handling:** every external call in `tools/catalog.py` is wrapped in `try/except Exception as e: return f"[<tool_name> error: {e}]"`. Tool functions return `str`, never raise — the agent sees the error string and can recover. Mirror this when adding tools.
- **Type-checking strictness:** `mypy` runs with `check_untyped_defs`, `no_implicit_optional`, `warn_unused_ignores`, `warn_redundant_casts`. Use `from __future__ import annotations` (every module does). `ignore_missing_imports = true` is set globally — don't add per-import ignores unless mypy actually complains.
- **Ruff:** line length 110; `E501` and `B008` are ignored. `B008` matters because LangChain decorators wrap functions at module load.
- **System prompt:** the agent is *required* to finish with a render tool. If you change `_SYSTEM_PROMPT` in `graph/nodes.py`, preserve that contract or the CLI's painted output goes away.
