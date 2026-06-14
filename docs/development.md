# Development

## Run tests
```bash
uv run pytest tests/ -v
```

## Lint and type-check
```bash
uv run ruff check src/ tests/
uv run mypy src/deepresearch
```

## Dependencies
- Add a runtime dep: `uv add <package>`
- Add a dev dep: `uv add --dev <package>`

## Adding a new tool
1. Write a `@tool def my_tool(...) -> str:` function in [src/deepresearch/tools/catalog.py](../src/deepresearch/tools/catalog.py).
2. Append it to `ALL_DATA_TOOLS` at the bottom of the file.
3. Wrap external calls in `try/except Exception as e: return f"[my_tool error: {e}]"` (matches existing pattern).
4. Add a unit test in `tests/`.

The BM25 retriever picks new tools up automatically — no other registration needed. Tools listed in `ALWAYS_INCLUDE` (in [retriever.py](../src/deepresearch/tools/retriever.py)) are bound on every call regardless of relevance.

## Adding a new render tool
1. Write `@tool def render_xxx(...) -> str:` in [src/deepresearch/tools/render.py](../src/deepresearch/tools/render.py), returning `f"{_SENTINEL}xxx\n{json.dumps(payload)}"`.
2. Append it to `ALL_RENDER_TOOLS`.
3. Add a `_paint_xxx(d: dict) -> str:` painter in [src/deepresearch/streaming/render_cli.py](../src/deepresearch/streaming/render_cli.py) and register it in `_PAINTERS`.
4. Add tests covering the payload + painter.

---

# Troubleshooting

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
