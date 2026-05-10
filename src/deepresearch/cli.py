from __future__ import annotations

import argparse
import os
import signal
import sys
import uuid
from types import FrameType
from typing import Any

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.state import CompiledStateGraph

from deepresearch.commands.registry import dispatch
from deepresearch.graph.builder import build_graph
from deepresearch.streaming.render_cli import maybe_paint
from deepresearch.tools.catalog import ALL_DATA_TOOLS
from deepresearch.tools.render import ALL_RENDER_TOOLS
from deepresearch.tools.retriever import ToolRetriever


def _install_sigint_handler() -> dict[str, bool]:
    state: dict[str, bool] = {"cancelled": False}

    def handler(signum: int, frame: FrameType | None) -> None:
        state["cancelled"] = True
        print("\n[Ctrl-C received — finishing current step then stopping]", file=sys.stderr)

    signal.signal(signal.SIGINT, handler)
    return state


def _run_query(graph: CompiledStateGraph, text: str, *, thread_id: str) -> None:
    sigstate = _install_sigint_handler()

    result = dispatch(text)
    if result.kind == "pure":
        print(result.output)
        return

    config: RunnableConfig = {
        "configurable": {"thread_id": thread_id},
        "metadata": {
            "command_used": result.command_used,
            "iterations": 0,
            "tool_count": 0,
            "cancelled": False,
        },
        "tags": ["deepresearch", "v1.0"],
    }

    initial = {
        "messages": [HumanMessage(content=result.query or text)],
        "iterations": 0,
        "cancelled": False,
        "metadata": {"command_used": result.command_used},
    }

    try:
        for event in graph.stream(initial, config=config, stream_mode="values"):
            if sigstate["cancelled"]:
                print("[cancelled]", file=sys.stderr)
                graph.update_state(config, {"cancelled": True})
                return
            _print_latest(event)
    except KeyboardInterrupt:
        print("[cancelled]", file=sys.stderr)
        return


_PRINTED_IDS: set[str] = set()


def _print_latest(state: dict[str, Any]) -> None:
    messages = state.get("messages", [])
    if not messages:
        return
    msg = messages[-1]
    msg_id = getattr(msg, "id", None) or f"{type(msg).__name__}:{len(messages)}"
    if msg_id in _PRINTED_IDS:
        return
    _PRINTED_IDS.add(msg_id)

    if isinstance(msg, ToolMessage):
        painted = maybe_paint(msg.content if isinstance(msg.content, str) else "")
        if painted is not None:
            print(painted)
            return
        snippet = (msg.content if isinstance(msg.content, str) else str(msg.content))
        if len(snippet) > 600:
            snippet = snippet[:600] + " ..."
        print(f"[tool:{msg.name}] {snippet}")
        return

    if isinstance(msg, AIMessage):
        if msg.tool_calls:
            calls = ", ".join(f"{tc['name']}({_brief_args(tc.get('args', {}))})" for tc in msg.tool_calls)
            print(f"[agent → tools] {calls}")
        text = msg.content if isinstance(msg.content, str) else _join_blocks(msg.content)
        if text.strip():
            print(text)


def _brief_args(args: dict[str, Any]) -> str:
    items = []
    for k, v in list(args.items())[:3]:
        s = str(v)
        if len(s) > 40:
            s = s[:40] + "…"
        items.append(f"{k}={s}")
    return ", ".join(items)


def _join_blocks(content: list[Any]) -> str:
    parts: list[str] = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
    return "".join(parts)


def _print_llm_registry() -> None:
    from deepresearch.llm import SOURCES
    print(f"{'name':<18} {'provider':<10} {'model':<32} {'cache':<6} notes")
    print("-" * 100)
    for src in SOURCES.values():
        cache = "yes" if src.supports_prompt_cache else "—"
        print(f"{src.name:<18} {src.provider:<10} {src.model:<32} {cache:<6} {src.notes}")


def _explain_tools(query: str) -> None:
    retriever = ToolRetriever(ALL_DATA_TOOLS + ALL_RENDER_TOOLS)
    ranked = retriever.explain(query, k=12)
    print(f"Top tools for: {query!r}")
    for r in ranked:
        print(f"  {r.score:6.3f}  {r.name}")


def _repl(graph: CompiledStateGraph) -> None:
    print("DeepResearch REPL — type /help for commands, Ctrl-D to exit.")
    thread_id = str(uuid.uuid4())
    while True:
        try:
            text = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not text:
            continue
        if text in {"/exit", "/quit"}:
            return
        _PRINTED_IDS.clear()
        _run_query(graph, text, thread_id=thread_id)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="deepresearch")
    parser.add_argument("query", nargs="*", help="Research question (omit for REPL).")
    parser.add_argument("--explain-tools", action="store_true",
                        help="Print BM25 tool ranking for the query and exit.")
    parser.add_argument("--no-trace", action="store_true",
                        help="Disable LangSmith tracing for this run.")
    parser.add_argument("--llm", default=None,
                        help="LLM source name (e.g. opus, gemma4-e4b). See --list-llms.")
    parser.add_argument("--list-llms", action="store_true",
                        help="Print the registered LLM sources and exit.")
    args = parser.parse_args(argv)

    load_dotenv()
    if args.no_trace:
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
    if args.llm:
        # The agent and any sub-component reads AGENT_LLM via deepresearch.llm.resolve.
        os.environ["AGENT_LLM"] = args.llm

    if args.list_llms:
        _print_llm_registry()
        return 0

    text = " ".join(args.query).strip()

    if args.explain_tools:
        if not text:
            print("--explain-tools requires a query", file=sys.stderr)
            return 2
        _explain_tools(text)
        return 0

    graph = build_graph(checkpointer=MemorySaver())

    if not text:
        _repl(graph)
        return 0

    _run_query(graph, text, thread_id=str(uuid.uuid4()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
