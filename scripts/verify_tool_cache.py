"""Prove the tool-def cache hits.

Fires the same agent LLM call twice back-to-back and reads input_tokens
vs cache_read_input_tokens from the Anthropic usage payload. After call 2,
the system prompt + tool defs should be served from cache.

Usage: uv run python scripts/verify_tool_cache.py
"""
from __future__ import annotations

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from deepresearch.llm import (
    build_chat,
    build_system_message,
    prepare_tools_for_caching,
    resolve,
)
from deepresearch.prompts import render_prompt
from deepresearch.tools.catalog import ALL_DATA_TOOLS
from deepresearch.tools.render import ALL_RENDER_TOOLS
from deepresearch.tools.retriever import ToolRetriever


def usage_of(resp):
    md = getattr(resp, "response_metadata", {}) or {}
    u = md.get("usage", {})
    return {
        "input_tokens": u.get("input_tokens"),
        "cache_creation_input_tokens": u.get("cache_creation_input_tokens"),
        "cache_read_input_tokens": u.get("cache_read_input_tokens"),
        "output_tokens": u.get("output_tokens"),
    }


def main() -> None:
    load_dotenv()
    src = resolve("agent", override="haiku")
    retriever = ToolRetriever(ALL_DATA_TOOLS + ALL_RENDER_TOOLS)
    query = "what is BM25 in one sentence"
    top = retriever.search(query, k=8)

    cacheable = prepare_tools_for_caching(top, src)
    sysmsg = build_system_message(render_prompt("system"), src)
    msgs = [sysmsg, HumanMessage(content=query)]

    llm = build_chat(src, streaming=False).bind_tools(cacheable)

    print("=== Call 1 (cold cache — populates) ===")
    r1 = llm.invoke(msgs)
    print(usage_of(r1))

    print()
    print("=== Call 2 (warm cache — should hit) ===")
    r2 = llm.invoke(msgs)
    print(usage_of(r2))


if __name__ == "__main__":
    main()
