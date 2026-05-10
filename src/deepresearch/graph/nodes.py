from __future__ import annotations

import os
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from langgraph.prebuilt import ToolNode

from deepresearch.graph.state import AgentState
from deepresearch.prompts import render_prompt
from deepresearch.tools.catalog import ALL_DATA_TOOLS
from deepresearch.tools.render import ALL_RENDER_TOOLS
from deepresearch.tools.retriever import ToolRetriever

MAX_ITERATIONS = 12

_ALL_TOOLS = ALL_DATA_TOOLS + ALL_RENDER_TOOLS
_RETRIEVER = ToolRetriever(_ALL_TOOLS)


def _build_llm() -> BaseChatModel:
    provider = os.getenv("LLM_PROVIDER", "anthropic").lower()
    if provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=os.getenv("OLLAMA_MODEL", "qwen2.5:7b"),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            temperature=0,
        )
    from langchain_anthropic import ChatAnthropic
    # The `anthropic-beta: prompt-caching-2024-07-31` header is required for
    # `cache_control` markers on content blocks to be honored — without it the
    # API silently ignores them (verified empirically on opus-4-7, sonnet-4-6,
    # haiku-4-5). The marker itself is added to the SystemMessage in
    # `agent_node` below.
    return ChatAnthropic(  # type: ignore[call-arg]
        model=os.getenv("ANTHROPIC_MODEL", "claude-opus-4-5"),
        streaming=True,
        max_tokens=4096,
        default_headers={"anthropic-beta": "prompt-caching-2024-07-31"},
    )


def agent_node(state: AgentState) -> dict[str, Any]:
    if state.get("cancelled"):
        return {
            "messages": [AIMessage(content="[run cancelled]")],
            "iterations": state.get("iterations", 0),
        }

    iterations = state.get("iterations", 0)
    if iterations >= MAX_ITERATIONS:
        return {
            "messages": [AIMessage(content=f"[max iterations ({MAX_ITERATIONS}) reached — stopping]")],
            "iterations": iterations,
        }

    messages = state["messages"]
    query = _extract_query(messages)
    top_tools = _RETRIEVER.search(query, k=8)

    llm = _build_llm().bind_tools(top_tools)
    system_message = SystemMessage(
        content=[
            {
                "type": "text",
                "text": render_prompt("system"),
                "cache_control": {"type": "ephemeral"},
            }
        ]
    )
    response = llm.invoke([system_message, *messages])

    return {
        "messages": [response],
        "iterations": iterations + 1,
        "metadata": {**state.get("metadata", {}), "tool_count": len(top_tools)},
    }


tool_node = ToolNode(_ALL_TOOLS)


def should_continue(state: AgentState) -> str:
    if state.get("cancelled"):
        return "end"
    if state.get("iterations", 0) >= MAX_ITERATIONS:
        return "end"
    last = state["messages"][-1] if state["messages"] else None
    if isinstance(last, AIMessage) and last.tool_calls:
        return "tools"
    return "end"


def _extract_query(messages: list[BaseMessage]) -> str:
    for msg in reversed(messages):
        if getattr(msg, "type", None) == "human":
            content = msg.content
            return content if isinstance(content, str) else str(content)
    return ""
