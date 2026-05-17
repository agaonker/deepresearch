from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, BaseMessage
from langgraph.prebuilt import ToolNode

from deepresearch.graph.state import AgentState
from deepresearch.llm import build_chat, build_system_message, prepare_tools_for_caching, resolve
from deepresearch.prompts import render_prompt
from deepresearch.tools.catalog import ALL_DATA_TOOLS
from deepresearch.tools.render import ALL_RENDER_TOOLS
from deepresearch.tools.retriever import ToolRetriever

MAX_ITERATIONS = 12

_ALL_TOOLS = ALL_DATA_TOOLS + ALL_RENDER_TOOLS
_RETRIEVER = ToolRetriever(_ALL_TOOLS)


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

    source = resolve("agent")
    # `prepare_tools_for_caching` wraps the last tool with cache_control:ephemeral
    # for Anthropic providers (no-op elsewhere). Currently a no-op at the wire
    # level too: langchain-anthropic 1.4.2 strips cache_control during request
    # serialization. Keeping the call here so caching auto-activates when that
    # passthrough is fixed upstream. See docs/prompt-caching.md.
    cacheable_tools = prepare_tools_for_caching(top_tools, source)
    llm = build_chat(source).bind_tools(cacheable_tools)
    system_message = build_system_message(render_prompt("system"), source)
    response = llm.invoke([system_message, *messages])

    return {
        "messages": [response],
        "iterations": iterations + 1,
        "metadata": {
            **state.get("metadata", {}),
            "tool_count": len(top_tools),
            "llm_source": source.name,
        },
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
