from __future__ import annotations

import os
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from langgraph.prebuilt import ToolNode

from deepresearch.graph.state import AgentState
from deepresearch.tools.catalog import ALL_DATA_TOOLS
from deepresearch.tools.render import ALL_RENDER_TOOLS
from deepresearch.tools.retriever import ToolRetriever

MAX_ITERATIONS = 12

_SYSTEM_PROMPT = """You are DeepResearch, a tool-using research agent.

Workflow:
1. Pick the smallest set of data tools that answer the user's question. Run independent tools in parallel.
2. After gathering evidence, ALWAYS finish by calling exactly one render tool (render_qa, render_card, render_table, or render_chart) so the user sees a structured result. render_qa is the default for cited answers.
3. Cite sources (URLs) when you have them.
4. Keep reasoning concise. Do not call tools you don't need.
"""

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
    return ChatAnthropic(  # type: ignore[call-arg]
        model=os.getenv("ANTHROPIC_MODEL", "claude-opus-4-5"),
        streaming=True,
        max_tokens=4096,
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
    response = llm.invoke([SystemMessage(content=_SYSTEM_PROMPT), *messages])

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
