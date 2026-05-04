from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from deepresearch.graph.nodes import agent_node, should_continue, tool_node
from deepresearch.graph.state import AgentState


def build_graph(checkpointer: MemorySaver | None = None) -> CompiledStateGraph:
    builder = StateGraph(AgentState)
    builder.add_node("agent", agent_node)
    builder.add_node("tools", tool_node)

    builder.add_edge(START, "agent")
    builder.add_conditional_edges(
        "agent",
        should_continue,
        {"tools": "tools", "end": END},
    )
    builder.add_edge("tools", "agent")

    return builder.compile(checkpointer=checkpointer or MemorySaver())


graph = build_graph()
