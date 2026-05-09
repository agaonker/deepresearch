from __future__ import annotations

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from deepresearch.graph.nodes import agent_node, should_continue, tool_node
from deepresearch.graph.state import AgentState


def build_graph(checkpointer: BaseCheckpointSaver | None = None) -> CompiledStateGraph:
    """Compile the agent graph. Pass `checkpointer=None` to skip persistence.

    The CLI passes `MemorySaver()` so REPL state survives across queries in
    one process. The module-level `graph` symbol below is for `langgraph dev`
    and Studio, which provide their own persistence — passing a custom
    checkpointer there raises a startup error.
    """
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

    return builder.compile(checkpointer=checkpointer)


graph = build_graph()
