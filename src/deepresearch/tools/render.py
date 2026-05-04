from __future__ import annotations

import json

from langchain_core.tools import tool

_SENTINEL = "_render::"


@tool
def render_card(title: str, content: str, metadata: dict | None = None) -> str:
    """Display a formatted info card with a title and content body.
    Use for presenting a single key finding, summary, or highlighted fact.
    Optionally include metadata key-value pairs for extra context."""
    payload = {"type": "card", "title": title, "content": content, "metadata": metadata or {}}
    return f"{_SENTINEL}card\n{json.dumps(payload)}"


@tool
def render_table(title: str, headers: list[str], rows: list[list]) -> str:
    """Display structured data as a formatted ASCII table with headers and rows.
    Use for comparing multiple items, financial data, or any tabular results.
    Cell values are automatically stringified."""
    str_rows = [[str(cell) for cell in row] for row in rows]
    payload = {"type": "table", "title": title, "headers": headers, "rows": str_rows}
    return f"{_SENTINEL}table\n{json.dumps(payload)}"


@tool
def render_chart(title: str, chart_type: str, labels: list[str], values: list[float]) -> str:
    """Display a simple ASCII bar or line chart for visualizing data.
    chart_type must be 'bar' or 'line'. Use for time-series, comparisons, or distributions.
    labels and values must have the same length."""
    if chart_type not in ("bar", "line"):
        return f"[render_chart error: chart_type must be 'bar' or 'line', got '{chart_type}']"
    if len(labels) != len(values):
        return "[render_chart error: labels and values must have the same length]"
    payload = {"type": "chart", "title": title, "chart_type": chart_type,
               "labels": labels, "values": values}
    return f"{_SENTINEL}chart\n{json.dumps(payload)}"


@tool
def render_qa(question: str, answer: str, sources: list[str] | None = None) -> str:
    """Display a question-answer pair with optional source citations.
    Use this to present the final answer to the user's research question.
    sources should be a list of URLs or reference strings."""
    payload = {"type": "qa", "question": question, "answer": answer, "sources": sources or []}
    return f"{_SENTINEL}qa\n{json.dumps(payload)}"


@tool
def render_timeline(title: str, events: list[dict]) -> str:
    """Display a chronological timeline of events.
    events is a list of {"date": "YYYY-MM-DD or any ordered string", "label": "..."}.
    Use for 'history of X' queries, product launch sequences, or any time-ordered narrative."""
    if not isinstance(events, list) or not events:
        return "[render_timeline error: events must be a non-empty list of {date, label} dicts]"
    cleaned = [{"date": str(e.get("date", "")), "label": str(e.get("label", ""))} for e in events]
    payload = {"type": "timeline", "title": title, "events": cleaned}
    return f"{_SENTINEL}timeline\n{json.dumps(payload)}"


@tool
def render_tree(title: str, root: dict) -> str:
    """Display hierarchical data as an indented tree.
    root is a {"name": "...", "children": [{"name": "...", "children": [...]}, ...]} dict.
    Use for taxonomies, org charts, file structures, or any nested categorisation."""
    if not isinstance(root, dict) or "name" not in root:
        return "[render_tree error: root must be a {name, children?} dict]"
    payload = {"type": "tree", "title": title, "root": root}
    return f"{_SENTINEL}tree\n{json.dumps(payload)}"


ALL_RENDER_TOOLS = [render_card, render_table, render_chart, render_qa, render_timeline, render_tree]

for _t in ALL_RENDER_TOOLS:
    _t.tags = ["render"]
