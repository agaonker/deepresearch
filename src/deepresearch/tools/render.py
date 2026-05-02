import json
from langchain_core.tools import tool

_SENTINEL = "_render::"


@tool(tags=["render"])
def render_card(title: str, content: str, metadata: dict = None) -> str:
    """Display a formatted info card with a title and content body.
    Use for presenting a single key finding, summary, or highlighted fact.
    Optionally include metadata key-value pairs for extra context."""
    payload = {"type": "card", "title": title, "content": content, "metadata": metadata or {}}
    return f"{_SENTINEL}card\n{json.dumps(payload)}"


@tool(tags=["render"])
def render_table(title: str, headers: list[str], rows: list[list]) -> str:
    """Display structured data as a formatted ASCII table with headers and rows.
    Use for comparing multiple items, financial data, or any tabular results.
    Cell values are automatically stringified."""
    str_rows = [[str(cell) for cell in row] for row in rows]
    payload = {"type": "table", "title": title, "headers": headers, "rows": str_rows}
    return f"{_SENTINEL}table\n{json.dumps(payload)}"


@tool(tags=["render"])
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


@tool(tags=["render"])
def render_qa(question: str, answer: str, sources: list[str] = None) -> str:
    """Display a question-answer pair with optional source citations.
    Use this to present the final answer to the user's research question.
    sources should be a list of URLs or reference strings."""
    payload = {"type": "qa", "question": question, "answer": answer, "sources": sources or []}
    return f"{_SENTINEL}qa\n{json.dumps(payload)}"


ALL_RENDER_TOOLS = [render_card, render_table, render_chart, render_qa]
