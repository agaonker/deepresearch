from __future__ import annotations

from deepresearch.streaming.events import RenderPayload, parse_render


def _wrap(text: str, width: int) -> list[str]:
    out: list[str] = []
    for line in str(text).splitlines() or [""]:
        while len(line) > width:
            out.append(line[:width])
            line = line[width:]
        out.append(line)
    return out


def _paint_card(d: dict) -> str:
    title = d.get("title", "")
    content = d.get("content", "")
    metadata = d.get("metadata") or {}
    longest_line = max((len(line) for line in content.splitlines()), default=0)
    width = max(40, min(80, max(len(title), longest_line) + 4))
    bar = "+" + "-" * (width - 2) + "+"
    lines = [bar, f"| {title.ljust(width - 4)} |", bar]
    for line in _wrap(content, width - 4):
        lines.append(f"| {line.ljust(width - 4)} |")
    if metadata:
        lines.append(bar)
        for k, v in metadata.items():
            lines.append(f"| {f'{k}: {v}'.ljust(width - 4)} |")
    lines.append(bar)
    return "\n".join(lines)


def _paint_table(d: dict) -> str:
    headers = [str(h) for h in d.get("headers", [])]
    rows = [[str(c) for c in r] for r in d.get("rows", [])]
    if not headers:
        return f"[empty table: {d.get('title', '')}]"
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row[: len(widths)]):
            widths[i] = max(widths[i], len(cell))
    sep = "+" + "+".join("-" * (w + 2) for w in widths) + "+"
    out = []
    title = d.get("title")
    if title:
        out.append(title)
    out.append(sep)
    out.append("| " + " | ".join(h.ljust(w) for h, w in zip(headers, widths, strict=True)) + " |")
    out.append(sep)
    for row in rows:
        cells = [(row[i] if i < len(row) else "").ljust(widths[i]) for i in range(len(widths))]
        out.append("| " + " | ".join(cells) + " |")
    out.append(sep)
    return "\n".join(out)


def _paint_chart(d: dict) -> str:
    labels = [str(label) for label in d.get("labels", [])]
    values = [float(v) for v in d.get("values", [])]
    title = d.get("title", "")
    chart_type = d.get("chart_type", "bar")
    if not values:
        return f"[empty chart: {title}]"
    label_w = max(len(label) for label in labels)
    max_v = max(abs(v) for v in values) or 1.0
    bar_w = 40
    out = [title] if title else []
    char = "#" if chart_type == "bar" else "*"
    for label, value in zip(labels, values, strict=True):
        n = int(round((value / max_v) * bar_w))
        out.append(f"{label.ljust(label_w)} | {char * max(n, 0)} {value:g}")
    return "\n".join(out)


def _paint_qa(d: dict) -> str:
    q = d.get("question", "")
    a = d.get("answer", "")
    sources = d.get("sources") or []
    out = [f"Q: {q}", "", f"A: {a}"]
    if sources:
        out.append("")
        out.append("Sources:")
        for s in sources:
            out.append(f"  - {s}")
    return "\n".join(out)


def _paint_timeline(d: dict) -> str:
    title = d.get("title", "")
    events = d.get("events", [])
    if not events:
        return f"[empty timeline: {title}]"
    date_w = max(len(str(e.get("date", ""))) for e in events)
    out = [title] if title else []
    for i, e in enumerate(events):
        connector = "├──" if i < len(events) - 1 else "└──"
        out.append(f"{connector} {str(e.get('date', '')).ljust(date_w)}  {e.get('label', '')}")
    return "\n".join(out)


def _paint_tree(d: dict) -> str:
    root = d.get("root") or {}
    title = d.get("title", "")
    out = [title] if title else []

    def walk(node: dict, prefix: str = "", is_last: bool = True, is_root: bool = True) -> None:
        name = node.get("name", "")
        if is_root:
            out.append(name)
            child_prefix = ""
        else:
            connector = "└── " if is_last else "├── "
            out.append(f"{prefix}{connector}{name}")
            child_prefix = prefix + ("    " if is_last else "│   ")
        children = node.get("children") or []
        for i, child in enumerate(children):
            walk(child, child_prefix, i == len(children) - 1, is_root=False)

    walk(root)
    return "\n".join(out)


_PAINTERS = {
    "card": _paint_card,
    "table": _paint_table,
    "chart": _paint_chart,
    "qa": _paint_qa,
    "timeline": _paint_timeline,
    "tree": _paint_tree,
}


def paint(payload: RenderPayload) -> str:
    painter = _PAINTERS.get(payload.kind)
    if painter is None:
        return f"[unknown render kind: {payload.kind}]"
    return painter(payload.data)


def maybe_paint(text: str) -> str | None:
    payload = parse_render(text)
    if payload is None:
        return None
    return paint(payload)
