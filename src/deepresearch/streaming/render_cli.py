from __future__ import annotations

from deepresearch.streaming.events import RenderPayload, parse_render
from deepresearch.streaming.render_tokens import Box, Color, pad, rpad, style


def _wrap(text: str, width: int) -> list[str]:
    out: list[str] = []
    for line in str(text).splitlines() or [""]:
        while len(line) > width:
            out.append(line[:width])
            line = line[width:]
        out.append(line)
    return out


def _is_numeric_col(rows: list[list[str]], col: int) -> bool:
    if not rows:
        return False
    for row in rows:
        if col >= len(row):
            continue
        cell = row[col].strip()
        if not cell:
            continue
        # Strip common numeric decorators ($, %, /, comma, period, sign)
        bare = cell.lstrip("$-+").rstrip("%").replace(",", "").replace(".", "")
        if "/" in bare:
            bare = bare.replace("/", "")
        if not bare.isdigit():
            return False
    return True


def _paint_card(d: dict) -> str:
    title = d.get("title", "")
    content = d.get("content", "")
    metadata = d.get("metadata") or {}
    heavy = bool(d.get("heavy"))

    longest_line = max((len(line) for line in content.splitlines()), default=0)
    width = max(40, min(80, max(len(title), longest_line) + 4))

    if heavy:
        tl, tr, bl, br, h, v = Box.HTL, Box.HTR, Box.HBL, Box.HBR, Box.HH, Box.HV
        sep_l, sep_r = Box.HLT, Box.HRT
    else:
        tl, tr, bl, br, h, v = Box.TL, Box.TR, Box.BL, Box.BR, Box.H, Box.V
        sep_l, sep_r = Box.LT, Box.RT

    top = tl + h * (width - 2) + tr
    bot = bl + h * (width - 2) + br
    sep = sep_l + h * (width - 2) + sep_r

    title_styled = style(title, Color.ACCENT, Color.BOLD)
    lines = [top, f"{v} {pad(title_styled, width - 4)} {v}", sep]
    for line in _wrap(content, width - 4):
        lines.append(f"{v} {pad(line, width - 4)} {v}")
    if metadata:
        lines.append(sep)
        for k, val in metadata.items():
            meta_line = f"{style(str(k), Color.MUTED)}: {val}"
            lines.append(f"{v} {pad(meta_line, width - 4)} {v}")
    lines.append(bot)
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

    numeric_cols = {i for i in range(len(widths)) if _is_numeric_col(rows, i)}

    def hline(left: str, mid: str, right: str) -> str:
        return left + mid.join(Box.H * (w + 2) for w in widths) + right

    top = hline(Box.TL, Box.TT, Box.TR)
    middle = hline(Box.LT, Box.CROSS, Box.RT)
    bottom = hline(Box.BL, Box.BT, Box.BR)

    out: list[str] = []
    title = d.get("title")
    if title:
        out.append(style(str(title), Color.ACCENT, Color.BOLD))
    out.append(top)
    header_cells = [
        style(h.ljust(w), Color.ACCENT, Color.BOLD) for h, w in zip(headers, widths, strict=True)
    ]
    out.append(Box.V + " " + (" " + Box.V + " ").join(header_cells) + " " + Box.V)
    out.append(middle)
    for row in rows:
        cells: list[str] = []
        for i, w in enumerate(widths):
            cell = row[i] if i < len(row) else ""
            if i in numeric_cols:
                cells.append(style(rpad(cell, w), Color.BOLD))
            else:
                cells.append(pad(cell, w))
        out.append(Box.V + " " + (" " + Box.V + " ").join(cells) + " " + Box.V)
    out.append(bottom)
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
    out: list[str] = []
    if title:
        out.append(style(str(title), Color.ACCENT, Color.BOLD))
    char = Box.BLOCK if chart_type == "bar" else Box.DOT
    for label, value in zip(labels, values, strict=True):
        n = int(round((value / max_v) * bar_w))
        bar = style(char * max(n, 0), Color.ACCENT)
        out.append(
            f"{style(label.ljust(label_w), Color.MUTED)} "
            f"{style(Box.V, Color.MUTED)} {bar} {style(f'{value:g}', Color.BOLD)}"
        )
    return "\n".join(out)


def _paint_qa(d: dict) -> str:
    q = d.get("question", "")
    a = d.get("answer", "")
    sources = d.get("sources") or []
    q_mark = style("Q", Color.ACCENT, Color.BOLD)
    a_mark = style("A", Color.ACCENT, Color.BOLD)
    out = [f"{q_mark}  {q}", "", f"{a_mark}  {a}"]
    if sources:
        out.append("")
        out.append(style("sources", Color.MUTED))
        for s in sources:
            out.append(style(f"  · {s}", Color.MUTED))
    return "\n".join(out)


def _paint_timeline(d: dict) -> str:
    title = d.get("title", "")
    events = d.get("events", [])
    if not events:
        return f"[empty timeline: {title}]"
    date_w = max(len(str(e.get("date", ""))) for e in events)
    out: list[str] = []
    if title:
        out.append(style(str(title), Color.ACCENT, Color.BOLD))
    for i, e in enumerate(events):
        connector = Box.BRANCH if i < len(events) - 1 else Box.LAST
        date = style(str(e.get("date", "")).ljust(date_w), Color.ACCENT)
        label = str(e.get("label", ""))
        out.append(f"{style(connector, Color.MUTED)} {date}  {label}")
    return "\n".join(out)


def _paint_tree(d: dict) -> str:
    root = d.get("root") or {}
    title = d.get("title", "")
    out: list[str] = []
    if title:
        out.append(style(str(title), Color.ACCENT, Color.BOLD))

    def walk(node: dict, prefix: str = "", is_last: bool = True, is_root: bool = True) -> None:
        name = node.get("name", "")
        if is_root:
            out.append(style(name, Color.BOLD))
            child_prefix = ""
        else:
            connector = Box.LAST + " " if is_last else Box.BRANCH + " "
            out.append(f"{style(prefix + connector, Color.MUTED)}{name}")
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
