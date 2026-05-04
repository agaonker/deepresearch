
from deepresearch.streaming.events import parse_render
from deepresearch.streaming.render_cli import maybe_paint
from deepresearch.tools.render import (
    render_card,
    render_chart,
    render_qa,
    render_table,
    render_timeline,
    render_tree,
)


def test_render_card_payload():
    out = render_card.invoke({"title": "T", "content": "Hello", "metadata": {"k": "v"}})
    payload = parse_render(out)
    assert payload is not None
    assert payload.kind == "card"
    assert payload.data["title"] == "T"
    assert payload.data["content"] == "Hello"
    assert payload.data["metadata"] == {"k": "v"}


def test_render_table_stringifies_cells():
    out = render_table.invoke({"title": "Nums", "headers": ["a", "b"], "rows": [[1, 2.5]]})
    payload = parse_render(out)
    assert payload.kind == "table"
    assert payload.data["rows"] == [["1", "2.5"]]


def test_render_chart_validates():
    out = render_chart.invoke({
        "title": "x", "chart_type": "pie", "labels": ["a"], "values": [1.0],
    })
    assert "error" in out
    assert parse_render(out) is None


def test_render_chart_length_mismatch():
    out = render_chart.invoke({
        "title": "x", "chart_type": "bar", "labels": ["a", "b"], "values": [1.0],
    })
    assert "error" in out


def test_render_qa_payload():
    out = render_qa.invoke({"question": "Q?", "answer": "A.", "sources": ["http://x"]})
    payload = parse_render(out)
    assert payload.kind == "qa"
    assert payload.data["sources"] == ["http://x"]


def test_parse_render_rejects_non_sentinel():
    assert parse_render("hello world") is None
    assert parse_render("_render::") is None  # no body


def test_parse_render_rejects_invalid_json():
    assert parse_render("_render::card\nnot-json") is None


def test_maybe_paint_card():
    out = render_card.invoke({"title": "Hi", "content": "world"})
    painted = maybe_paint(out)
    assert painted is not None
    assert "Hi" in painted
    assert "world" in painted


def test_maybe_paint_table():
    out = render_table.invoke({"title": "T", "headers": ["a", "b"], "rows": [["1", "2"]]})
    painted = maybe_paint(out)
    assert "a" in painted and "b" in painted and "1" in painted


def test_maybe_paint_chart_bar():
    out = render_chart.invoke({
        "title": "T", "chart_type": "bar", "labels": ["x", "y"], "values": [1.0, 2.0],
    })
    painted = maybe_paint(out)
    assert "x" in painted and "y" in painted


def test_maybe_paint_qa_with_sources():
    out = render_qa.invoke({"question": "Q?", "answer": "A.", "sources": ["s1", "s2"]})
    painted = maybe_paint(out)
    assert "Q?" in painted and "A." in painted and "s1" in painted


def test_maybe_paint_returns_none_for_plain_text():
    assert maybe_paint("just a regular tool result") is None


def test_render_timeline_payload():
    out = render_timeline.invoke({
        "title": "Apple launches",
        "events": [
            {"date": "2007-06-29", "label": "iPhone"},
            {"date": "2010-04-03", "label": "iPad"},
        ],
    })
    payload = parse_render(out)
    assert payload is not None
    assert payload.kind == "timeline"
    assert payload.data["events"][0]["label"] == "iPhone"


def test_render_timeline_validates_empty():
    out = render_timeline.invoke({"title": "x", "events": []})
    assert "error" in out
    assert parse_render(out) is None


def test_maybe_paint_timeline():
    out = render_timeline.invoke({
        "title": "Releases",
        "events": [{"date": "2020", "label": "v1"}, {"date": "2024", "label": "v2"}],
    })
    painted = maybe_paint(out)
    assert painted is not None
    assert "v1" in painted and "v2" in painted
    assert "Releases" in painted


def test_render_tree_payload():
    out = render_tree.invoke({
        "title": "Org",
        "root": {"name": "CEO", "children": [{"name": "CTO"}, {"name": "CFO"}]},
    })
    payload = parse_render(out)
    assert payload.kind == "tree"
    assert payload.data["root"]["name"] == "CEO"


def test_render_tree_validates_root():
    out = render_tree.invoke({"title": "x", "root": {"no_name_key": True}})
    assert "error" in out
    assert parse_render(out) is None


def test_maybe_paint_tree_indents_children():
    out = render_tree.invoke({
        "title": "T",
        "root": {
            "name": "root",
            "children": [
                {"name": "a", "children": [{"name": "a1"}]},
                {"name": "b"},
            ],
        },
    })
    painted = maybe_paint(out)
    assert "root" in painted
    assert "a1" in painted
    assert "b" in painted
