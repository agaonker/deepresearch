"""Tests for the programmatic scorers — deterministic and LLM-free."""
from __future__ import annotations

from types import SimpleNamespace

from deepresearch.eval.scorers import iterations_used, render_match, tool_recall


def _run(**outputs):
    return SimpleNamespace(outputs=outputs)


def _example(inputs=None, **outputs):
    return SimpleNamespace(inputs=inputs or {}, outputs=outputs)


def test_tool_recall_full_hit():
    run = _run(tools_called=["stock_price", "render_qa"])
    ex = _example(must_include_tools=["stock_price"])
    result = tool_recall(run, ex)
    assert result["key"] == "tool_recall"
    assert result["score"] == 1.0


def test_tool_recall_partial_hit():
    run = _run(tools_called=["arxiv_search", "render_table"])
    ex = _example(must_include_tools=["arxiv_search", "pubmed_search"])
    result = tool_recall(run, ex)
    assert result["score"] == 0.5
    assert "pubmed_search" in result["comment"]


def test_tool_recall_total_miss():
    run = _run(tools_called=["web_search"])
    ex = _example(must_include_tools=["arxiv_search"])
    assert tool_recall(run, ex)["score"] == 0.0


def test_tool_recall_empty_expectation():
    run = _run(tools_called=[])
    ex = _example(must_include_tools=[])
    assert tool_recall(run, ex)["score"] == 1.0


def test_render_match_hit():
    run = _run(final_render="render_table")
    ex = _example(expected_render="render_table")
    assert render_match(run, ex)["score"] == 1.0


def test_render_match_miss():
    run = _run(final_render="render_qa")
    ex = _example(expected_render="render_table")
    assert render_match(run, ex)["score"] == 0.0


def test_render_match_no_render():
    run = _run(final_render=None)
    ex = _example(expected_render="render_qa")
    assert render_match(run, ex)["score"] == 0.0


def test_iterations_used_one():
    run = _run(iterations=1)
    ex = _example()
    assert iterations_used(run, ex)["score"] == 1.0


def test_iterations_used_max():
    run = _run(iterations=12)
    ex = _example()
    assert iterations_used(run, ex)["score"] == 0.0


def test_iterations_used_zero():
    run = _run(iterations=0)
    ex = _example()
    result = iterations_used(run, ex)
    assert result["score"] == 0.0
    assert "no iterations" in result["comment"]


def test_iterations_used_midrange_monotonic():
    run_a = _run(iterations=3)
    run_b = _run(iterations=6)
    ex = _example()
    assert iterations_used(run_a, ex)["score"] > iterations_used(run_b, ex)["score"]
