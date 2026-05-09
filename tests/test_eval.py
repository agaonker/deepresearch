"""Eval harness for the BM25 retriever against the golden dataset.

Asserts that for every golden query, the tools we expect the agent to reach
for appear in the BM25 top-K selection. No LLM is called — these tests stay
fast, deterministic, and free.
"""
from __future__ import annotations

import pytest

from deepresearch.eval import GOLDEN_QUERIES, EvalCase
from deepresearch.tools.catalog import ALL_DATA_TOOLS
from deepresearch.tools.render import ALL_RENDER_TOOLS
from deepresearch.tools.retriever import ToolRetriever

K = 8


@pytest.fixture(scope="module")
def retriever() -> ToolRetriever:
    return ToolRetriever(ALL_DATA_TOOLS + ALL_RENDER_TOOLS)


def test_dataset_size():
    assert len(GOLDEN_QUERIES) == 50


def test_dataset_names_unique():
    names = [c.name for c in GOLDEN_QUERIES]
    assert len(names) == len(set(names))


def test_expected_render_is_known_render_tool():
    render_names = {t.name for t in ALL_RENDER_TOOLS}
    for case in GOLDEN_QUERIES:
        assert case.expected_render in render_names, (
            f"{case.name}: expected_render {case.expected_render!r} not in catalog"
        )


def test_must_include_tools_exist_in_catalog():
    catalog = {t.name for t in ALL_DATA_TOOLS + ALL_RENDER_TOOLS}
    for case in GOLDEN_QUERIES:
        for tool_name in case.must_include_tools:
            assert tool_name in catalog, (
                f"{case.name}: must_include tool {tool_name!r} not in catalog"
            )


@pytest.mark.parametrize("case", GOLDEN_QUERIES, ids=lambda c: c.name)
def test_bm25_includes_expected_tools(retriever: ToolRetriever, case: EvalCase):
    selected = {t.name for t in retriever.search(case.query, k=K)}
    missing = [name for name in case.must_include_tools if name not in selected]
    assert not missing, (
        f"{case.name}: BM25 top-{K} missing {missing} for query {case.query!r}. "
        f"Selected: {sorted(selected)}"
    )


def test_overall_recall_is_perfect(retriever: ToolRetriever):
    """Aggregate sanity check — useful as a single CI-friendly summary metric."""
    hits = 0
    total = 0
    for case in GOLDEN_QUERIES:
        selected = {t.name for t in retriever.search(case.query, k=K)}
        for name in case.must_include_tools:
            total += 1
            if name in selected:
                hits += 1
    assert total > 0
    recall = hits / total
    assert recall == 1.0, f"recall={recall:.2f} ({hits}/{total}) — see per-case failures"
