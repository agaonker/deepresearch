from deepresearch.tools.catalog import ALL_DATA_TOOLS
from deepresearch.tools.render import ALL_RENDER_TOOLS
from deepresearch.tools.retriever import ALWAYS_INCLUDE, ToolRetriever


def _build():
    return ToolRetriever(ALL_DATA_TOOLS + ALL_RENDER_TOOLS)


def test_search_returns_at_least_k_plus_always_include():
    r = _build()
    selected = r.search("nvidia stock price", k=4)
    names = {t.name for t in selected}
    assert len(selected) >= 4
    for must in ALWAYS_INCLUDE:
        assert must in names


def test_search_ranks_relevant_tool_first():
    r = _build()
    selected = r.search("arxiv paper on transformers", k=8)
    names = [t.name for t in selected]
    assert "arxiv_search" in names[:4]


def test_search_handles_empty_query():
    r = _build()
    selected = r.search("", k=3)
    assert len(selected) >= 3


def test_explain_returns_scored_list():
    r = _build()
    ranked = r.explain("github repo openai", k=5)
    assert len(ranked) == 5
    assert ranked[0].score >= ranked[-1].score
    names = [x.name for x in ranked]
    assert "get_github_repo" in names


def test_search_no_duplicates():
    r = _build()
    selected = r.search("calculate compound interest", k=8)
    names = [t.name for t in selected]
    assert len(names) == len(set(names))


def test_always_include_render_tools_present():
    r = _build()
    selected = r.search("anything", k=2)
    names = {t.name for t in selected}
    for render in ("render_card", "render_table", "render_chart", "render_qa"):
        assert render in names
