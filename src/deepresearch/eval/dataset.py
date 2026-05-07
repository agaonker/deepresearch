from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EvalCase:
    """One golden query and the tools an ideal run should reach for.

    `must_include_tools` is asserted against BM25 top-K (no LLM call) so the
    suite stays cost-free and deterministic. `expected_render` documents the
    render tool an ideal run would finish with — render selection is an LLM
    decision at runtime, not a retriever decision, so it's informational.
    """

    name: str
    query: str
    must_include_tools: tuple[str, ...]
    expected_render: str
    tags: tuple[str, ...]


GOLDEN_QUERIES: tuple[EvalCase, ...] = (
    EvalCase(
        name="stock_quote",
        query="What is the current stock price of NVDA?",
        must_include_tools=("stock_price",),
        expected_render="render_qa",
        tags=("finance",),
    ),
    EvalCase(
        name="stock_compare",
        query="Compare NVDA vs AMD annual revenue and net income",
        must_include_tools=("stock_financials",),
        expected_render="render_table",
        tags=("finance", "comparison"),
    ),
    EvalCase(
        name="arxiv_lit",
        query="Find arxiv papers on transformer attention mechanisms",
        must_include_tools=("arxiv_search",),
        expected_render="render_table",
        tags=("academic",),
    ),
    EvalCase(
        name="wiki_topic",
        query="Summarize the Wikipedia article on the French Revolution",
        must_include_tools=("wiki_summary",),
        expected_render="render_card",
        tags=("encyclopedic",),
    ),
    EvalCase(
        name="weather_now",
        query="What is the weather in Tokyo right now?",
        must_include_tools=("weather_now",),
        expected_render="render_qa",
        tags=("geo", "weather"),
    ),
    EvalCase(
        name="currency_convert",
        query="Convert 100 USD to EUR",
        must_include_tools=("currency_convert",),
        expected_render="render_qa",
        tags=("utility",),
    ),
    EvalCase(
        name="pubmed_lit",
        query="Find PubMed papers on CRISPR gene editing in cancer",
        must_include_tools=("pubmed_search",),
        expected_render="render_table",
        tags=("academic", "medical"),
    ),
    EvalCase(
        name="crypto_price",
        query="What is the current bitcoin price in USD?",
        must_include_tools=("coingecko_price",),
        expected_render="render_qa",
        tags=("finance", "crypto"),
    ),
    EvalCase(
        name="macro_indicator",
        query="Show GDP of India from the World Bank indicators",
        must_include_tools=("world_bank_indicator",),
        expected_render="render_chart",
        tags=("macro", "demographics"),
    ),
    EvalCase(
        name="hackernews_topic",
        query="Top Hacker News stories about LangChain and LangGraph",
        must_include_tools=("hackernews_search",),
        expected_render="render_table",
        tags=("web", "news"),
    ),
)
