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
    # ── Finance & markets (8) ──────────────────────────────────────────────
    EvalCase(
        name="stock_quote_nvda",
        query="What is the current stock price of NVDA?",
        must_include_tools=("stock_price",),
        expected_render="render_qa",
        tags=("finance",),
    ),
    EvalCase(
        name="stock_quote_aapl",
        query="What is the current stock price of AAPL?",
        must_include_tools=("stock_price",),
        expected_render="render_qa",
        tags=("finance",),
    ),
    EvalCase(
        name="stock_compare_nvda_amd",
        query="Compare NVDA vs AMD annual revenue and net income",
        must_include_tools=("stock_financials",),
        expected_render="render_table",
        tags=("finance", "comparison"),
    ),
    EvalCase(
        name="stock_financials_msft",
        query="Show Microsoft's annual revenue, net income, and EBITDA financial statements",
        must_include_tools=("stock_financials",),
        expected_render="render_table",
        tags=("finance",),
    ),
    EvalCase(
        name="stock_earnings_aapl",
        query="Show recent earnings history and upcoming earnings dates for AAPL",
        must_include_tools=("stock_earnings",),
        expected_render="render_table",
        tags=("finance",),
    ),
    EvalCase(
        name="stock_news_tsla",
        query="Recent news headlines for Tesla stock TSLA",
        must_include_tools=("stock_news",),
        expected_render="render_table",
        tags=("finance", "news"),
    ),
    EvalCase(
        name="market_summary",
        query="Summary of major US stock market indices: S&P 500, Dow Jones, NASDAQ",
        must_include_tools=("market_summary",),
        expected_render="render_table",
        tags=("finance",),
    ),
    EvalCase(
        name="crypto_price_btc",
        query="What is the current bitcoin cryptocurrency price in USD via CoinGecko?",
        must_include_tools=("coingecko_price",),
        expected_render="render_qa",
        tags=("finance", "crypto"),
    ),

    # ── Academic literature (5) ────────────────────────────────────────────
    EvalCase(
        name="arxiv_attention",
        query="Find arxiv papers on transformer attention mechanisms",
        must_include_tools=("arxiv_search",),
        expected_render="render_table",
        tags=("academic",),
    ),
    EvalCase(
        name="arxiv_diffusion",
        query="Search arxiv for recent papers on diffusion models",
        must_include_tools=("arxiv_search",),
        expected_render="render_table",
        tags=("academic",),
    ),
    EvalCase(
        name="arxiv_paper_details",
        query="Get full details for the arxiv paper with id 2301.07041",
        must_include_tools=("arxiv_paper_details",),
        expected_render="render_card",
        tags=("academic",),
    ),
    EvalCase(
        name="pubmed_crispr",
        query="Find PubMed papers on CRISPR gene editing in cancer",
        must_include_tools=("pubmed_search",),
        expected_render="render_table",
        tags=("academic", "medical"),
    ),
    EvalCase(
        name="crossref_climate",
        query="Search Crossref for academic papers on climate change adaptation",
        must_include_tools=("crossref_search",),
        expected_render="render_table",
        tags=("academic",),
    ),

    # ── Encyclopedic & factual (5) ─────────────────────────────────────────
    EvalCase(
        name="wiki_summary_french_revolution",
        query="Summarize the Wikipedia article on the French Revolution",
        must_include_tools=("wiki_summary",),
        expected_render="render_card",
        tags=("encyclopedic",),
    ),
    EvalCase(
        name="wiki_summary_einstein",
        query="Summarize the Wikipedia entry on Albert Einstein",
        must_include_tools=("wiki_summary",),
        expected_render="render_card",
        tags=("encyclopedic", "biography"),
    ),
    EvalCase(
        name="wiki_search_quantum",
        query="Search Wikipedia for articles about quantum computing",
        must_include_tools=("wiki_search",),
        expected_render="render_table",
        tags=("encyclopedic",),
    ),
    EvalCase(
        name="wikidata_sparql",
        query="Run a SPARQL query against Wikidata for capital cities of EU countries",
        must_include_tools=("wikidata_query",),
        expected_render="render_table",
        tags=("encyclopedic", "structured"),
    ),
    EvalCase(
        name="country_info_japan",
        query="Tell me about the country Japan: capital, population, region, languages",
        must_include_tools=("get_country_info",),
        expected_render="render_card",
        tags=("encyclopedic", "geo"),
    ),

    # ── Web search & news (5) ──────────────────────────────────────────────
    EvalCase(
        name="web_search_ai_regulation",
        query="Search the web for recent updates on AI regulation policy",
        must_include_tools=("web_search",),
        expected_render="render_table",
        tags=("web",),
    ),
    EvalCase(
        name="ddg_news_openai",
        query="Latest news articles about OpenAI from DuckDuckGo News",
        must_include_tools=("ddg_news",),
        expected_render="render_table",
        tags=("web", "news"),
    ),
    EvalCase(
        name="ddg_image_eiffel",
        query="Find images of the Eiffel Tower with DuckDuckGo image search",
        must_include_tools=("ddg_image_search",),
        expected_render="render_table",
        tags=("web", "images"),
    ),
    EvalCase(
        name="hackernews_langchain",
        query="Top Hacker News stories about LangChain and LangGraph",
        must_include_tools=("hackernews_search",),
        expected_render="render_table",
        tags=("web", "news"),
    ),
    EvalCase(
        name="reddit_ml",
        query="Search Reddit posts about machine learning and deep learning",
        must_include_tools=("reddit_search",),
        expected_render="render_table",
        tags=("web", "social"),
    ),

    # ── Geography & weather (4) ────────────────────────────────────────────
    EvalCase(
        name="weather_now_tokyo",
        query="What is the current weather in Tokyo right now?",
        must_include_tools=("weather_now",),
        expected_render="render_qa",
        tags=("geo", "weather"),
    ),
    EvalCase(
        name="weather_now_paris",
        query="What's the weather in Paris today?",
        must_include_tools=("weather_now",),
        expected_render="render_qa",
        tags=("geo", "weather"),
    ),
    EvalCase(
        name="weather_forecast_open_meteo",
        query="Get a 7 day weather forecast for latitude 35.6 longitude 139.7 from Open-Meteo",
        must_include_tools=("open_meteo_forecast",),
        expected_render="render_chart",
        tags=("geo", "weather"),
    ),
    EvalCase(
        name="osm_geocode_address",
        query="Geocode the address 1600 Amphitheatre Parkway Mountain View via OpenStreetMap",
        must_include_tools=("osm_geocode",),
        expected_render="render_qa",
        tags=("geo",),
    ),

    # ── Macro & demographics (3) ───────────────────────────────────────────
    EvalCase(
        name="worldbank_gdp_india",
        query="Show India's GDP from the World Bank indicator NY.GDP.MKTP.CD",
        must_include_tools=("world_bank_indicator",),
        expected_render="render_chart",
        tags=("macro",),
    ),
    EvalCase(
        name="worldbank_population_china",
        query="Show China's population over time from the World Bank indicator SP.POP.TOTL",
        must_include_tools=("world_bank_indicator",),
        expected_render="render_chart",
        tags=("macro", "demographics"),
    ),
    EvalCase(
        name="worldbank_life_expectancy",
        query="Get the World Bank life expectancy indicator for Brazil",
        must_include_tools=("world_bank_indicator",),
        expected_render="render_chart",
        tags=("macro", "demographics"),
    ),

    # ── Web fetch & document extraction (3) ────────────────────────────────
    EvalCase(
        name="fetch_url_page",
        query="Fetch the readable text content of the web page at https://example.com",
        must_include_tools=("fetch_url",),
        expected_render="render_card",
        tags=("web", "fetch"),
    ),
    EvalCase(
        name="fetch_url_headers",
        query="Get the HTTP response headers for https://github.com without downloading the body",
        must_include_tools=("fetch_url_headers",),
        expected_render="render_table",
        tags=("web", "fetch"),
    ),
    EvalCase(
        name="pdf_extract",
        query="Extract text from the PDF file at https://arxiv.org/pdf/2301.07041",
        must_include_tools=("pdf_extract_text",),
        expected_render="render_card",
        tags=("document",),
    ),

    # ── Multimedia (1) ─────────────────────────────────────────────────────
    EvalCase(
        name="youtube_transcript",
        query="Fetch the YouTube transcript for video https://youtu.be/dQw4w9WgXcQ",
        must_include_tools=("youtube_transcript",),
        expected_render="render_card",
        tags=("multimedia",),
    ),

    # ── Developer ecosystem (3) ────────────────────────────────────────────
    EvalCase(
        name="github_repo_langgraph",
        query="Get GitHub repository info for langchain-ai/langgraph: stars, forks, license",
        must_include_tools=("get_github_repo",),
        expected_render="render_card",
        tags=("dev",),
    ),
    EvalCase(
        name="github_repo_pytorch",
        query="Get the GitHub repository details for pytorch/pytorch",
        must_include_tools=("get_github_repo",),
        expected_render="render_card",
        tags=("dev",),
    ),
    EvalCase(
        name="pypi_requests",
        query="Look up the requests Python package on PyPI: version, summary, license",
        must_include_tools=("search_pypi",),
        expected_render="render_card",
        tags=("dev",),
    ),

    # ── Utilities (10) ─────────────────────────────────────────────────────
    EvalCase(
        name="calculate_basic",
        query="Calculate 12345 multiplied by 6789",
        must_include_tools=("calculate",),
        expected_render="render_qa",
        tags=("utility", "math"),
    ),
    EvalCase(
        name="calculate_compound",
        query="Calculate the compound interest on 1000 at 5 percent for 10 years",
        must_include_tools=("calculate",),
        expected_render="render_qa",
        tags=("utility", "math"),
    ),
    EvalCase(
        name="datetime_singapore",
        query="What is the current date and time in the Asia/Singapore timezone?",
        must_include_tools=("get_current_datetime",),
        expected_render="render_qa",
        tags=("utility", "time"),
    ),
    EvalCase(
        name="convert_units_km_miles",
        query="Convert 50 kilometers to miles using a unit converter",
        must_include_tools=("convert_units",),
        expected_render="render_qa",
        tags=("utility",),
    ),
    EvalCase(
        name="convert_units_temperature",
        query="Convert 100 degrees fahrenheit to celsius using unit conversion",
        must_include_tools=("convert_units",),
        expected_render="render_qa",
        tags=("utility",),
    ),
    EvalCase(
        name="currency_usd_eur",
        query="Convert 100 USD to EUR using current currency exchange rates",
        must_include_tools=("currency_convert",),
        expected_render="render_qa",
        tags=("utility", "finance"),
    ),
    EvalCase(
        name="currency_inr_usd",
        query="Convert 5000 INR to USD via live currency exchange rates",
        must_include_tools=("currency_convert",),
        expected_render="render_qa",
        tags=("utility", "finance"),
    ),
    EvalCase(
        name="define_word_ephemeral",
        query="Define the English dictionary word ephemeral",
        must_include_tools=("define_word",),
        expected_render="render_card",
        tags=("utility", "language"),
    ),
    EvalCase(
        name="define_word_serendipity",
        query="Look up the dictionary definition of serendipity",
        must_include_tools=("define_word",),
        expected_render="render_card",
        tags=("utility", "language"),
    ),
    EvalCase(
        name="public_ip_info",
        query="Show information about my current public IP address: location, ISP, timezone",
        must_include_tools=("get_public_ip_info",),
        expected_render="render_card",
        tags=("utility", "network"),
    ),

    # ── Cross-tool / multi-step (3) ────────────────────────────────────────
    EvalCase(
        name="compare_cities_weather",
        query="Compare current weather in Tokyo and Paris side by side",
        must_include_tools=("weather_now",),
        expected_render="render_table",
        tags=("comparison", "weather"),
    ),
    EvalCase(
        name="finance_research_amd",
        query="Research AMD: current stock price, recent earnings, and recent news",
        must_include_tools=("stock_price", "stock_earnings", "stock_news"),
        expected_render="render_card",
        tags=("finance", "research"),
    ),
    EvalCase(
        name="academic_overview_llms",
        query="Find arxiv papers and PubMed papers on large language models in healthcare",
        must_include_tools=("arxiv_search", "pubmed_search"),
        expected_render="render_table",
        tags=("academic", "research"),
    ),
)
