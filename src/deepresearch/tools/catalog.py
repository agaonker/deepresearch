import ast
import operator
import os
from datetime import datetime, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import httpx
from langchain_core.tools import tool


# ---------------------------------------------------------------------------
# DuckDuckGo tools (free, no API key)
# ---------------------------------------------------------------------------

@tool
def web_search(query: str, max_results: int = 5) -> str:
    """Search the web using DuckDuckGo. Returns titles, URLs, and snippets for recent web pages.
    Use for general information, news, current events, and research on any topic."""
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        if not results:
            return "[web_search: no results found]"
        return "\n\n".join(
            f"**{r['title']}**\n{r['href']}\n{r['body']}" for r in results
        )
    except Exception as e:
        return f"[web_search error: {e}]"


@tool
def ddg_news(query: str, max_results: int = 5) -> str:
    """Search recent news articles using DuckDuckGo News. Returns headlines, sources, and summaries.
    Use for current events, breaking news, and recent developments on any topic."""
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.news(query, max_results=max_results))
        if not results:
            return "[ddg_news: no results found]"
        return "\n\n".join(
            f"**{r['title']}** — {r.get('source', '')}\n{r['url']}\n{r.get('body', '')}"
            for r in results
        )
    except Exception as e:
        return f"[ddg_news error: {e}]"


@tool
def ddg_image_search(query: str, max_results: int = 5) -> str:
    """Search for images using DuckDuckGo. Returns image metadata including titles and URLs.
    Use to find visual references, logos, diagrams, or illustrations for a topic."""
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.images(query, max_results=max_results))
        if not results:
            return "[ddg_image_search: no results found]"
        return "\n\n".join(
            f"**{r.get('title', 'Image')}**\nImage: {r['image']}\nPage: {r.get('url', '')}"
            for r in results
        )
    except Exception as e:
        return f"[ddg_image_search error: {e}]"


# ---------------------------------------------------------------------------
# Wikipedia tools (free, no API key)
# ---------------------------------------------------------------------------

@tool
def wiki_summary(topic: str, sentences: int = 5) -> str:
    """Get a Wikipedia summary for a topic. Returns a concise encyclopedia-style summary.
    Use for factual background information, definitions, and overviews of people, places, concepts."""
    try:
        import wikipedia
        return wikipedia.summary(topic, sentences=sentences, auto_suggest=True)
    except Exception as e:
        return f"[wiki_summary error: {e}]"


@tool
def wiki_search(query: str, results: int = 5) -> str:
    """Search Wikipedia for articles matching a query. Returns a list of matching article titles.
    Use to discover relevant Wikipedia pages before fetching their summaries."""
    try:
        import wikipedia
        titles = wikipedia.search(query, results=results)
        return "\n".join(f"- {t}" for t in titles) if titles else "[wiki_search: no results]"
    except Exception as e:
        return f"[wiki_search error: {e}]"


# ---------------------------------------------------------------------------
# arXiv tools (free, no API key)
# ---------------------------------------------------------------------------

@tool
def arxiv_search(query: str, max_results: int = 5) -> str:
    """Search arXiv for academic papers, preprints, and scientific research.
    Returns titles, authors, abstracts, and links. Use for academic literature, AI/ML research, physics, math."""
    try:
        import arxiv
        search = arxiv.Search(query=query, max_results=max_results,
                              sort_by=arxiv.SortCriterion.Relevance)
        client = arxiv.Client()
        out = []
        for r in client.results(search):
            authors = ", ".join(a.name for a in r.authors[:3])
            out.append(f"**{r.title}**\nAuthors: {authors}\nID: {r.get_short_id()}\n{r.summary[:300]}...")
        return "\n\n".join(out) if out else "[arxiv_search: no results]"
    except Exception as e:
        return f"[arxiv_search error: {e}]"


@tool
def arxiv_paper_details(arxiv_id: str) -> str:
    """Get full details for a specific arXiv paper by its ID (e.g. '2301.07041').
    Returns title, authors, abstract, publication date, and PDF link."""
    try:
        import arxiv
        client = arxiv.Client()
        search = arxiv.Search(id_list=[arxiv_id])
        results = list(client.results(search))
        if not results:
            return f"[arxiv_paper_details: paper {arxiv_id} not found]"
        r = results[0]
        authors = ", ".join(a.name for a in r.authors)
        return (
            f"**{r.title}**\n"
            f"Authors: {authors}\n"
            f"Published: {r.published.strftime('%Y-%m-%d')}\n"
            f"PDF: {r.pdf_url}\n\n"
            f"{r.summary}"
        )
    except Exception as e:
        return f"[arxiv_paper_details error: {e}]"


# ---------------------------------------------------------------------------
# Yahoo Finance tools via yfinance (free, no API key)
# ---------------------------------------------------------------------------

@tool
def stock_price(ticker: str) -> str:
    """Get the current stock price and key metrics for a ticker symbol (e.g. AAPL, TSLA, NVDA).
    Returns current price, market cap, P/E ratio, 52-week range, and volume."""
    try:
        import yfinance as yf
        t = yf.Ticker(ticker.upper())
        info = t.info
        return (
            f"**{info.get('longName', ticker)}** ({ticker.upper()})\n"
            f"Price: ${info.get('currentPrice', info.get('regularMarketPrice', 'N/A'))}\n"
            f"Market Cap: ${info.get('marketCap', 'N/A'):,}\n"
            f"P/E Ratio: {info.get('trailingPE', 'N/A')}\n"
            f"52W Range: {info.get('fiftyTwoWeekLow', 'N/A')} – {info.get('fiftyTwoWeekHigh', 'N/A')}\n"
            f"Volume: {info.get('volume', 'N/A'):,}"
        )
    except Exception as e:
        return f"[stock_price error: {e}]"


@tool
def stock_financials(ticker: str) -> str:
    """Get annual financial statements for a stock ticker: revenue, net income, EBITDA.
    Use for fundamental analysis and comparing company financials across years."""
    try:
        import yfinance as yf
        t = yf.Ticker(ticker.upper())
        fin = t.financials
        if fin is None or fin.empty:
            return f"[stock_financials: no data for {ticker}]"
        rows = []
        for metric in ["Total Revenue", "Net Income", "EBITDA", "Gross Profit"]:
            if metric in fin.index:
                vals = fin.loc[metric]
                row = [metric] + [f"${v/1e9:.2f}B" if isinstance(v, (int, float)) else "N/A"
                                  for v in vals]
                rows.append(row)
        headers = ["Metric"] + [str(c)[:10] for c in fin.columns]
        lines = ["\t".join(headers)]
        for row in rows:
            lines.append("\t".join(str(x) for x in row))
        return "\n".join(lines)
    except Exception as e:
        return f"[stock_financials error: {e}]"


@tool
def stock_earnings(ticker: str) -> str:
    """Get recent earnings history and upcoming earnings dates for a stock ticker.
    Returns EPS estimates vs actuals and earnings dates."""
    try:
        import yfinance as yf
        t = yf.Ticker(ticker.upper())
        cal = t.calendar
        hist = t.earnings_dates
        out = []
        if cal is not None and not cal.empty:
            out.append(f"Next Earnings: {cal.get('Earnings Date', ['N/A'])[0]}")
        if hist is not None and not hist.empty:
            out.append("\nRecent Earnings:")
            for date, row in hist.head(4).iterrows():
                eps_est = row.get("EPS Estimate", "N/A")
                eps_rep = row.get("Reported EPS", "N/A")
                out.append(f"  {str(date)[:10]}: Est {eps_est} | Actual {eps_rep}")
        return "\n".join(out) if out else f"[stock_earnings: no data for {ticker}]"
    except Exception as e:
        return f"[stock_earnings error: {e}]"


@tool
def stock_news(ticker: str, limit: int = 5) -> str:
    """Get recent news headlines for a stock ticker from Yahoo Finance.
    Returns article titles, publishers, and links."""
    try:
        import yfinance as yf
        t = yf.Ticker(ticker.upper())
        news = t.news or []
        if not news:
            return f"[stock_news: no news for {ticker}]"
        return "\n\n".join(
            f"**{n.get('content', {}).get('title', 'N/A')}**\n"
            f"{n.get('content', {}).get('provider', {}).get('displayName', '')}\n"
            f"{n.get('content', {}).get('canonicalUrl', {}).get('url', '')}"
            for n in news[:limit]
        )
    except Exception as e:
        return f"[stock_news error: {e}]"


@tool
def market_summary() -> str:
    """Get a summary of major US stock market indices: S&P 500, Dow Jones, and NASDAQ.
    Returns current levels, daily change, and percentage change."""
    try:
        import yfinance as yf
        indices = {"S&P 500": "^GSPC", "Dow Jones": "^DJI", "NASDAQ": "^IXIC", "VIX": "^VIX"}
        lines = []
        for name, symbol in indices.items():
            t = yf.Ticker(symbol)
            info = t.info
            price = info.get("regularMarketPrice", "N/A")
            change = info.get("regularMarketChange", "N/A")
            pct = info.get("regularMarketChangePercent", "N/A")
            if isinstance(pct, float):
                pct = f"{pct:.2f}%"
            lines.append(f"{name}: {price} ({change}, {pct})")
        return "\n".join(lines)
    except Exception as e:
        return f"[market_summary error: {e}]"


# ---------------------------------------------------------------------------
# HTTP / web scraping tools (httpx + beautifulsoup4, free)
# ---------------------------------------------------------------------------

@tool
def fetch_url(url: str) -> str:
    """Fetch the text content of a web page by URL. Strips HTML and returns readable text.
    Use to read articles, documentation, or any public web page by its URL."""
    try:
        from bs4 import BeautifulSoup
        resp = httpx.get(url, follow_redirects=True, timeout=15,
                         headers={"User-Agent": "DeepResearch/1.0"})
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        text = " ".join(soup.get_text(separator=" ").split())
        return text[:4000] if len(text) > 4000 else text
    except Exception as e:
        return f"[fetch_url error: {e}]"


@tool
def fetch_url_headers(url: str) -> str:
    """Fetch HTTP response headers for a URL without downloading the full page.
    Returns status code, content-type, server, and other headers. Use to inspect URLs."""
    try:
        resp = httpx.head(url, follow_redirects=True, timeout=10,
                          headers={"User-Agent": "DeepResearch/1.0"})
        lines = [f"Status: {resp.status_code}"]
        for k, v in resp.headers.items():
            lines.append(f"{k}: {v}")
        return "\n".join(lines)
    except Exception as e:
        return f"[fetch_url_headers error: {e}]"


# ---------------------------------------------------------------------------
# Calculation and utility tools (stdlib only, free)
# ---------------------------------------------------------------------------

_SAFE_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _safe_eval(node):
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.BinOp):
        return _SAFE_OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp):
        return _SAFE_OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError(f"Unsafe expression node: {ast.dump(node)}")


@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression safely. Supports +, -, *, /, **, %, //.
    Examples: '2 ** 10', '(15 * 4) / 3', '100 * 1.08 ** 5'. No variables or functions."""
    try:
        tree = ast.parse(expression, mode="eval")
        result = _safe_eval(tree.body)
        return str(result)
    except Exception as e:
        return f"[calculate error: {e}]"


@tool
def get_current_datetime(timezone_name: str = "UTC") -> str:
    """Get the current date and time in a specified timezone.
    timezone_name should be an IANA timezone string like 'UTC', 'America/New_York', 'Asia/Tokyo'."""
    try:
        tz = ZoneInfo(timezone_name)
        now = datetime.now(tz)
        return now.strftime(f"%Y-%m-%d %H:%M:%S %Z ({timezone_name})")
    except ZoneInfoNotFoundError:
        return f"[get_current_datetime error: unknown timezone '{timezone_name}']"
    except Exception as e:
        return f"[get_current_datetime error: {e}]"


@tool
def convert_units(value: float, from_unit: str, to_unit: str) -> str:
    """Convert a value between physical units using pint. Supports length, weight, temperature, volume, speed, etc.
    Examples: convert_units(5, 'km', 'miles'), convert_units(100, 'celsius', 'fahrenheit')."""
    try:
        from pint import UnitRegistry
        ureg = UnitRegistry()
        qty = value * ureg(from_unit)
        result = qty.to(to_unit)
        return f"{value} {from_unit} = {result:.6g}"
    except Exception as e:
        return f"[convert_units error: {e}]"


@tool
def summarize_text(text: str, max_words: int = 200) -> str:
    """Truncate or summarize long text to a given word limit.
    Use when fetched content is too long to fit in context. Returns the first max_words words."""
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + f" ... [truncated {len(words) - max_words} more words]"


# ---------------------------------------------------------------------------
# Free public HTTP API tools (no API key required)
# ---------------------------------------------------------------------------

@tool
def currency_convert(amount: float, from_currency: str, to_currency: str) -> str:
    """Convert an amount between currencies using live exchange rates.
    Uses exchangerate-api.com (free, no API key). Currency codes: USD, EUR, GBP, JPY, etc."""
    try:
        resp = httpx.get(
            f"https://api.exchangerate-api.com/v4/latest/{from_currency.upper()}",
            timeout=10
        )
        resp.raise_for_status()
        data = resp.json()
        rates = data.get("rates", {})
        to = to_currency.upper()
        if to not in rates:
            return f"[currency_convert: unknown currency '{to_currency}']"
        converted = amount * rates[to]
        return f"{amount} {from_currency.upper()} = {converted:.4f} {to}"
    except Exception as e:
        return f"[currency_convert error: {e}]"


@tool
def define_word(word: str) -> str:
    """Look up the definition of an English word using the Free Dictionary API.
    Returns part of speech, definition, and usage example if available."""
    try:
        resp = httpx.get(
            f"https://api.dictionaryapi.dev/api/v2/entries/en/{word.lower()}",
            timeout=10
        )
        if resp.status_code == 404:
            return f"[define_word: '{word}' not found in dictionary]"
        resp.raise_for_status()
        entries = resp.json()
        out = []
        for entry in entries[:2]:
            for meaning in entry.get("meanings", [])[:2]:
                pos = meaning.get("partOfSpeech", "")
                for defn in meaning.get("definitions", [])[:2]:
                    d = defn.get("definition", "")
                    ex = defn.get("example", "")
                    out.append(f"[{pos}] {d}" + (f'\n  e.g. "{ex}"' if ex else ""))
        return "\n".join(out) if out else f"[define_word: no definitions found for '{word}']"
    except Exception as e:
        return f"[define_word error: {e}]"


@tool
def get_country_info(country_name: str) -> str:
    """Get information about a country: capital, population, region, languages, currencies.
    Uses the REST Countries API (free, no key). Pass the full country name, e.g. 'Germany'."""
    try:
        resp = httpx.get(
            f"https://restcountries.com/v3.1/name/{country_name}",
            timeout=10
        )
        if resp.status_code == 404:
            return f"[get_country_info: country '{country_name}' not found]"
        resp.raise_for_status()
        data = resp.json()[0]
        name = data.get("name", {}).get("common", country_name)
        capital = ", ".join(data.get("capital", ["N/A"]))
        population = f"{data.get('population', 0):,}"
        region = data.get("region", "N/A")
        subregion = data.get("subregion", "N/A")
        langs = ", ".join(data.get("languages", {}).values())
        currencies = ", ".join(
            f"{v.get('name')} ({v.get('symbol', '')})"
            for v in data.get("currencies", {}).values()
        )
        return (
            f"**{name}**\n"
            f"Capital: {capital}\n"
            f"Region: {region} / {subregion}\n"
            f"Population: {population}\n"
            f"Languages: {langs}\n"
            f"Currencies: {currencies}"
        )
    except Exception as e:
        return f"[get_country_info error: {e}]"


@tool
def get_github_repo(owner: str, repo: str) -> str:
    """Get public information about a GitHub repository: stars, forks, description, language, license.
    No authentication required for public repos (60 req/hr limit). E.g. owner='openai', repo='gpt-4'."""
    try:
        resp = httpx.get(
            f"https://api.github.com/repos/{owner}/{repo}",
            headers={"Accept": "application/vnd.github+json"},
            timeout=10
        )
        if resp.status_code == 404:
            return f"[get_github_repo: {owner}/{repo} not found]"
        resp.raise_for_status()
        d = resp.json()
        return (
            f"**{d['full_name']}**\n"
            f"{d.get('description', 'No description')}\n"
            f"Stars: {d.get('stargazers_count', 0):,} | Forks: {d.get('forks_count', 0):,}\n"
            f"Language: {d.get('language', 'N/A')} | License: {d.get('license', {}).get('name', 'N/A')}\n"
            f"Created: {d.get('created_at', '')[:10]} | Updated: {d.get('updated_at', '')[:10]}\n"
            f"URL: {d.get('html_url', '')}"
        )
    except Exception as e:
        return f"[get_github_repo error: {e}]"


@tool
def search_pypi(package_name: str) -> str:
    """Look up a Python package on PyPI. Returns version, summary, author, license, and homepage.
    Use to research Python libraries and their metadata."""
    try:
        resp = httpx.get(
            f"https://pypi.org/pypi/{package_name}/json",
            timeout=10
        )
        if resp.status_code == 404:
            return f"[search_pypi: package '{package_name}' not found on PyPI]"
        resp.raise_for_status()
        info = resp.json().get("info", {})
        return (
            f"**{info.get('name')}** v{info.get('version')}\n"
            f"{info.get('summary', 'No summary')}\n"
            f"Author: {info.get('author', 'N/A')}\n"
            f"License: {info.get('license', 'N/A')}\n"
            f"Homepage: {info.get('home_page') or info.get('project_url', 'N/A')}\n"
            f"Requires Python: {info.get('requires_python', 'N/A')}"
        )
    except Exception as e:
        return f"[search_pypi error: {e}]"


@tool
def get_public_ip_info() -> str:
    """Get information about the current machine's public IP address: location, ISP, timezone.
    Uses ipapi.co (free, no API key). Useful for network diagnostics or geolocation context."""
    try:
        resp = httpx.get("https://ipapi.co/json/", timeout=10)
        resp.raise_for_status()
        d = resp.json()
        return (
            f"IP: {d.get('ip')}\n"
            f"Location: {d.get('city')}, {d.get('region')}, {d.get('country_name')}\n"
            f"ISP: {d.get('org')}\n"
            f"Timezone: {d.get('timezone')}\n"
            f"Coordinates: {d.get('latitude')}, {d.get('longitude')}"
        )
    except Exception as e:
        return f"[get_public_ip_info error: {e}]"


ALL_DATA_TOOLS = [
    web_search,
    ddg_news,
    ddg_image_search,
    wiki_summary,
    wiki_search,
    arxiv_search,
    arxiv_paper_details,
    stock_price,
    stock_financials,
    stock_earnings,
    stock_news,
    market_summary,
    fetch_url,
    fetch_url_headers,
    calculate,
    get_current_datetime,
    convert_units,
    summarize_text,
    currency_convert,
    define_word,
    get_country_info,
    get_github_repo,
    search_pypi,
    get_public_ip_info,
]
