# Tool catalog (36 data + 6 render = 42 total)

> Source of truth: [`src/deepresearch/tools/catalog.py`](../src/deepresearch/tools/catalog.py) and [`src/deepresearch/tools/render.py`](../src/deepresearch/tools/render.py). The list below is generated to match.

## Data tools (36)

**Web search & news** (5)
| Tool | Description |
|---|---|
| `web_search` | DuckDuckGo web search — titles, URLs, snippets. |
| `ddg_news` | DuckDuckGo News — recent headlines and summaries. |
| `ddg_image_search` | DuckDuckGo image search — image metadata + URLs. |
| `hackernews_search` | Hacker News stories and comments via public Algolia API. |
| `reddit_search` | Reddit posts via the public JSON endpoint (no auth). |

**Encyclopedic & factual** (4)
| Tool | Description |
|---|---|
| `wiki_summary` | Concise Wikipedia summary for a topic. |
| `wiki_search` | List of Wikipedia article titles matching a query. |
| `wikidata_query` | Run a SPARQL query against Wikidata for structured facts. |
| `get_country_info` | Country profile — capital, population, region, languages, currencies. |

**Academic literature** (4)
| Tool | Description |
|---|---|
| `arxiv_search` | Search arXiv for papers, preprints, scientific research. |
| `arxiv_paper_details` | Full metadata for a specific arXiv paper by ID. |
| `pubmed_search` | Biomedical and life-sciences literature via NCBI E-utilities. |
| `crossref_search` | Academic papers and DOI metadata across all disciplines. |

**Finance & markets** (6)
| Tool | Description |
|---|---|
| `stock_price` | Current price + key metrics (cap, P/E, 52-week range) for a ticker. |
| `stock_financials` | Annual financial statements — revenue, net income, EBITDA. |
| `stock_earnings` | Recent earnings history and upcoming earnings dates. |
| `stock_news` | Recent news headlines for a ticker (Yahoo Finance). |
| `market_summary` | S&P 500, Dow, NASDAQ snapshot. |
| `coingecko_price` | Current cryptocurrency prices via CoinGecko (no API key). |

**Geography & weather** (3)
| Tool | Description |
|---|---|
| `osm_geocode` | Address/place → coordinates via OSM Nominatim. |
| `weather_now` | Current weather + short forecast via wttr.in. |
| `open_meteo_forecast` | Multi-day forecast via Open-Meteo. |

**Macro & demographics** (1)
| Tool | Description |
|---|---|
| `world_bank_indicator` | World Bank indicator time-series (GDP, population, life expectancy, …). |

**Web fetch & document extraction** (3)
| Tool | Description |
|---|---|
| `fetch_url` | Fetch a web page and return readable text (HTML stripped). |
| `fetch_url_headers` | Fetch HTTP response headers without downloading the body. |
| `pdf_extract_text` | Extract text from a PDF (URL or local path). |

**Multimedia** (1)
| Tool | Description |
|---|---|
| `youtube_transcript` | Fetch the captions/transcript for a YouTube video. |

**Developer ecosystem** (2)
| Tool | Description |
|---|---|
| `get_github_repo` | Public GitHub repo info — stars, forks, language, license. |
| `search_pypi` | PyPI package metadata — version, summary, author, license. |

**Utilities** (7)
| Tool | Description |
|---|---|
| `calculate` | Evaluate a math expression safely. |
| `get_current_datetime` | Current date/time in a given timezone. |
| `convert_units` | Convert between physical units (length, mass, temperature, …) via pint. |
| `currency_convert` | Convert between currencies at live exchange rates. |
| `define_word` | English dictionary lookup via Free Dictionary API. |
| `get_public_ip_info` | Geolocation/ISP info for the host's public IP. |
| `summarize_text` | Truncate or summarize long text to a word limit. |

## Render tools (6)

Each emits a `_render::<kind>\n<json>` sentinel that the CLI parses and paints as ASCII.

| Tool | Description |
|---|---|
| `render_card` | Info card with a title and content body. |
| `render_table` | ASCII table with headers and rows. |
| `render_chart` | ASCII bar or line chart. |
| `render_qa` | Question/answer pair with optional source citations. |
| `render_timeline` | Chronological list of events. |
| `render_tree` | Indented hierarchical tree. |
