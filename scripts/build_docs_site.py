"""Build the deepresearch docs site.

Renders the project's markdown into a small static site styled with the same
brutalist tokens as the CLI render system (`streaming/render_tokens.py`):
sharp single-line borders, semantic accent palette, monospace throughout,
terminal-friendly dark/light.

Output: `_site/` at repo root (gitignored).

Usage:
    uv run python scripts/build_docs_site.py
    open _site/index.html

The site is the design system applied to a web surface, on purpose: same
palette, same structure-first philosophy, same "no painted background"
rule (CSS uses CSS variables that honor `prefers-color-scheme`).
"""
from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "_site"


@dataclass(frozen=True)
class Page:
    slug: str
    src: Path
    title: str
    nav_label: str


PAGES: tuple[Page, ...] = (
    Page("index", ROOT / "README.md", "deepresearch", "home"),
    Page("design-system", ROOT / "DESIGN.md", "design system", "design"),
    Page("testing-providers", ROOT / "docs" / "testing-providers.md",
         "testing providers", "providers"),
    Page("reviewer-agent", ROOT / "docs" / "reviewer-agent.md",
         "reviewer agent (parked)", "reviewer"),
)


CSS = r"""
:root {
  --accent: #5FAFD7;
  --muted:  #6C6C6C;
  --success:#87D75F;
  --warn:   #FFD75F;
  --error:  #FF5F5F;
  --bg:     #0e1014;
  --fg:     #e6e6e6;
  --rule:   #2a2d33;
}
@media (prefers-color-scheme: light) {
  :root {
    --bg:   #ffffff;
    --fg:   #1a1a1a;
    --muted:#6c6c6c;
    --rule: #e6e6e6;
  }
}
html, body { margin: 0; padding: 0; background: var(--bg); color: var(--fg); }
body {
  font-family: "JetBrains Mono", "SF Mono", "Menlo", "Consolas", ui-monospace, monospace;
  font-size: 13.5px;
  line-height: 1.55;
}
.wrap { max-width: 84ch; margin: 0 auto; padding: 32px 32px 96px; }

/* Header / nav */
header {
  border-bottom: 1px solid var(--rule);
  padding: 24px 32px;
  position: sticky; top: 0;
  background: var(--bg);
  z-index: 10;
}
.bar { max-width: 84ch; margin: 0 auto; display: flex; gap: 24px; align-items: baseline; }
.brand {
  font-weight: 700;
  color: var(--accent);
  letter-spacing: 0.04em;
  text-decoration: none;
}
.brand::before { content: "▌ "; color: var(--accent); }
nav { display: flex; gap: 16px; flex-wrap: wrap; }
nav a {
  color: var(--muted);
  text-decoration: none;
  border-bottom: 1px solid transparent;
}
nav a:hover { color: var(--fg); border-bottom-color: var(--accent); }
nav a.current { color: var(--accent); border-bottom-color: var(--accent); }

/* Typography */
main h1 { font-size: 1.6em; font-weight: 700; color: var(--accent); margin: 0 0 12px; }
main h2 {
  font-size: 1.15em;
  font-weight: 700;
  margin: 32px 0 12px;
  padding-bottom: 4px;
  border-bottom: 1px solid var(--rule);
}
main h3 { font-size: 1em; font-weight: 700; margin: 24px 0 8px; }
main p, main li { margin: 8px 0; }
main a { color: var(--accent); text-decoration: underline; text-underline-offset: 3px; }
main a:hover { text-decoration-thickness: 2px; }
main strong { color: var(--fg); font-weight: 700; }
main em { font-style: italic; color: var(--fg); }

/* Code (sharp single-line, no rounded corners, semantic accent) */
code {
  font-family: inherit;
  font-size: 0.95em;
  color: var(--accent);
  background: transparent;
  padding: 0 2px;
}
pre {
  border: 1px solid var(--rule);
  padding: 12px 16px;
  margin: 12px 0;
  overflow-x: auto;
  white-space: pre;
  background: transparent;
}
pre code { color: var(--fg); padding: 0; }

/* Tables */
table {
  border-collapse: collapse;
  margin: 16px 0;
  width: 100%;
}
th, td {
  border: 1px solid var(--rule);
  padding: 6px 10px;
  text-align: left;
  vertical-align: top;
}
th {
  color: var(--accent);
  font-weight: 700;
}

/* Lists */
main ul, main ol { padding-left: 24px; }
main ul li::marker { color: var(--muted); }
main ol li::marker { color: var(--muted); }

/* Blockquote */
blockquote {
  border-left: 2px solid var(--accent);
  padding-left: 16px;
  margin: 16px 0;
  color: var(--muted);
}
blockquote strong { color: var(--accent); }

/* HR */
hr { border: none; border-top: 1px solid var(--rule); margin: 32px 0; }

/* Images */
img { max-width: 100%; height: auto; border: 1px solid var(--rule); }

/* Footer */
footer {
  border-top: 1px solid var(--rule);
  padding: 24px 32px;
  color: var(--muted);
  font-size: 12.5px;
}
.footer-inner { max-width: 84ch; margin: 0 auto; }
"""


TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title} · deepresearch</title>
<link rel="stylesheet" href="styles.css">
</head>
<body>
<header>
  <div class="bar">
    <a href="index.html" class="brand">DEEPRESEARCH</a>
    <nav>{nav}</nav>
  </div>
</header>
<main class="wrap">
{content}
</main>
<footer>
  <div class="footer-inner">
    Brutalist render design system, applied to a web surface. Source:
    <a href="https://github.com/agaonker/deepresearch">github.com/agaonker/deepresearch</a>.
    Same palette and rules as <code>src/deepresearch/streaming/render_tokens.py</code>.
  </div>
</footer>
</body>
</html>
"""


def render_nav(current_slug: str, pages: list[Page]) -> str:
    return "".join(
        f'<a href="{p.slug}.html" '
        f'class="{"current" if p.slug == current_slug else ""}">{p.nav_label}</a>'
        for p in pages
    )


def rewrite_relative_links(html: str) -> str:
    """Rewrite repo-relative markdown links so they resolve inside _site/."""
    return (
        html
        .replace('href="DESIGN.md"', 'href="design-system.html"')
        .replace('href="docs/testing-providers.md"', 'href="testing-providers.html"')
        .replace('href="docs/reviewer-agent.md"', 'href="reviewer-agent.html"')
        .replace('href="README.md"', 'href="index.html"')
        .replace('src="docs/images/', 'src="images/')
        .replace('src="docs/screenshots/', 'src="screenshots/')
    )


def build() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)

    # Copy assets
    for asset_dir in ("docs/images", "docs/screenshots"):
        src = ROOT / asset_dir
        if src.exists():
            shutil.copytree(src, OUT / src.name)

    # Write stylesheet
    (OUT / "styles.css").write_text(CSS)

    # Render each page (skip missing sources — e.g. reviewer-agent only lands on a branch)
    md = markdown.Markdown(extensions=["fenced_code", "tables", "toc", "sane_lists"])
    rendered_pages: list[Page] = []
    for page in PAGES:
        if not page.src.exists():
            print(f"  skip {page.slug} ({page.src.relative_to(ROOT)} not found)")
            continue
        rendered_pages.append(page)

    for page in rendered_pages:
        text = page.src.read_text()
        md.reset()
        html_body = md.convert(text)
        html_body = rewrite_relative_links(html_body)
        out_path = OUT / f"{page.slug}.html"
        out_path.write_text(
            TEMPLATE.format(
                title=page.title,
                nav=render_nav(page.slug, rendered_pages),
                content=html_body,
            )
        )
        print(f"  wrote {out_path.relative_to(ROOT)}")

    print(f"\ndone. open {(OUT / 'index.html').relative_to(ROOT)} to preview.")


if __name__ == "__main__":
    build()
