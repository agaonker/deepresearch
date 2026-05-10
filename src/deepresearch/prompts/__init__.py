"""Centralized Jinja2-rendered prompts.

All LLM-facing prompt text lives in `*.j2` files alongside this module so it's
auditable in one place. Call sites use `render_prompt(name, **vars)` to get a
fully-rendered string. Names map directly to filenames (`"system"` →
`system.j2`).

Why centralize:

- A long, stable `system` prompt is the prerequisite for Anthropic prompt
  caching (≥1024 tokens). Keeping it in a template makes it easy to grow
  without bloating Python source files.
- Slash-command expansions and the eval judge prompt benefit from the same
  versioning treatment so we can iterate without grepping the codebase.
"""
from __future__ import annotations

from jinja2 import Environment, PackageLoader, StrictUndefined, select_autoescape

KNOWN_TEMPLATES = ("system", "research", "compare", "summarize", "judge")

_env = Environment(
    loader=PackageLoader("deepresearch", "prompts"),
    autoescape=select_autoescape(disabled_extensions=("j2",), default=False),
    undefined=StrictUndefined,
    trim_blocks=True,
    lstrip_blocks=True,
    keep_trailing_newline=False,
)


def render_prompt(name: str, **vars: object) -> str:
    """Render `<name>.j2` from the prompts package with `vars` as the context.

    Uses `StrictUndefined` so a typo in a variable name fails loudly at
    render time rather than producing an empty string.
    """
    template = _env.get_template(f"{name}.j2")
    return template.render(**vars)


__all__ = ["KNOWN_TEMPLATES", "render_prompt"]
