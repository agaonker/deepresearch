"""Tests for the centralized Jinja2 prompt layer."""
from __future__ import annotations

import pytest
from jinja2 import TemplateNotFound, UndefinedError

from deepresearch.prompts import KNOWN_TEMPLATES, render_prompt


def test_known_templates_match_files_on_disk():
    expected = {"system", "research", "compare", "summarize", "judge"}
    assert set(KNOWN_TEMPLATES) == expected


def test_system_prompt_renders_without_vars():
    out = render_prompt("system")
    assert "DeepResearch" in out
    # All six render tools must be advertised so the LLM knows the menu.
    for render in ("render_qa", "render_card", "render_table",
                   "render_chart", "render_timeline", "render_tree"):
        assert render in out


def test_system_prompt_long_enough_for_anthropic_caching():
    """Anthropic prompt caching has a per-block minimum that varies by model.
    Empirically on `claude-opus-4-7`, blocks below ~2000 tokens silently fall
    through to the non-cached path. ~4 chars/token in English, so the rendered
    text must clear ~6000 chars (≈1500 tokens, with a safety margin) to land
    above the threshold for the models we use today."""
    out = render_prompt("system")
    assert len(out) > 6000, f"system prompt is {len(out)} chars; need >6000 for caching"


def test_research_template_substitutes_arg():
    out = render_prompt("research", arg="transformer attention")
    assert "transformer attention" in out
    assert "render_qa" in out


def test_compare_template_substitutes_arg():
    out = render_prompt("compare", arg="NVDA vs AMD")
    assert "NVDA vs AMD" in out
    assert "render_table" in out


def test_summarize_template_substitutes_arg():
    out = render_prompt("summarize", arg="https://example.com/article")
    assert "https://example.com/article" in out
    assert "render_card" in out


def test_judge_template_substitutes_query_and_answer():
    out = render_prompt(
        "judge",
        query="What is the capital of France?",
        answer="Paris is the capital of France.",
    )
    assert "What is the capital of France?" in out
    assert "Paris is the capital of France." in out
    # Rubric anchors are required for the score parser to land on a real value.
    assert "Score:" in out
    for n in ("1", "2", "3", "4", "5"):
        assert n in out


def test_unknown_template_raises():
    with pytest.raises(TemplateNotFound):
        render_prompt("does_not_exist")


def test_strict_undefined_catches_missing_var():
    """Calling a template that needs `arg` without it should fail loudly,
    not silently render an empty string."""
    with pytest.raises(UndefinedError):
        render_prompt("research")
