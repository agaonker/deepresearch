"""Scorers for the LangSmith eval runner.

Two flavors:

- Programmatic scorers (`tool_recall`, `render_match`, `iterations_used`)
  are deterministic, free, and CI-safe.
- `llm_correctness` calls Haiku as judge and costs ~$0.005/example. The
  runner can skip it via `--no-llm-judge`.

Each scorer takes the LangSmith `Run` (the agent's output) and `Example`
(the dataset row) and returns a dict in the standard LangSmith feedback
shape: `{"key": <name>, "score": <0..1>, "comment": <optional>}`.
"""
from __future__ import annotations

import os
import re
from typing import Any


def _outputs(run: Any) -> dict:
    return run.outputs or {} if hasattr(run, "outputs") else (run.get("outputs") or {})


def _example_outputs(example: Any) -> dict:
    return example.outputs or {} if hasattr(example, "outputs") else (example.get("outputs") or {})


def _example_inputs(example: Any) -> dict:
    return example.inputs or {} if hasattr(example, "inputs") else (example.get("inputs") or {})


def tool_recall(run: Any, example: Any) -> dict:
    """Fraction of `must_include_tools` that the agent actually called."""
    expected = set(_example_outputs(example).get("must_include_tools", []))
    if not expected:
        return {"key": "tool_recall", "score": 1.0, "comment": "no expected tools"}
    actual = set(_outputs(run).get("tools_called", []))
    hits = len(expected & actual)
    score = hits / len(expected)
    missing = sorted(expected - actual)
    comment = f"missing: {missing}" if missing else "all expected tools called"
    return {"key": "tool_recall", "score": score, "comment": comment}


def render_match(run: Any, example: Any) -> dict:
    """1.0 if the run finished with the expected render tool, else 0.0."""
    expected = _example_outputs(example).get("expected_render")
    actual = _outputs(run).get("final_render")
    score = 1.0 if (expected and actual == expected) else 0.0
    return {
        "key": "render_match",
        "score": score,
        "comment": f"expected={expected!r} actual={actual!r}",
    }


def iterations_used(run: Any, example: Any) -> dict:
    """Inverted-cost metric: 1.0 for 1-2 iterations, lower for more.

    Capped at MAX_ITERATIONS=12. Useful as a soft signal — a run that
    needs many iterations may be retrying after tool failures.
    """
    n = int(_outputs(run).get("iterations", 0) or 0)
    if n <= 0:
        return {"key": "iterations_used", "score": 0.0, "comment": "no iterations recorded"}
    score = max(0.0, 1.0 - (n - 1) / 11)  # 1 → 1.0, 12 → 0.0
    return {"key": "iterations_used", "score": score, "comment": f"iterations={n}"}


_SCORE_RE = re.compile(r"\b([1-5])\b")


def llm_correctness(run: Any, example: Any) -> dict:
    """Haiku-as-judge: rate the agent's answer against the original query.

    Returns a 0..1 score derived from a 1-5 rating. Costs ~$0.005 per
    call. The runner skips this scorer when `--no-llm-judge` is set.
    """
    from langchain_anthropic import ChatAnthropic

    from deepresearch.prompts import render_prompt

    query = _example_inputs(example).get("query", "")
    answer = _outputs(run).get("answer_text", "")
    if not answer.strip():
        return {"key": "answer_correctness", "score": 0.0, "comment": "empty answer"}

    prompt = render_prompt("judge", query=query, answer=answer[:4000])
    judge = ChatAnthropic(  # type: ignore[call-arg]
        model=os.getenv("JUDGE_MODEL", "claude-haiku-4-5-20251001"),
        max_tokens=200,
        timeout=60,
    )
    try:
        response = judge.invoke(prompt)
        text = response.content if isinstance(response.content, str) else str(response.content)
    except Exception as e:
        return {"key": "answer_correctness", "score": 0.0, "comment": f"judge error: {e}"}

    match = _SCORE_RE.search(text)
    if not match:
        return {"key": "answer_correctness", "score": 0.0, "comment": f"no score parsed: {text[:200]}"}
    rating = int(match.group(1))
    return {
        "key": "answer_correctness",
        "score": rating / 5.0,
        "comment": text.strip()[:300],
    }


PROGRAMMATIC_SCORERS = (tool_recall, render_match, iterations_used)
ALL_SCORERS = PROGRAMMATIC_SCORERS + (llm_correctness,)
