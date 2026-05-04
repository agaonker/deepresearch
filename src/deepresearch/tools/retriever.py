from __future__ import annotations

import re
from dataclasses import dataclass

from langchain_core.tools import BaseTool
from langsmith import traceable
from rank_bm25 import BM25Okapi

ALWAYS_INCLUDE = {
    "render_card",
    "render_table",
    "render_chart",
    "render_qa",
    "calculate",
    "summarize_text",
}

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text or "")]


@dataclass
class RankedTool:
    """A tool name paired with its BM25 relevance score for a given query."""

    name: str
    score: float


class ToolRetriever:
    """BM25-based dynamic tool selector.

    Indexes each tool's `name + description` and ranks them against a query.
    Used by the agent to bind only the top-K most relevant tools per LLM call,
    keeping the tool-call prompt small and the model focused. Tools listed in
    `always_include` are appended after the top-K so render tools and
    general-purpose utilities are reachable on every turn.

    Why: a 28-tool catalog blown into every prompt wastes tokens and confuses
    smaller models. BM25 over names + docstrings is a cheap, deterministic,
    explainable alternative to embedding-based retrieval.
    """

    def __init__(
        self,
        tools: list[BaseTool],
        always_include: set[str] | None = None,
    ) -> None:
        self.tools = tools
        self.by_name = {t.name: t for t in tools}
        self.always_include = always_include if always_include is not None else ALWAYS_INCLUDE

        self._corpus = [_tokenize(f"{t.name} {t.description or ''}") for t in tools]
        self._bm25 = BM25Okapi(self._corpus)

    @traceable(run_type="retriever", name="bm25_tool_selection")
    def search(self, query: str, k: int = 8) -> list[BaseTool]:
        """Return the top-K tools by BM25 score for `query`, plus `always_include`.

        The returned list has at most `k + len(always_include)` items, deduped.
        Empty queries fall back to insertion order so callers always get a
        non-empty result. Decorated with `@traceable` so each selection appears
        as a `bm25_tool_selection` span in LangSmith.
        """
        tokens = _tokenize(query)
        if not tokens:
            ranked_names = [t.name for t in self.tools]
        else:
            scores = self._bm25.get_scores(tokens)
            ranked = sorted(
                zip(self.tools, scores, strict=True), key=lambda x: x[1], reverse=True
            )
            ranked_names = [t.name for t, _ in ranked]

        selected: list[BaseTool] = []
        seen: set[str] = set()
        for name in ranked_names:
            if len(selected) >= k:
                break
            if name in seen:
                continue
            selected.append(self.by_name[name])
            seen.add(name)

        for name in self.always_include:
            if name in seen:
                continue
            tool = self.by_name.get(name)
            if tool is not None:
                selected.append(tool)
                seen.add(name)

        return selected

    def explain(self, query: str, k: int = 8) -> list[RankedTool]:
        """Return the top-K tools paired with their raw BM25 scores.

        Used by `--explain-tools` and `/why` to debug ranking decisions
        without invoking the LLM. Does not apply `always_include`.
        """
        tokens = _tokenize(query)
        scores = self._bm25.get_scores(tokens) if tokens else [0.0] * len(self.tools)
        ranked = sorted(
            (RankedTool(t.name, float(s)) for t, s in zip(self.tools, scores, strict=True)),
            key=lambda r: r.score,
            reverse=True,
        )
        return ranked[:k]
