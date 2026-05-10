"""Run an evaluation experiment against the LangSmith golden dataset.

For each example in `deepresearch-golden-v1`, run the compiled agent and
score the output with both programmatic scorers and (optionally) a Haiku
LLM-as-judge. Results land in LangSmith as a comparable experiment row.

Usage:
    uv run python scripts/run_experiment.py                       # 5 examples (smoke test)
    uv run python scripts/run_experiment.py --limit 50            # full pass
    uv run python scripts/run_experiment.py --no-llm-judge        # skip Haiku, free
    uv run python scripts/run_experiment.py --dataset my-other-ds # run a different dataset
    uv run python scripts/run_experiment.py --prefix my-exp       # name the experiment

Cost: ~$0.05 per agent run (Opus) + ~$0.005 per judge call (Haiku).
A 50-example pass with the judge on costs roughly $2.75.
"""
from __future__ import annotations

import argparse
import sys
import time
import uuid
from typing import Any

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver

from deepresearch.eval.scorers import ALL_SCORERS, PROGRAMMATIC_SCORERS
from deepresearch.graph.builder import build_graph

DEFAULT_DATASET = "deepresearch-golden-v1"


def _final_render(messages: list[Any]) -> str | None:
    """Last AI tool call whose name starts with `render_`."""
    for msg in reversed(messages):
        if not isinstance(msg, AIMessage) or not msg.tool_calls:
            continue
        for tc in msg.tool_calls:
            name = tc.get("name", "")
            if name.startswith("render_"):
                return name
    return None


def _all_tool_calls(messages: list[Any]) -> list[str]:
    out: list[str] = []
    for msg in messages:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                name = tc.get("name")
                if name:
                    out.append(name)
    return out


def _last_text(messages: list[Any]) -> str:
    """Return the user-visible answer text from the agent's run.

    The agent's deliberate, structured answer is whatever it passed to
    its final `render_*` tool call. The preamble prose ("I'll fetch the
    weather…") is mid-flow narration, not the answer. **Render content
    always wins** — only fall back to plain text when no render call
    happened.
    """
    for msg in reversed(messages):
        if not isinstance(msg, AIMessage) or not msg.tool_calls:
            continue
        for tc in msg.tool_calls:
            name = tc.get("name", "")
            if name.startswith("render_"):
                return _render_to_text(name, tc.get("args") or {})

    for msg in reversed(messages):
        if not isinstance(msg, AIMessage):
            continue
        content = msg.content
        if isinstance(content, str) and content.strip():
            return content
        if isinstance(content, list):
            parts = [b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"]
            joined = "".join(parts).strip()
            if joined:
                return joined
    return ""


def _render_to_text(name: str, args: dict) -> str:
    """Flatten a render_* tool call's args into a human-readable string."""
    if name == "render_qa":
        out = [str(args.get("answer", "")).strip()]
        sources = args.get("sources") or []
        if sources:
            out.append("Sources: " + ", ".join(str(s) for s in sources))
        return "\n".join(p for p in out if p)
    if name == "render_card":
        title = str(args.get("title", "")).strip()
        content = str(args.get("content", "")).strip()
        metadata = args.get("metadata") or {}
        meta = "\n".join(f"{k}: {v}" for k, v in metadata.items()) if isinstance(metadata, dict) else ""
        return "\n".join(p for p in (title, content, meta) if p)
    if name == "render_table":
        title = str(args.get("title", "")).strip()
        headers = args.get("headers") or []
        rows = args.get("rows") or []
        body = " | ".join(str(h) for h in headers) + "\n"
        body += "\n".join(" | ".join(str(c) for c in row) for row in rows)
        return f"{title}\n{body}".strip()
    if name == "render_chart":
        title = str(args.get("title", "")).strip()
        labels = args.get("labels") or []
        values = args.get("values") or []
        body = ", ".join(f"{lbl}={val}" for lbl, val in zip(labels, values, strict=False))
        return f"{title}\n{body}".strip()
    # render_timeline, render_tree, anything else — JSON-ish fallback
    import json
    try:
        return f"{args.get('title', name)}\n{json.dumps(args, default=str)[:2000]}"
    except Exception:
        return str(args)[:2000]


def _build_target(graph: Any) -> Any:
    """Return a target(inputs) -> outputs callable for langsmith.evaluate."""
    def target(inputs: dict) -> dict:
        query = inputs.get("query", "")
        config: RunnableConfig = {
            "configurable": {"thread_id": str(uuid.uuid4())},
            "metadata": {"command_used": "experiment", "iterations": 0, "tool_count": 0, "cancelled": False},
            "tags": ["deepresearch", "experiment"],
        }
        initial = {
            "messages": [HumanMessage(content=query)],
            "iterations": 0,
            "cancelled": False,
            "metadata": {},
        }
        result = graph.invoke(initial, config=config)
        messages = result.get("messages", [])
        return {
            "tools_called": _all_tool_calls(messages),
            "final_render": _final_render(messages),
            "iterations": result.get("iterations", 0),
            "answer_text": _last_text(messages),
        }

    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="run_experiment")
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="LangSmith dataset name")
    parser.add_argument("--limit", type=int, default=5, help="Cap on examples (default 5 for smoke)")
    parser.add_argument("--no-llm-judge", action="store_true", help="Skip the Haiku judge — programmatic scorers only")
    parser.add_argument("--prefix", default=None, help="Experiment name prefix (default: timestamp)")
    parser.add_argument("--max-concurrency", type=int, default=2, help="Parallel agent runs (default 2)")
    args = parser.parse_args(argv)

    load_dotenv()

    from langsmith import Client
    from langsmith.evaluation import evaluate

    client = Client()

    # Pick the example IDs we want to score so we can cap cost without
    # mutating the dataset itself.
    examples = list(client.list_examples(dataset_name=args.dataset))
    if args.limit and args.limit < len(examples):
        examples = examples[: args.limit]
    example_ids = [str(ex.id) for ex in examples]
    print(f"Running {len(example_ids)} example(s) from {args.dataset!r}")

    scorers = PROGRAMMATIC_SCORERS if args.no_llm_judge else ALL_SCORERS
    print(f"Scorers: {[s.__name__ for s in scorers]}")

    graph = build_graph(checkpointer=MemorySaver())
    target = _build_target(graph)

    prefix = args.prefix or f"exp-{time.strftime('%Y%m%d-%H%M%S')}"
    print(f"Experiment prefix: {prefix}")

    results = evaluate(
        target,
        data=client.list_examples(dataset_name=args.dataset, example_ids=example_ids),
        evaluators=list(scorers),
        experiment_prefix=prefix,
        max_concurrency=args.max_concurrency,
    )

    # Pull the experiment URL/name to print at the end. The shape of
    # `results` varies by langsmith version, so be defensive.
    exp_name = getattr(results, "experiment_name", None)
    if exp_name:
        print(f"Experiment: {exp_name}")
    print("Done. View results at https://smith.langchain.com → Experiments")
    return 0


if __name__ == "__main__":
    sys.exit(main())
