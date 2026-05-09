"""Upload the golden eval dataset to LangSmith.

Creates (or reuses) a LangSmith dataset and pushes every `EvalCase` from
`deepresearch.eval.dataset` as a single example. Idempotent on `name`:
re-running updates examples whose case name already exists rather than
duplicating them.

Reads `LANGCHAIN_API_KEY` from the environment (loaded from .env). Run with:

    uv run python scripts/upload_dataset.py
    uv run python scripts/upload_dataset.py --name deepresearch-golden-v2
    uv run python scripts/upload_dataset.py --dry-run
"""
from __future__ import annotations

import argparse
import sys
from typing import Any

from dotenv import load_dotenv
from langsmith import Client

from deepresearch.eval import GOLDEN_QUERIES, EvalCase

DEFAULT_NAME = "deepresearch-golden-v1"
DEFAULT_DESCRIPTION = (
    "Golden queries for the DeepResearch agent. Each example pairs a "
    "natural-language query with the tools an ideal run should reach for "
    "and the render tool that best fits the answer shape."
)


def _example_payload(case: EvalCase) -> dict[str, Any]:
    return {
        "inputs": {"query": case.query},
        "outputs": {
            "must_include_tools": list(case.must_include_tools),
            "expected_render": case.expected_render,
        },
        "metadata": {
            "name": case.name,
            "tags": list(case.tags),
        },
    }


def _get_or_create_dataset(client: Client, name: str, description: str) -> Any:
    try:
        return client.read_dataset(dataset_name=name)
    except Exception:
        return client.create_dataset(dataset_name=name, description=description)


def _existing_by_name(client: Client, dataset_id: str) -> dict[str, Any]:
    """Index existing examples by their metadata `name` so we can update in place."""
    out: dict[str, Any] = {}
    for ex in client.list_examples(dataset_id=dataset_id):
        meta = ex.metadata or {}
        case_name = meta.get("name")
        if case_name:
            out[case_name] = ex
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="upload_dataset")
    parser.add_argument("--name", default=DEFAULT_NAME, help="LangSmith dataset name")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print payloads without calling LangSmith.")
    args = parser.parse_args(argv)

    load_dotenv()

    print(f"Preparing {len(GOLDEN_QUERIES)} examples for dataset {args.name!r}")
    if args.dry_run:
        for case in GOLDEN_QUERIES:
            payload = _example_payload(case)
            print(f"  - {case.name}: {payload}")
        return 0

    client = Client()
    dataset = _get_or_create_dataset(client, args.name, DEFAULT_DESCRIPTION)
    print(f"Dataset id: {dataset.id}")

    existing = _existing_by_name(client, str(dataset.id))
    created, updated = 0, 0
    for case in GOLDEN_QUERIES:
        payload = _example_payload(case)
        if case.name in existing:
            ex = existing[case.name]
            client.update_example(
                example_id=ex.id,
                inputs=payload["inputs"],
                outputs=payload["outputs"],
                metadata=payload["metadata"],
            )
            updated += 1
            print(f"  ↻ updated {case.name}")
        else:
            client.create_example(
                dataset_id=dataset.id,
                inputs=payload["inputs"],
                outputs=payload["outputs"],
                metadata=payload["metadata"],
            )
            created += 1
            print(f"  + created {case.name}")

    print(f"Done. created={created} updated={updated} dataset={args.name!r}")
    print(f"View at https://smith.langchain.com/datasets/{dataset.id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
