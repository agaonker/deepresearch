#!/usr/bin/env bash
#
# demo.sh — five representative queries that exercise each render tool.
#
# Use this to:
#   1. Smoke-test the agent end-to-end on a fresh checkout.
#   2. Generate a clean batch of LangSmith traces for screenshots.
#   3. Capture example terminal output for the README.
#
# Each query is chosen to bias the LLM toward a different render tool so
# screenshots cover the full UI surface (qa, card, table, chart, timeline).
# Live runs cost a small amount of Anthropic credits (~$0.05–$0.30 total).
#
# Usage:
#   ./scripts/demo.sh                 # run all 5 queries, normal LangSmith tracing
#   ./scripts/demo.sh --no-trace      # skip LangSmith (offline / cheap iteration)
#   ./scripts/demo.sh --explain-only  # print BM25 ranking only, no LLM call
#   ./scripts/demo.sh 3               # run only query #3
#
# Tip: tee the output to a file to capture example transcripts for the README:
#   ./scripts/demo.sh | tee docs/example-usage.txt

set -euo pipefail

cd "$(dirname "$0")/.."

CLI=(uv run research)
EXTRA_FLAGS=()
ONLY=""
EXPLAIN_ONLY=0

for arg in "$@"; do
  case "$arg" in
    --no-trace)
      EXTRA_FLAGS+=(--no-trace)
      ;;
    --explain-only)
      EXPLAIN_ONLY=1
      ;;
    [1-5])
      ONLY="$arg"
      ;;
    *)
      echo "Unknown arg: $arg" >&2
      exit 2
      ;;
  esac
done

# Each entry: "<id>|<expected_render>|<query>"
QUERIES=(
  "1|render_qa|What is the current stock price of NVDA, and what is its P/E ratio?"
  "2|render_card|Summarize the Wikipedia article on the French Revolution in five sentences."
  "3|render_table|Compare NVDA and AMD: market cap, P/E ratio, and 52-week range."
  "4|render_chart|Show India's GDP from 2015 to 2023 from the World Bank, then plot it."
  "5|render_timeline|Build a timeline of major SpaceX launches from 2008 to 2024."
)

run_one() {
  local id="$1" expected="$2" query="$3"
  echo
  echo "═══════════════════════════════════════════════════════════════════════════════"
  echo "  Demo $id  ·  expected render: $expected"
  echo "  Query: $query"
  echo "═══════════════════════════════════════════════════════════════════════════════"
  echo

  if [[ "$EXPLAIN_ONLY" == "1" ]]; then
    "${CLI[@]}" --explain-tools "$query"
  else
    "${CLI[@]}" ${EXTRA_FLAGS[@]+"${EXTRA_FLAGS[@]}"} "$query"
  fi
}

for entry in "${QUERIES[@]}"; do
  IFS='|' read -r id expected query <<<"$entry"
  if [[ -n "$ONLY" && "$ONLY" != "$id" ]]; then
    continue
  fi
  run_one "$id" "$expected" "$query"
done

echo
echo "Done. If LangSmith tracing was on, see https://smith.langchain.com → project deepresearch-agent."
