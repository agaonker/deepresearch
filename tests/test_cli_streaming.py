"""Tests for token-level streaming in the CLI.

These exercise the two streaming hooks added in `cli.py` against
`langchain_core` message objects, using `capsys` to capture stdout. No
LLM call or graph execution is involved — we drive `_handle_message_chunk`
and `_print_latest` directly with the same shapes LangGraph would produce.
"""
from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage

from deepresearch.cli import (
    _PRINTED_IDS,
    _STREAMED_IDS,
    _handle_message_chunk,
    _print_latest,
)


@pytest.fixture(autouse=True)
def _reset_module_state() -> Any:
    """Module-level sets leak between tests; clear before and after."""
    _PRINTED_IDS.clear()
    _STREAMED_IDS.clear()
    yield
    _PRINTED_IDS.clear()
    _STREAMED_IDS.clear()


def _chunk(content: Any, msg_id: str = "m1", node: str = "agent") -> tuple[Any, dict[str, Any]]:
    """Build the (chunk, metadata) tuple LangGraph's messages mode emits."""
    return AIMessageChunk(content=content, id=msg_id), {"langgraph_node": node}


# ---------------------------------------------------------------------------
# _handle_message_chunk
# ---------------------------------------------------------------------------


def test_chunk_prints_string_content_without_newline(capsys: pytest.CaptureFixture[str]) -> None:
    _handle_message_chunk(_chunk("Hello, "))
    assert capsys.readouterr().out == "Hello, "


def test_chunk_concatenates_anthropic_style_text_blocks(
    capsys: pytest.CaptureFixture[str],
) -> None:
    blocks = [
        {"type": "text", "text": "Part one. "},
        {"type": "text", "text": "Part two."},
    ]
    _handle_message_chunk(_chunk(blocks))
    assert capsys.readouterr().out == "Part one. Part two."


def test_chunk_ignores_blocks_that_are_not_text(capsys: pytest.CaptureFixture[str]) -> None:
    blocks = [
        {"type": "text", "text": "keep"},
        {"type": "tool_use", "id": "tu1", "name": "x", "input": {}},
    ]
    _handle_message_chunk(_chunk(blocks))
    assert capsys.readouterr().out == "keep"


def test_chunk_ignores_non_agent_node(capsys: pytest.CaptureFixture[str]) -> None:
    _handle_message_chunk(_chunk("ignored", node="tools"))
    assert capsys.readouterr().out == ""


def test_chunk_records_streamed_message_id(capsys: pytest.CaptureFixture[str]) -> None:
    _handle_message_chunk(_chunk("hi", msg_id="msg-42"))
    assert "msg-42" in _STREAMED_IDS


def test_chunk_skips_empty_content_and_does_not_mark_streamed(
    capsys: pytest.CaptureFixture[str],
) -> None:
    _handle_message_chunk(_chunk("", msg_id="empty-1"))
    assert capsys.readouterr().out == ""
    # No content -> no print -> id should NOT be marked as streamed.
    assert "empty-1" not in _STREAMED_IDS


def test_chunk_ignores_non_message_chunk_objects(capsys: pytest.CaptureFixture[str]) -> None:
    # An already-complete AIMessage is not an AIMessageChunk; should be ignored.
    payload = (AIMessage(content="hi", id="m"), {"langgraph_node": "agent"})
    _handle_message_chunk(payload)
    assert capsys.readouterr().out == ""


# ---------------------------------------------------------------------------
# _print_latest — streamed AIMessage path
# ---------------------------------------------------------------------------


def test_print_latest_emits_newline_only_when_msg_was_streamed(
    capsys: pytest.CaptureFixture[str],
) -> None:
    msg = AIMessage(content="streamed body", id="m1")
    _STREAMED_IDS.add("m1")
    _print_latest({"messages": [msg]})
    # Body already shown via streaming; we only flush a trailing newline.
    assert capsys.readouterr().out == "\n"


def test_print_latest_appends_tool_calls_line_after_streamed_message(
    capsys: pytest.CaptureFixture[str],
) -> None:
    msg = AIMessage(
        content="streamed body",
        id="m1",
        tool_calls=[{"name": "wiki_search", "args": {"query": "BM25"}, "id": "tc1"}],
    )
    _STREAMED_IDS.add("m1")
    _print_latest({"messages": [msg]})
    out = capsys.readouterr().out
    # The tool-calls line lands AFTER the streamed body (reasoning -> action order).
    assert "[agent → tools] wiki_search(query=BM25)" in out
    assert out.startswith("\n")  # newline closes the streamed body first


def test_print_latest_falls_back_to_batched_print_when_not_streamed(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Mimics non-streaming providers (e.g., Ollama) where messages arrive whole.
    msg = AIMessage(content="not streamed body", id="m2")
    _print_latest({"messages": [msg]})
    assert "not streamed body" in capsys.readouterr().out


def test_print_latest_emits_tool_calls_for_batched_message(
    capsys: pytest.CaptureFixture[str],
) -> None:
    msg = AIMessage(
        content="non-streamed reasoning",
        id="m3",
        tool_calls=[{"name": "render_qa", "args": {"question": "Q?"}, "id": "tc"}],
    )
    _print_latest({"messages": [msg]})
    out = capsys.readouterr().out
    assert "non-streamed reasoning" in out
    assert "[agent → tools] render_qa" in out


# ---------------------------------------------------------------------------
# _print_latest — ToolMessage path (unchanged behavior, regression guard)
# ---------------------------------------------------------------------------


def test_print_latest_paints_render_tool_output(capsys: pytest.CaptureFixture[str]) -> None:
    render_payload = '_render::qa\n{"question":"Q?","answer":"A."}'
    msg = ToolMessage(content=render_payload, name="render_qa", tool_call_id="tc1", id="m-r")
    _print_latest({"messages": [msg]})
    out = capsys.readouterr().out
    assert "Q?" in out
    assert "A." in out


def test_print_latest_prints_truncated_snippet_for_non_render_tool(
    capsys: pytest.CaptureFixture[str],
) -> None:
    long = "x" * 800
    msg = ToolMessage(content=long, name="wiki_search", tool_call_id="tc2", id="m-t")
    _print_latest({"messages": [msg]})
    out = capsys.readouterr().out
    assert out.startswith("[tool:wiki_search] ")
    assert "..." in out  # the 600-char truncation marker
    assert len(out) < 800  # truncated


# ---------------------------------------------------------------------------
# _print_latest — dedup behavior (regression guard)
# ---------------------------------------------------------------------------


def test_print_latest_skips_messages_it_already_printed(
    capsys: pytest.CaptureFixture[str],
) -> None:
    msg = AIMessage(content="hello", id="m4")
    state = {"messages": [msg]}
    _print_latest(state)
    _print_latest(state)  # second call is a no-op via _PRINTED_IDS
    out = capsys.readouterr().out
    assert out.count("hello") == 1


def test_print_latest_handles_empty_messages_list(capsys: pytest.CaptureFixture[str]) -> None:
    _print_latest({"messages": []})
    assert capsys.readouterr().out == ""


def test_print_latest_handles_missing_messages_key(capsys: pytest.CaptureFixture[str]) -> None:
    _print_latest({})
    assert capsys.readouterr().out == ""
