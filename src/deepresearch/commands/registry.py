from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from langsmith import traceable

from deepresearch.tools.catalog import ALL_DATA_TOOLS
from deepresearch.tools.render import ALL_RENDER_TOOLS
from deepresearch.tools.retriever import ToolRetriever

_ALL_TOOLS = ALL_DATA_TOOLS + ALL_RENDER_TOOLS
_RETRIEVER = ToolRetriever(_ALL_TOOLS)


@dataclass
class CommandResult:
    """Result of dispatching a slash command.

    - kind="pure": already handled, `output` is text to print, do not run agent.
    - kind="expand": rewrite user query to `query`; run the agent with it.
    - kind="passthrough": no slash command matched; run the agent on the original text.
    """
    kind: str
    output: str = ""
    query: str = ""
    command_used: str = "free_text"


def _cmd_help(_arg: str) -> CommandResult:
    lines = [
        "Available commands:",
        "  /help                  Show this help",
        "  /tools                 List all tools in the catalog",
        "  /why <query>           Show top-K BM25-ranked tools for a query (no LLM call)",
        "  /research <topic>      Run a deep-research pass on a topic",
        "  /compare <a> vs <b>    Compare two items side-by-side",
        "  /summarize <text>      Summarize the provided text or URL",
        "Type any other text to send a free-form question to the agent.",
    ]
    return CommandResult(kind="pure", output="\n".join(lines), command_used="/help")


def _cmd_tools(_arg: str) -> CommandResult:
    lines = ["Tools in catalog:"]
    for t in _ALL_TOOLS:
        first = (t.description or "").strip().splitlines()[0] if t.description else ""
        lines.append(f"  - {t.name}: {first}")
    return CommandResult(kind="pure", output="\n".join(lines), command_used="/tools")


def _cmd_why(arg: str) -> CommandResult:
    if not arg.strip():
        return CommandResult(kind="pure", output="usage: /why <query>", command_used="/why")
    ranked = _RETRIEVER.explain(arg, k=8)
    out = [f"Top tools for: {arg!r}"]
    for r in ranked:
        out.append(f"  {r.score:6.3f}  {r.name}")
    return CommandResult(kind="pure", output="\n".join(out), command_used="/why")


def _cmd_research(arg: str) -> CommandResult:
    if not arg.strip():
        return CommandResult(kind="pure", output="usage: /research <topic>", command_used="/research")
    query = (
        f"Conduct a thorough research investigation on: {arg}. "
        "Use web search, Wikipedia, and other sources as needed. "
        "Cite URLs and finish with render_qa summarizing the key findings."
    )
    return CommandResult(kind="expand", query=query, command_used="/research")


def _cmd_compare(arg: str) -> CommandResult:
    if not arg.strip():
        return CommandResult(kind="pure", output="usage: /compare <a> vs <b>", command_used="/compare")
    query = (
        f"Compare the following: {arg}. "
        "Gather evidence on each item in parallel, then present the comparison "
        "using render_table with one row per item."
    )
    return CommandResult(kind="expand", query=query, command_used="/compare")


def _cmd_summarize(arg: str) -> CommandResult:
    if not arg.strip():
        return CommandResult(kind="pure", output="usage: /summarize <text or url>", command_used="/summarize")
    query = (
        f"Summarize the following. If it looks like a URL, fetch it first. "
        f"Then call render_card with a concise summary.\n\n{arg}"
    )
    return CommandResult(kind="expand", query=query, command_used="/summarize")


_HANDLERS: dict[str, Callable[[str], CommandResult]] = {
    "/help": _cmd_help,
    "/tools": _cmd_tools,
    "/why": _cmd_why,
    "/research": _cmd_research,
    "/compare": _cmd_compare,
    "/summarize": _cmd_summarize,
}


@traceable(run_type="chain", name="slash_command_dispatch")
def dispatch(text: str) -> CommandResult:
    """Parse a single line of user input and route it.

    Returns a `CommandResult` describing what should happen next:

    - `kind="pure"`: command was fully handled (e.g. `/help`, `/tools`,
      `/why`). Caller should print `output` and skip the agent.
    - `kind="expand"`: a slash command rewrote the query into a richer
      prompt for the LLM. Caller should run the agent on `query`.
    - `kind="passthrough"`: input was free text. Caller should run the
      agent on `query` (== the original text).

    Slash command matching is case-insensitive on the head only
    (`/HELP` == `/help`); the argument is preserved verbatim. Unknown
    slash commands are returned as `kind="pure"` with a hint, not
    raised — the REPL stays alive on typos.

    Decorated with `@traceable` so each dispatch appears as a
    `slash_command_dispatch` span in LangSmith.
    """
    text = text.strip()
    if not text.startswith("/"):
        return CommandResult(kind="passthrough", query=text, command_used="free_text")
    head, _, rest = text.partition(" ")
    handler = _HANDLERS.get(head.lower())
    if handler is None:
        return CommandResult(
            kind="pure",
            output=f"unknown command: {head}. Try /help.",
            command_used=head,
        )
    return handler(rest)
