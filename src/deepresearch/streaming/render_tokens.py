from __future__ import annotations

import os
import re
import sys

# Color enabled by default when stdout is a TTY and NO_COLOR is not set.
# Tests + pipes get monochrome automatically.
_color_enabled: bool = sys.stdout.isatty() and "NO_COLOR" not in os.environ


def set_color_enabled(value: bool) -> None:
    global _color_enabled
    _color_enabled = value


def color_enabled() -> bool:
    return _color_enabled


class Color:
    ACCENT = "\x1b[38;5;74m"
    MUTED = "\x1b[38;5;8m"
    SUCCESS = "\x1b[38;5;113m"
    WARN = "\x1b[38;5;221m"
    ERROR = "\x1b[38;5;203m"
    BOLD = "\x1b[1m"
    DIM = "\x1b[2m"
    RESET = "\x1b[0m"


class Box:
    # Standard single-line (cards, tables, separators)
    TL, TR, BL, BR = "┌", "┐", "└", "┘"
    H, V = "─", "│"
    LT, RT = "├", "┤"
    TT, BT, CROSS = "┬", "┴", "┼"
    # Heavy (reserved for the final committed answer)
    HTL, HTR, HBL, HBR = "┏", "┓", "┗", "┛"
    HH, HV = "━", "┃"
    HLT, HRT = "┣", "┫"
    # Tree / timeline connectors
    BRANCH, LAST = "├──", "└──"
    PIPE = "│   "
    SPACE = "    "
    # Bars
    BLOCK = "█"
    DOT = "·"


_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def visible_len(s: str) -> int:
    return len(_ANSI_RE.sub("", s))


def pad(s: str, w: int) -> str:
    return s + " " * max(0, w - visible_len(s))


def rpad(s: str, w: int) -> str:
    return " " * max(0, w - visible_len(s)) + s


def style(text: str, *codes: str) -> str:
    if not _color_enabled or not codes:
        return text
    return "".join(codes) + text + Color.RESET
