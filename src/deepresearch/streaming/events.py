from __future__ import annotations

import json
from dataclasses import dataclass

_SENTINEL = "_render::"


@dataclass
class RenderPayload:
    kind: str
    data: dict


def parse_render(text: str) -> RenderPayload | None:
    if not isinstance(text, str) or not text.startswith(_SENTINEL):
        return None
    rest = text[len(_SENTINEL):]
    newline = rest.find("\n")
    if newline == -1:
        return None
    kind = rest[:newline].strip()
    body = rest[newline + 1:]
    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        return None
    return RenderPayload(kind=kind, data=data)
