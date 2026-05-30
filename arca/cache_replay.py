"""Convert a stored SSE byte buffer back into an Anthropic /v1/messages JSON response.

Anthropic SSE event sequence (verified against platform.claude.com/docs/en/api/messages-streaming):
    message_start, content_block_start, content_block_delta*,
    content_block_stop, message_delta, message_stop

The canonical JSON response for stream=false contains the assembled message
with content blocks concatenated from their deltas. We rebuild it by scanning
`data: {...}` lines and merging events.

Pitfall 7: SSE RFC permits both LF and CRLF. `line.strip()` handles both.
"""
from __future__ import annotations

import json
from typing import Any


def sse_to_message_json(raw: bytes) -> bytes:
    events: list[dict[str, Any]] = []
    for line in raw.split(b"\n"):
        line = line.strip()  # Pitfall 7: strip \r (CRLF) and whitespace
        if not line.startswith(b"data: "):
            continue
        try:
            events.append(json.loads(line[6:].decode("utf-8")))
        except Exception:
            continue

    message: dict[str, Any] = {"content": []}
    content: list[dict[str, Any]] = []

    for ev in events:
        t = ev.get("type")
        if t == "message_start":
            message = dict(ev.get("message", {}))
            message.setdefault("content", [])
        elif t == "content_block_start":
            block = dict(ev.get("content_block", {}))
            idx = ev.get("index", 0)
            while len(content) <= idx:
                content.append({})
            content[idx] = block
        elif t == "content_block_delta":
            idx = ev.get("index", 0)
            delta = ev.get("delta", {})
            if idx < len(content):
                blk = content[idx]
                if delta.get("type") == "text_delta":
                    blk["text"] = blk.get("text", "") + delta.get("text", "")
                elif delta.get("type") == "input_json_delta":
                    blk["partial_json"] = blk.get("partial_json", "") + delta.get("partial_json", "")
        elif t == "message_delta":
            d = ev.get("delta", {})
            if "stop_reason" in d:
                message["stop_reason"] = d["stop_reason"]
            if "stop_sequence" in d:
                message["stop_sequence"] = d["stop_sequence"]
            if "usage" in ev:
                message["usage"] = ev["usage"]

    message["content"] = content
    return json.dumps(message, ensure_ascii=False).encode("utf-8")
