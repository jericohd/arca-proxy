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
        elif t == "content_block_stop":
            idx = ev.get("index", 0)
            if idx < len(content):
                _finalize_block(content[idx])
        elif t == "message_delta":
            d = ev.get("delta", {})
            if "stop_reason" in d:
                message["stop_reason"] = d["stop_reason"]
            if "stop_sequence" in d:
                message["stop_sequence"] = d["stop_sequence"]
            if "usage" in ev:
                message["usage"] = ev["usage"]

    for blk in content:
        _finalize_block(blk)  # idempotent — covers streams missing content_block_stop
    message["content"] = content
    return json.dumps(message, ensure_ascii=False).encode("utf-8")


def _finalize_block(blk: dict[str, Any]) -> None:
    """Convert accumulated ``partial_json`` into the real ``input`` object.

    The non-streaming API shape for a tool_use block is
    ``{"type": "tool_use", "id": ..., "name": ..., "input": {...}}`` —
    ``partial_json`` is a streaming-only artifact and must not leak into the
    replayed message (a client would see a malformed tool call).
    """
    partial = blk.pop("partial_json", None)
    if partial is None:
        return
    try:
        blk["input"] = json.loads(partial) if partial else {}
    except ValueError:
        # Undecodable accumulated JSON — keep the raw text for forensics
        # rather than dropping the data silently.
        blk["input"] = {}
        blk["partial_json_unparsed"] = partial


def message_json_to_sse(raw: bytes) -> bytes:
    """Inverse of ``sse_to_message_json``: synthesize a minimal SSE stream from
    a complete non-streaming /v1/messages JSON body.

    Used when a response cached from a non-streaming miss is replayed to a
    client that requested ``stream: true``. The synthesized stream contains the
    same events Anthropic would send, with each text block emitted as a single
    delta (chunking granularity is not semantically meaningful to clients).
    """
    try:
        message = json.loads(raw)
    except ValueError:
        return raw  # not JSON — caller stored SSE already; pass through
    if not isinstance(message, dict):
        return raw

    content = message.get("content") or []
    usage = message.get("usage") or {}

    def _event(name: str, payload: dict[str, Any]) -> bytes:
        return f"event: {name}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n".encode("utf-8")

    head = {k: v for k, v in message.items() if k not in ("content", "stop_reason", "stop_sequence")}
    head["content"] = []
    out = [_event("message_start", {"type": "message_start", "message": head})]
    for idx, blk in enumerate(content):
        btype = blk.get("type")
        if btype == "text":
            out.append(_event("content_block_start", {
                "type": "content_block_start", "index": idx,
                "content_block": {"type": "text", "text": ""},
            }))
            out.append(_event("content_block_delta", {
                "type": "content_block_delta", "index": idx,
                "delta": {"type": "text_delta", "text": blk.get("text", "")},
            }))
        elif btype == "tool_use":
            out.append(_event("content_block_start", {
                "type": "content_block_start", "index": idx,
                "content_block": {"type": "tool_use", "id": blk.get("id"), "name": blk.get("name"), "input": {}},
            }))
            out.append(_event("content_block_delta", {
                "type": "content_block_delta", "index": idx,
                "delta": {"type": "input_json_delta",
                          "partial_json": json.dumps(blk.get("input", {}), ensure_ascii=False)},
            }))
        else:
            out.append(_event("content_block_start", {
                "type": "content_block_start", "index": idx, "content_block": blk,
            }))
        out.append(_event("content_block_stop", {"type": "content_block_stop", "index": idx}))
    out.append(_event("message_delta", {
        "type": "message_delta",
        "delta": {"stop_reason": message.get("stop_reason"), "stop_sequence": message.get("stop_sequence")},
        "usage": {"output_tokens": usage.get("output_tokens", 0)},
    }))
    out.append(_event("message_stop", {"type": "message_stop"}))
    return b"".join(out)
