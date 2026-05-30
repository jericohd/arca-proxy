"""SSE → canonical JSON replay (the demo-critical fidelity test).

RED until arca/cache_replay.py exists.
"""
from __future__ import annotations

import json
from pathlib import Path

from arca.cache_replay import sse_to_message_json


FIXTURES = Path(__file__).parent / "data"


def test_sse_to_json_plain_text():
    raw = (FIXTURES / "sse_plain_text.bin").read_bytes()
    msg = json.loads(sse_to_message_json(raw))
    assert msg["role"] == "assistant"
    assert msg["model"] == "claude-3-5-haiku-20241022"
    assert msg["content"] == [{"type": "text", "text": "Hello world"}]
    assert msg["stop_reason"] == "end_turn"
    assert msg["usage"]["output_tokens"] == 2


def test_sse_to_json_tool_use():
    raw = (FIXTURES / "sse_tool_use.bin").read_bytes()
    msg = json.loads(sse_to_message_json(raw))
    assert msg["role"] == "assistant"
    assert len(msg["content"]) == 1
    block = msg["content"][0]
    assert block["type"] == "tool_use"
    assert block["name"] == "get_weather"
    # partial_json concatenated
    assert block.get("partial_json") == '{"city":"Madrid"}'
    assert msg["stop_reason"] == "tool_use"


def test_sse_to_json_handles_crlf():
    """SSE RFC allows CRLF. Replay must strip \\r before JSON parsing."""
    raw = (FIXTURES / "sse_plain_text.bin").read_bytes().replace(b"\n", b"\r\n")
    msg = json.loads(sse_to_message_json(raw))
    assert msg["content"] == [{"type": "text", "text": "Hello world"}]


def test_sse_to_json_empty_stream_returns_empty_message():
    msg = json.loads(sse_to_message_json(b""))
    assert msg.get("content") == []
