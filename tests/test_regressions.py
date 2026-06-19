"""Regression tests for the 2026-06 audit fixes.

Each test pins one fix from AUDIT.md so it cannot silently regress:
configurable namespace, circuit-breaker timeout accounting, local semantic
fallback, non-streaming caching, tool_use replay, and VS guards.
"""
from __future__ import annotations

import asyncio
import json
from unittest.mock import MagicMock, patch

import httpx
import numpy as np
import pytest
from httpx import ASGITransport, AsyncClient

from arca.config import Settings, get_settings


# ---------------------------------------------------------------------------
# Config (AUDIT P0-1)
# ---------------------------------------------------------------------------
def test_settings_defaults():
    s = Settings()
    assert s.cache_table == f"{s.catalog}.{s.db_schema}.cache_store"
    assert s.usage_table.endswith(".usage_log")
    assert s.vs_index.endswith(".prompt_index")
    assert s.similarity_threshold == 0.90  # lowered from 0.95; safe via the L2 polarity guard


def test_settings_env_overrides(monkeypatch):
    monkeypatch.setenv("ARCA_CATALOG", "acme")
    monkeypatch.setenv("ARCA_SCHEMA", "ai_cache")
    monkeypatch.setenv("ARCA_SIMILARITY_THRESHOLD", "0.9")
    s = Settings()
    assert s.cache_table == "acme.ai_cache.cache_store"
    assert s.vs_index == "acme.ai_cache.prompt_index"
    assert s.similarity_threshold == 0.9


def test_get_settings_is_cached():
    assert get_settings() is get_settings()


# ---------------------------------------------------------------------------
# Circuit breaker timeout accounting (AUDIT P1-2)
# ---------------------------------------------------------------------------
async def test_circuit_breaker_records_timeouts():
    """A hanging dependency must trip the breaker — the original code wrapped
    the breaker in an external wait_for, so cancellation was never recorded."""
    from arca.proxy import CBState, CircuitBreaker

    cb = CircuitBreaker(failure_threshold=2, window_seconds=30.0, reset_timeout=60.0)

    async def hang():
        await asyncio.sleep(10)

    for _ in range(2):
        with pytest.raises((asyncio.TimeoutError, TimeoutError)):
            await cb.call(hang, timeout=0.02)

    assert cb.state == CBState.OPEN


async def test_circuit_breaker_timeout_none_keeps_old_behavior():
    from arca.proxy import CBState, CircuitBreaker

    cb = CircuitBreaker(failure_threshold=2, window_seconds=30.0, reset_timeout=60.0)

    async def ok():
        return 42

    assert await cb.call(ok) == 42
    assert cb.state == CBState.CLOSED


# ---------------------------------------------------------------------------
# Pricing (AUDIT P1-1) — cross-generation coverage beyond test_observability
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "model,in_rate,out_rate",
    [
        ("claude-fable-5", 10.0, 50.0),
        ("claude-opus-4-8", 5.0, 25.0),
        ("claude-opus-4-5-20251101", 5.0, 25.0),
        ("claude-opus-4-1-20250805", 15.0, 75.0),
        ("claude-opus-4-20250514", 15.0, 75.0),
        ("claude-sonnet-4-6", 3.0, 15.0),
        ("claude-haiku-4-5-20251001", 1.0, 5.0),
        ("claude-3-7-sonnet-20250219", 3.0, 15.0),
        ("claude-3-5-haiku-20241022", 0.80, 4.0),
        ("claude-3-haiku-20240307", 0.25, 1.25),
        ("claude-3-opus-20240229", 15.0, 75.0),
    ],
)
def test_every_known_family_priced(model, in_rate, out_rate):
    from arca.observability import calculate_cost

    got = calculate_cost(model, 1_000_000, 1_000_000)
    assert got == pytest.approx(in_rate + out_rate, rel=1e-9), model


# ---------------------------------------------------------------------------
# extract_tokens on non-streaming JSON bodies (AUDIT P1-4)
# ---------------------------------------------------------------------------
def test_extract_tokens_json_body():
    from arca.observability import extract_tokens

    body = json.dumps({
        "type": "message",
        "usage": {"input_tokens": 321, "output_tokens": 45},
    }).encode()
    assert extract_tokens(body) == (321, 45)


def test_extract_tokens_invalid_json_body():
    from arca.observability import extract_tokens

    assert extract_tokens(b"{not json") == (0, 0)


# ---------------------------------------------------------------------------
# Replay conversions (AUDIT P1-4 / P1-5)
# ---------------------------------------------------------------------------
def test_message_json_to_sse_roundtrip():
    from arca.cache_replay import message_json_to_sse, sse_to_message_json

    message = {
        "id": "msg_x",
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet-4-6",
        "content": [
            {"type": "text", "text": "Use shutil.rmtree."},
            {"type": "tool_use", "id": "toolu_1", "name": "get_weather",
             "input": {"city": "Madrid"}},
        ],
        "stop_reason": "tool_use",
        "stop_sequence": None,
        "usage": {"input_tokens": 10, "output_tokens": 25},
    }
    sse = message_json_to_sse(json.dumps(message).encode())
    assert b"event: message_start" in sse
    assert b"event: message_stop" in sse

    back = json.loads(sse_to_message_json(sse))
    assert back["content"][0] == {"type": "text", "text": "Use shutil.rmtree."}
    tool = back["content"][1]
    assert tool["type"] == "tool_use"
    assert tool["input"] == {"city": "Madrid"}
    assert "partial_json" not in tool
    assert back["stop_reason"] == "tool_use"
    assert back["usage"]["output_tokens"] == 25


def test_build_hit_response_json_to_both_client_modes():
    from arca.cache import _build_hit_response

    cached = json.dumps({
        "type": "message", "role": "assistant",
        "content": [{"type": "text", "text": "hi"}],
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }).encode()

    json_resp = _build_hit_response(cached, b'{"stream": false}')
    assert json_resp.media_type == "application/json"
    assert json.loads(json_resp.body)["content"][0]["text"] == "hi"

    sse_resp = _build_hit_response(cached, b'{"stream": true}')
    assert sse_resp.media_type == "text/event-stream"
    assert b"event: message_stop" in sse_resp.body


# ---------------------------------------------------------------------------
# VS guards (AUDIT P1-3)
# ---------------------------------------------------------------------------
def test_vs_upsert_noop_without_index():
    from arca.cache import _vs_upsert_sync

    with patch("arca.cache._get_vs_index", return_value=None):
        # Must not raise and must not log an AttributeError per miss
        _vs_upsert_sync("id", "hash", "text", [0.0] * 384, "{}")


# ---------------------------------------------------------------------------
# Local semantic fallback (new capability backing the demo/benchmark)
# ---------------------------------------------------------------------------
async def test_local_l2_lookup_hit_and_miss(arca_home):
    from arca.cache import SIMILARITY_THRESHOLD, _local_l2_lookup
    from arca.fallback import SQLiteFallback

    store = SQLiteFallback()
    await store.start()
    vec = np.zeros(384, dtype=np.float32)
    vec[0] = 1.0  # unit norm
    await store.enqueue({
        "id": "row1", "prompt_hash": "hash1", "prompt_text": "{}",
        "embedding": vec.tolist(), "response_json": '{"type":"message"}',
        "model": "m", "cost_usd": 0.0,
    })
    try:
        with patch("arca.cache._get_sqlite_fallback", return_value=store):
            hit = await _local_l2_lookup(vec)
            assert hit is not None
            score, phash, raw = hit
            assert score >= SIMILARITY_THRESHOLD
            assert phash == "hash1"
            assert raw == b'{"type":"message"}'

            other = np.zeros(384, dtype=np.float32)
            other[1] = 1.0  # orthogonal — cosine 0
            assert await _local_l2_lookup(other) is None
    finally:
        await store.stop()


async def test_local_l2_disabled_via_env(monkeypatch, arca_home):
    from arca import cache as cache_mod

    monkeypatch.setenv("ARCA_LOCAL_L2", "false")
    get_settings.cache_clear()
    try:
        with patch("arca.cache._get_sqlite_fallback") as fb:
            assert await cache_mod._local_l2_lookup(np.zeros(384, dtype=np.float32)) is None
            assert not fb.called
    finally:
        get_settings.cache_clear()


# ---------------------------------------------------------------------------
# Non-streaming responses are cached (AUDIT P1-4, proxy tee)
# ---------------------------------------------------------------------------
async def test_json_message_response_triggers_post_hook():
    from arca.proxy import _noop_post, _noop_pre, app, register_hooks

    message_body = json.dumps({
        "type": "message", "role": "assistant",
        "content": [{"type": "text", "text": "cached!"}],
        "usage": {"input_tokens": 5, "output_tokens": 7},
    }).encode()

    class JsonTransport(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request):
            return httpx.Response(200, content=message_body,
                                  headers={"content-type": "application/json"})

    received: list[bytes] = []

    async def capture_post(request, raw: bytes) -> None:
        received.append(raw)

    client = httpx.AsyncClient(transport=JsonTransport())
    app.state.client = client
    register_hooks(post=capture_post)
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as tc:
            resp = await tc.post("/v1/messages", content=b'{"stream": false}',
                                 headers={"x-api-key": "sk-test"})
        assert resp.status_code == 200
        assert received and json.loads(received[0])["type"] == "message"
    finally:
        register_hooks(pre=_noop_pre, post=_noop_post)
        await client.aclose()


async def test_non_message_json_not_cached():
    from arca.proxy import _noop_post, _noop_pre, app, register_hooks

    class ErrTransport(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request):
            return httpx.Response(200, content=b'{"ok": true}',
                                  headers={"content-type": "application/json"})

    received: list[bytes] = []

    async def capture_post(request, raw: bytes) -> None:
        received.append(raw)

    client = httpx.AsyncClient(transport=ErrTransport())
    app.state.client = client
    register_hooks(post=capture_post)
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as tc:
            await tc.post("/v1/messages", content=b"{}", headers={"x-api-key": "sk"})
        assert not received
    finally:
        register_hooks(pre=_noop_pre, post=_noop_post)
        await client.aclose()


# ---------------------------------------------------------------------------
# Envelope-dilution fix: embed conversation content, not canonical JSON
# ---------------------------------------------------------------------------
def test_embedding_text_strips_envelope():
    from arca.normalizer import canonicalize, embedding_text

    body = json.dumps({
        "model": "claude-sonnet-4-6",
        "system": "Be terse.",
        "messages": [{"role": "user", "content": "what is the capital of france"}],
    }).encode()
    text = embedding_text(canonicalize(body))
    assert "capital of france" in text
    assert "Be terse." in text
    assert "{" not in text  # no JSON envelope tokens reach the embedder


def test_embedding_text_malformed_returns_input():
    from arca.normalizer import embedding_text

    assert embedding_text("not json") == "not json"


# ---------------------------------------------------------------------------
# Model-family guard: a haiku response must never satisfy an opus request
# ---------------------------------------------------------------------------
async def test_local_l2_rejects_cross_model_hit(arca_home):
    from arca.cache import _local_l2_lookup
    from arca.fallback import SQLiteFallback

    store = SQLiteFallback()
    await store.start()
    vec = np.zeros(384, dtype=np.float32)
    vec[0] = 1.0
    await store.enqueue({
        "id": "r1", "prompt_hash": "h1", "prompt_text": "{}",
        "embedding": vec.tolist(), "response_json": "{}",
        "model": "claude-haiku-4-5", "cost_usd": 0.0,
    })
    try:
        with patch("arca.cache._get_sqlite_fallback", return_value=store):
            assert await _local_l2_lookup(vec, "claude-opus-4-8") is None
            hit = await _local_l2_lookup(vec, "claude-haiku-4-5")
            assert hit is not None
    finally:
        await store.stop()


async def test_vs_l2_rejects_cross_model_hit():
    from arca.cache import _l2_lookup

    rows = [["h1", '{"type":"message"}', "claude-haiku-4-5", 0.99]]
    with patch("arca.cache._get_vs_index") as m:
        m.return_value.similarity_search.return_value = {"result": {"data_array": rows}}
        vec = np.zeros(384, dtype=np.float32)
        assert await _l2_lookup(vec, "claude-opus-4-8") is None
        hit = await _l2_lookup(vec, "claude-haiku-4-5")
        assert hit is not None and hit[1] == "h1"
