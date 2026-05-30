"""Phase 3 cache tests — RED until arca/cache.py is implemented."""
from __future__ import annotations

import asyncio
import json
import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Imports below will raise ModuleNotFoundError until Wave 1 lands arca/cache.py
from arca.cache import (  # noqa: E402
    L1_MAX_ENTRIES,
    SIMILARITY_THRESHOLD,
    _build_hit_response,
    _cache_enabled,
    _l1,
    _l1_get,
    _l1_put,
    _l2_lookup,
    _post_response,
    _pre_request,
    _wants_stream,
    _write_back,
)


# -------------------- L1 LRU (CACHE-02) --------------------

def test_l1_hit_moves_to_end():
    _l1.clear()
    _l1_put("a", b"A")
    _l1_put("b", b"B")
    _l1_put("c", b"C")
    assert _l1_get("a") == b"A"
    # 'a' just used → 'b' is now the oldest
    keys = list(_l1.keys())
    assert keys[0] == "b"
    assert keys[-1] == "a"


def test_l1_size_cap():
    _l1.clear()
    for i in range(L1_MAX_ENTRIES + 10):
        _l1_put(f"k{i}", b"v")
    assert len(_l1) == L1_MAX_ENTRIES


def test_l1_miss_returns_none():
    _l1.clear()
    assert _l1_get("nope") is None


def test_l1_key_is_prompt_hash():
    """Pre-request MUST derive L1 key via prompt_hash(canonicalize(body))."""
    from arca.normalizer import canonicalize, prompt_hash
    body = b'{"model":"claude-3-5-haiku-20241022","messages":[{"role":"user","content":"hi"}]}'
    expected = prompt_hash(canonicalize(body))
    _l1.clear()
    _l1_put(expected, b"cached-sse-bytes")
    assert _l1_get(expected) == b"cached-sse-bytes"


def test_l1_latency_under_5ms():
    """CACHE-02 SC: p95 L1 hit < 5 ms (we assert max < 5 ms over 1000 ops)."""
    _l1.clear()
    _l1_put("k", b"v" * 5000)
    durations = []
    for _ in range(1000):
        t0 = time.perf_counter()
        _l1_get("k")
        durations.append((time.perf_counter() - t0) * 1000)
    assert max(durations) < 5.0, f"L1 max latency {max(durations):.3f} ms exceeds 5 ms"


# -------------------- L2 Vector Search (CACHE-01, CACHE-03) --------------------

async def test_l2_hit_above_threshold():
    fake_response = {
        "result": {
            "data_array": [
                ["abc123", '{"content":[{"type":"text","text":"cached"}]}', 0.95]
            ]
        }
    }
    with patch("arca.cache._get_vs_index") as m:
        m.return_value.similarity_search.return_value = fake_response
        vec = np.zeros(384, dtype=np.float32)
        hit = await _l2_lookup(vec)
        assert hit is not None
        score, ph, raw = hit
        assert score == pytest.approx(0.95)
        assert ph == "abc123"
        assert b"cached" in raw


async def test_l2_miss_below_threshold():
    fake_response = {"result": {"data_array": [["abc123", "{}", 0.80]]}}
    with patch("arca.cache._get_vs_index") as m:
        m.return_value.similarity_search.return_value = fake_response
        vec = np.zeros(384, dtype=np.float32)
        hit = await _l2_lookup(vec)
        assert hit is None


async def test_l2_empty_result_returns_none():
    with patch("arca.cache._get_vs_index") as m:
        m.return_value.similarity_search.return_value = {"result": {"data_array": []}}
        vec = np.zeros(384, dtype=np.float32)
        assert await _l2_lookup(vec) is None


async def test_l2_uses_threshold():
    """VS similarity_search MUST be called with score_threshold=SIMILARITY_THRESHOLD (0.95)."""
    with patch("arca.cache._get_vs_index") as m:
        m.return_value.similarity_search.return_value = {"result": {"data_array": []}}
        vec = np.zeros(384, dtype=np.float32)
        await _l2_lookup(vec)
        kwargs = m.return_value.similarity_search.call_args.kwargs
        assert kwargs.get("score_threshold") == SIMILARITY_THRESHOLD
        assert kwargs["score_threshold"] == 0.95


async def test_l2_passes_list_not_ndarray():
    """VS SDK wants list[float], not np.ndarray — avoid JSON serialization error."""
    with patch("arca.cache._get_vs_index") as m:
        m.return_value.similarity_search.return_value = {"result": {"data_array": []}}
        vec = np.zeros(384, dtype=np.float32)
        await _l2_lookup(vec)
        qv = m.return_value.similarity_search.call_args.kwargs["query_vector"]
        assert isinstance(qv, list)
        assert all(isinstance(x, float) for x in qv)


async def test_l2_timeout_returns_none(monkeypatch):
    """ARCA_L2_DEADLINE_S expiry → None (not raised)."""
    monkeypatch.setenv("ARCA_L2_DEADLINE_S", "0.01")
    # reload not required — deadline read per-call
    async def slow(*a, **kw):
        await asyncio.sleep(0.5)
        return {"result": {"data_array": []}}
    with patch("arca.cache._vs_similarity_search_sync", side_effect=lambda v: asyncio.run(slow())):
        vec = np.zeros(384, dtype=np.float32)
        result = await _l2_lookup(vec)
        assert result is None


# -------------------- Pre-request hook (CACHE-05) --------------------

async def test_pre_request_skips_non_messages():
    req = MagicMock()
    req.method = "POST"
    req.url.path = "/v1/models"
    assert await _pre_request(req, {}, b"{}") is None


async def test_pre_request_skips_non_post():
    req = MagicMock()
    req.method = "GET"
    req.url.path = "/v1/messages"
    assert await _pre_request(req, {}, b"{}") is None


async def test_pre_request_malformed_body_returns_none():
    req = MagicMock()
    req.method = "POST"
    req.url.path = "/v1/messages"
    assert await _pre_request(req, {}, b"not json") is None


# -------------------- CACHE-07 feature flag --------------------

def test_cache_enabled_default_true(monkeypatch):
    monkeypatch.delenv("ARCA_CACHE_ENABLED", raising=False)
    assert _cache_enabled() is True


def test_cache_enabled_false_values(monkeypatch):
    for val in ("false", "False", "0", "no", "off", ""):
        monkeypatch.setenv("ARCA_CACHE_ENABLED", val)
        assert _cache_enabled() is False, f"{val!r} should disable cache"


def test_cache_enabled_true_values(monkeypatch):
    for val in ("true", "TRUE", "1", "yes", "on"):
        monkeypatch.setenv("ARCA_CACHE_ENABLED", val)
        assert _cache_enabled() is True, f"{val!r} should enable cache"


async def test_disabled_cache_returns_none(monkeypatch):
    monkeypatch.setenv("ARCA_CACHE_ENABLED", "false")
    req = MagicMock()
    req.method = "POST"
    req.url.path = "/v1/messages"
    assert await _pre_request(req, {}, b'{"messages":[]}') is None


async def test_env_var_read_per_call(monkeypatch):
    """Toggle without restart — each call re-reads the env var."""
    req = MagicMock()
    req.method = "POST"
    req.url.path = "/v1/messages"
    monkeypatch.setenv("ARCA_CACHE_ENABLED", "true")
    assert _cache_enabled() is True
    monkeypatch.setenv("ARCA_CACHE_ENABLED", "false")
    assert _cache_enabled() is False


# -------------------- Post-response hook / write-back (CACHE-04) --------------------

async def test_post_hook_skips_when_no_state():
    """If pre-hook didn't stash canonical/vector on request.state, skip."""
    req = MagicMock(spec=[])
    req.state = MagicMock(spec=[])  # no cache_canonical attribute
    # Should not raise
    await _post_response(req, b"some sse bytes")


async def test_write_back_parallel():
    """_write_back runs Delta + VS via asyncio.gather(..., return_exceptions=True)."""
    with patch("arca.cache._insert_cache_store_sync") as delta, \
         patch("arca.cache._vs_upsert_sync") as vs:
        vec = np.zeros(384, dtype=np.float32)
        await _write_back("hash123", '{"model":"x"}', vec, b"sse-bytes")
        assert delta.called
        assert vs.called


async def test_delta_failure_falls_back_sqlite():
    """Delta exception → SQLiteFallback.enqueue called."""
    with patch("arca.cache._insert_cache_store_sync", side_effect=RuntimeError("delta down")), \
         patch("arca.cache._vs_upsert_sync"), \
         patch("arca.cache._get_sqlite_fallback") as fb:
        fake = MagicMock()
        async def _aenq(*a, **kw):
            return None
        fake.enqueue = MagicMock(side_effect=_aenq)
        fb.return_value = fake
        vec = np.zeros(384, dtype=np.float32)
        await _write_back("hash123", '{"model":"x"}', vec, b"sse-bytes")
        assert fake.enqueue.called


async def test_post_hook_populates_l1():
    """After post-hook runs, L1 should contain the new key."""
    _l1.clear()
    req = MagicMock()
    req.state.cache_canonical = '{"model":"claude","messages":[]}'
    req.state.cache_prompt_hash = "deadbeef"
    req.state.cache_vector = np.zeros(384, dtype=np.float32)
    with patch("arca.cache._write_back") as wb:
        async def _noop(*a, **kw):
            return None
        wb.side_effect = _noop
        await _post_response(req, b"complete-sse")
    assert _l1_get("deadbeef") == b"complete-sse"


# -------------------- End-to-end latency (CACHE-05) --------------------

async def test_e2e_l1_under_100ms(monkeypatch):
    """L1 hit end-to-end via _pre_request < 100 ms."""
    from arca.normalizer import canonicalize, prompt_hash
    body = b'{"model":"claude-3-5-haiku-20241022","messages":[{"role":"user","content":"hi"}]}'
    key = prompt_hash(canonicalize(body))
    _l1.clear()
    _l1_put(key, b'data: {"type":"message_stop"}\n\n')

    req = MagicMock()
    req.method = "POST"
    req.url.path = "/v1/messages"
    req._body = body

    durations = []
    for _ in range(20):
        t0 = time.perf_counter()
        resp = await _pre_request(req, {}, body)
        durations.append((time.perf_counter() - t0) * 1000)
        assert resp is not None
    p95 = sorted(durations)[int(len(durations) * 0.95) - 1]
    assert p95 < 100.0, f"L1 e2e p95 {p95:.2f} ms exceeds 100 ms"
