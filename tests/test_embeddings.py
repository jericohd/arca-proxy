"""RED test suite for arca/embeddings.py — covers PKG-02 (singleton + async + determinism + CPU-only timing).

These tests lock the public API surface (embed, warm_up, _get_model, EMBEDDING_DIM,
EMBEDDING_MODEL) and every assertion encodes a requirement from PKG-02. They MUST fail
with ImportError/ModuleNotFoundError against the current repo because arca/embeddings.py
does not exist yet.

Plan 02-03 will implement the module; once that lands, these tests flip GREEN.

Notes:
- pytest-asyncio asyncio_mode=auto is set in pyproject.toml → no @pytest.mark.asyncio.
- The session-scoped autouse `_warmup_once` fixture amortizes the ~1-3s SentenceTransformer
  cold start over the whole session; individual tests do not re-warm.
- `embed()` is async (always wraps the blocking encode() in asyncio.to_thread);
  `_get_model()` is sync (the singleton accessor).
"""
from __future__ import annotations

import asyncio
import time

import numpy as np
import pytest

from arca.embeddings import EMBEDDING_DIM, EMBEDDING_MODEL, embed, warm_up, _get_model


@pytest.fixture(scope="session", autouse=True)
async def _warmup_once():
    """Load the model once per test session to amortize the ~1-3s cold start.

    Runs warm_up() in a worker thread via asyncio.to_thread so the event loop is not
    blocked during fixture setup. autouse=True means every test benefits without having
    to import the fixture.
    """
    await asyncio.to_thread(warm_up)
    yield


async def test_embedding_shape_and_dtype():
    """Vector is shape (384,), dtype float32; EMBEDDING_DIM guards the 384 invariant (DB-02)."""
    vec = await embed("hello world")
    assert vec.shape == (EMBEDDING_DIM,)
    assert vec.dtype == np.float32
    assert EMBEDDING_DIM == 384


async def test_embedding_is_l2_normalized():
    """L2 norm ≈ 1.0 — required so cosine similarity collapses to dot product downstream."""
    vec = await embed("hello world")
    norm = float(np.linalg.norm(vec))
    assert abs(norm - 1.0) < 1e-5


async def test_embedding_deterministic_across_calls():
    """Two calls with the same input yield BYTE-identical vectors (single-input, CPU, fp32)."""
    a = await embed("the quick brown fox")
    b = await embed("the quick brown fox")
    assert np.array_equal(a, b)


async def test_different_inputs_produce_different_vectors():
    """Distinct English strings must not collapse to cosine ≥ 0.99 (would break semantic search)."""
    a = await embed("the quick brown fox")
    b = await embed("something completely different")
    # Unit-length vectors → dot product is cosine similarity.
    assert float(np.dot(a, b)) < 0.99


async def test_singleton_identity():
    """_get_model() returns the same object reference on every call — one model per process."""
    m1 = _get_model()
    m2 = _get_model()
    assert m1 is m2


async def test_warm_call_is_fast():
    """After warmup, a subsequent embed() completes in <250ms (PKG-02 target <100ms + CI slack)."""
    # Explicit prime (session fixture already warmed, but be defensive against cold-start jitter)
    await embed("warm-prime")
    t0 = time.monotonic()
    await embed("measure this call")
    elapsed_ms = (time.monotonic() - t0) * 1000
    assert elapsed_ms < 250, f"Warm embed took {elapsed_ms:.1f}ms (expected <250ms)"


async def test_model_name_is_minilm_l6_v2():
    """EMBEDDING_MODEL is locked to all-MiniLM-L6-v2 — DB-02 index is 384-dim against this model."""
    assert EMBEDDING_MODEL == "sentence-transformers/all-MiniLM-L6-v2"
