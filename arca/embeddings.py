"""Local CPU-only embedding of canonical prompts via sentence-transformers.

Public API:
    async def embed(text: str) -> np.ndarray   # 384-dim float32, L2-normalized
    def warm_up() -> None                      # eager load; safe to call from lifespan
    def _get_model() -> SentenceTransformer    # thread-safe lazy singleton accessor
    EMBEDDING_MODEL                            # locked to all-MiniLM-L6-v2
    EMBEDDING_DIM                              # 384

The singleton model is loaded at most once per process. Thread-safe init via
``threading.Lock`` with double-checked locking. The async ``embed()`` wrapper
uses ``asyncio.to_thread`` because sentence-transformers ``encode()`` is
synchronous and CPU-bound — calling it directly from an async handler would
block the event loop (see RESEARCH.md Pitfall 2).

device="cpu" is explicit (defense-in-depth for PKG-02) even though the
installed torch wheel is CPU-only.
"""
from __future__ import annotations

import asyncio
import os
import threading
import time
from typing import Optional

import numpy as np
import structlog
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Constants (locked — DB-02 commits the 384-dim index to this exact model)
# ---------------------------------------------------------------------------
EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM: int = 384

_log = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Singleton state — module-global, guarded by _model_lock for init only
# ---------------------------------------------------------------------------
_model: Optional[SentenceTransformer] = None
_model_lock = threading.Lock()


def _get_model() -> SentenceTransformer:
    """Thread-safe lazy singleton. First call loads; subsequent calls return cached.

    Uses double-checked locking: the fast path avoids acquiring the lock once the
    model has been initialized, so hot-path calls from many threads never contend.
    """
    global _model
    # Fast path — no lock
    if _model is not None:
        return _model
    with _model_lock:
        # Slow path — re-check under lock (another thread may have raced us)
        if _model is None:
            t0 = time.monotonic()
            cache_folder = os.environ.get("ARCA_MODEL_CACHE_DIR")  # None -> SENTENCE_TRANSFORMERS_HOME / HF defaults
            _model = SentenceTransformer(
                EMBEDDING_MODEL,
                device="cpu",                # explicit — no CUDA probe (PKG-02 defense-in-depth)
                cache_folder=cache_folder,
            )
            _log.info(
                "embedding_model_loaded",
                model=EMBEDDING_MODEL,
                dim=EMBEDDING_DIM,
                load_ms=round((time.monotonic() - t0) * 1000, 1),
            )
        return _model


def warm_up() -> None:
    """Force the model to load. Call from the proxy lifespan so the first real
    request doesn't pay the 1-3 s cold-start cost.

    Safe to call more than once — subsequent calls are no-ops (singleton).
    """
    _get_model()


async def embed(text: str) -> np.ndarray:
    """Embed a canonical prompt string. Returns a 384-dim float32 numpy array.

    L2-normalized (norm = 1.0) so downstream cosine similarity collapses to a
    dot product, matching Databricks Vector Search's default cosine metric.

    Blocking ``model.encode`` is offloaded to the default asyncio thread pool
    so the event loop is never blocked (see RESEARCH.md Pitfall 2).
    """
    model = _get_model()
    vec = await asyncio.to_thread(
        model.encode,
        text,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    # sentence-transformers returns float32 by default; assert defensively
    if vec.dtype != np.float32:
        vec = vec.astype(np.float32)
    return vec
