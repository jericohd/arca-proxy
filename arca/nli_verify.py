"""Opt-in NLI contradiction verifier — a third L2 precision stage. DEFAULT OFF.

Enable with ``ARCA_NLI_VERIFY`` truthy. When off, ``nli_enabled()`` returns False
and the model is NEVER loaded — zero cost, the <50ms hit budget is untouched.
When on, a cross-encoder NLI model rejects candidate hits whose stored prompt
*contradicts* the query, catching numeric/quantity flips the deterministic guard
structurally cannot (e.g. "limit 10 rows" vs "limit 100 rows"). See
benchmarks/EVAL_NLI.md for the measured lift (~90% -> ~96% held-out precision).

Always stacked AFTER arca.semantic_guard (guard -> NLI). NLI alone is unreliable
on question/imperative prompts; it only helps as a second filter. Adds ~10ms per
candidate on CPU plus a ~330MB model. On Databricks, prefer running this as a
served model / Foundation Model API endpoint rather than local RAM.
"""
from __future__ import annotations

import os
import threading
import time
from typing import TYPE_CHECKING, Optional

import structlog

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder

_log = structlog.get_logger(__name__)

NLI_MODEL = "cross-encoder/nli-distilroberta-base"


def nli_enabled() -> bool:
    """Read ARCA_NLI_VERIFY fresh on every call (toggle without restart). Default OFF."""
    val = os.environ.get("ARCA_NLI_VERIFY", "false").strip().lower()
    return val not in {"false", "0", "no", "off", ""}


def _contradiction_threshold() -> float:
    try:
        return float(os.environ.get("ARCA_NLI_CONTRADICTION_THRESHOLD", "0.5"))
    except ValueError:
        return 0.5


_model: "Optional[CrossEncoder]" = None
_contra_idx: Optional[int] = None
_lock = threading.Lock()


def _get_model() -> "CrossEncoder":
    """Thread-safe lazy singleton — loaded only when NLI verification is enabled."""
    global _model, _contra_idx
    if _model is not None:
        return _model
    with _lock:
        if _model is None:
            from sentence_transformers import CrossEncoder  # lazy: keeps import cheap when off

            t0 = time.monotonic()
            m = CrossEncoder(NLI_MODEL, device="cpu")
            id2label = {k: str(v).lower() for k, v in m.model.config.id2label.items()}
            _contra_idx = next(k for k, v in id2label.items() if v == "contradiction")
            _model = m
            _log.info("nli_model_loaded", model=NLI_MODEL,
                      load_ms=round((time.monotonic() - t0) * 1000, 1))
    return _model


def is_contradiction(a: str, b: str) -> bool:
    """True if (a, b) is a contradiction in EITHER direction at/above the threshold.

    Sync + CPU-bound (~10ms). Callers on the event loop MUST offload to a thread.
    Never raises — on any model error, returns False (fail-open to the guard's
    decision; the deterministic guard already ran).
    """
    import numpy as np

    try:
        model = _get_model()
        logits = np.asarray(model.predict([(a, b), (b, a)]), dtype=np.float64)
        if logits.ndim == 1:
            logits = logits.reshape(1, -1)
        ex = np.exp(logits - logits.max(axis=-1, keepdims=True))
        probs = ex / ex.sum(axis=-1, keepdims=True)
        return float(probs[:, _contra_idx].max()) >= _contradiction_threshold()
    except Exception as exc:  # noqa: BLE001
        _log.warning("nli_verify_failed_fail_open", err=str(exc))
        return False


def warm_up() -> None:
    """Eager-load the model if NLI verification is enabled (call from lifespan)."""
    if nli_enabled():
        _get_model()
