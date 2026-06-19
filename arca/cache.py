"""Two-tier semantic cache plugged into arca.proxy via register_hooks().

L1: process-local OrderedDict LRU, keyed on SHA-256 of the canonical prompt.
L2: Databricks Vector Search (cosine >= ARCA_SIMILARITY_THRESHOLD, default 0.95),
    with a local brute-force fallback over the SQLite store when Databricks is absent.

Registration happens at import time — importing this module registers the
pre/post hooks on arca.proxy. Import it once from arca.proxy's lifespan.
"""
from __future__ import annotations

import asyncio
import json
import os
import threading
import time
from collections import OrderedDict
from typing import Any, Optional

import numpy as np
import structlog
from fastapi import Request
from fastapi.responses import Response

from arca.embeddings import embed
from arca.normalizer import canonicalize, embedding_text, prompt_hash
from arca.semantic_guard import is_safe_match
from arca.nli_verify import is_contradiction, nli_enabled
from arca.proxy import app, circuit_breaker, register_hooks  # `app` is the public FastAPI instance
from arca.cache_replay import message_json_to_sse, sse_to_message_json
from arca.config import get_settings
from arca.observability import (
    _compute_costs,
    _extract_model,
    extract_tokens,
    log_usage_event,
)

_log = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# All deployment names and tunables come from arca.config (ARCA_* env vars).
_settings = get_settings()
SIMILARITY_THRESHOLD = _settings.similarity_threshold
L1_MAX_ENTRIES = _settings.l1_max_entries
L2_DEADLINE_S = _settings.l2_deadline_s
CATALOG = _settings.catalog
SCHEMA = _settings.db_schema
CACHE_TABLE = _settings.cache_table
ENDPOINT = _settings.vs_endpoint
INDEX = _settings.vs_index

# ---------------------------------------------------------------------------
# Feature flag — reads env fresh every call (CACHE-07)
# ---------------------------------------------------------------------------

def _cache_enabled() -> bool:
    """Read ARCA_CACHE_ENABLED fresh on every lookup (CACHE-07).

    True-values: anything NOT in {"false","0","no","off",""} (case-insensitive, stripped).
    Default (env var unset): True.
    """
    val = os.environ.get("ARCA_CACHE_ENABLED", "true").strip().lower()
    return val not in {"false", "0", "no", "off", ""}


# ---------------------------------------------------------------------------
# L1 — OrderedDict LRU with a threading.Lock (CACHE-02)
# ---------------------------------------------------------------------------
# Why threading.Lock (not asyncio.Lock):
#   L1 operations are pure in-memory, O(1), no await points — never block the
#   event loop. threading.Lock is cheaper and is not pinned to an event loop.
_l1: "OrderedDict[str, bytes]" = OrderedDict()  # key = sha256 hex; value = raw SSE bytes
_l1_lock = threading.Lock()


def _l1_get(key: str) -> Optional[bytes]:
    with _l1_lock:
        val = _l1.get(key)
        if val is not None:
            _l1.move_to_end(key)  # LRU: mark recently used
        return val


def _l1_put(key: str, value: bytes) -> None:
    with _l1_lock:
        _l1[key] = value
        _l1.move_to_end(key)
        while len(_l1) > L1_MAX_ENTRIES:
            _l1.popitem(last=False)  # evict oldest


# ---------------------------------------------------------------------------
# L2 — Databricks Vector Search (CACHE-01 / CACHE-03)
# ---------------------------------------------------------------------------
_vs_index = None
_vs_index_lock = threading.Lock()
_vs_unavailable = False  # set after the first failed client init; reset on process restart


def _get_vs_index():
    """Lazy double-checked-locked singleton of VectorSearchClient().get_index(ENDPOINT, INDEX).

    Returns None if Databricks credentials are not configured (PRXY-05: degrade gracefully).
    The None sentinel prevents the circuit breaker from recording credential errors as
    Databricks operational failures — those are configuration issues, not service failures.
    """
    global _vs_index, _vs_unavailable
    if _vs_index is not None:
        return _vs_index
    if _vs_unavailable:
        return None
    with _vs_index_lock:
        if _vs_index is None and not _vs_unavailable:
            try:
                from databricks.vector_search.client import VectorSearchClient
                _vs_index = VectorSearchClient().get_index(
                    endpoint_name=ENDPOINT,
                    index_name=INDEX,
                )
            except Exception as exc:
                _vs_unavailable = True
                _log.warning("vs_index_unavailable_local_mode", err=str(exc), err_type=type(exc).__name__)
                return None
    return _vs_index


def _vs_similarity_search_sync(
    vec: list[float], model: Optional[str] = None, query_text: Optional[str] = None
) -> Optional[tuple[float, str, bytes]]:
    """Sync VS query. Returns (score, prompt_hash, response_bytes) on hit, None on miss.

    Response shape (Databricks VS SDK verified):
        {"result": {"data_array": [[col1, col2, ..., score], ...]}}

    We request columns=["prompt_hash","response_json","model"], so each row is:
        [prompt_hash, response_json, model, score] — score is ALWAYS the LAST
        element (Pitfall 2). ``model`` is parsed positionally only when the row
        is wide enough, so older 2-column indexes keep working.

    A hit must come from the SAME model family as the request: a response
    generated by haiku must never satisfy a request addressed to opus.
    """
    index = _get_vs_index()
    if index is None:
        return None  # Databricks not configured — degrade gracefully (PRXY-05)
    result = index.similarity_search(
        query_vector=vec,
        columns=["prompt_hash", "response_json", "model", "prompt_text"],
        num_results=5,  # over-fetch: the polarity guard may reject the top rows
        score_threshold=SIMILARITY_THRESHOLD,  # server-side threshold filter (SDK >= 0.57)
    )
    rows = (result or {}).get("result", {}).get("data_array") or []
    for row in rows:
        # Pitfall 2: score is the LAST element in the row
        score = row[-1]
        if not isinstance(score, (int, float)) or score < SIMILARITY_THRESHOLD:
            continue
        row_model = str(row[2]) if len(row) >= 4 and row[2] is not None else None
        if model and row_model and row_model != model:
            continue  # cross-model hit — reject
        # Polarity guard: reject candidates that flip meaning (encode/decode,
        # X->Y vs Y->X, negation). Required because cosine 0.90 lets these leak.
        if query_text is not None and len(row) >= 5 and row[3]:
            cand_text = embedding_text(str(row[3]))
            if not is_safe_match(query_text, cand_text):
                continue
            # Opt-in NLI stage (default off). Sync is fine here — this runs in a
            # worker thread (asyncio.to_thread), not on the event loop.
            if nli_enabled() and is_contradiction(query_text, cand_text):
                continue
        return float(score), str(row[0]), str(row[1]).encode("utf-8")
    return None


async def _local_l2_lookup(
    vec: np.ndarray, model: Optional[str] = None, query_text: Optional[str] = None
) -> Optional[tuple[float, str, bytes]]:
    """Brute-force cosine over the local SQLite store (ARCA_LOCAL_L2, default on).

    Embeddings are L2-normalized at encode time, so cosine reduces to a dot
    product. O(N * 384) per lookup over at most ``local_l2_max_rows`` rows --
    appropriate for a single developer's cache, not for a shared deployment
    (that is what Vector Search is for).
    """
    settings = get_settings()
    if not settings.local_l2:
        return None
    fallback = _get_sqlite_fallback()
    if fallback is None:
        return None
    try:
        rows = await fallback.fetch_all()
    except Exception as exc:
        _log.warning("local_l2_fetch_failed", err=str(exc))
        return None
    if not rows:
        return None
    candidates: list[tuple[float, dict]] = []
    for row in rows[-settings.local_l2_max_rows:]:
        emb = row.get("embedding")
        if not emb or len(emb) != vec.shape[0]:
            continue
        if model and row.get("model") and row["model"] != model:
            continue  # never serve a response generated by a different model
        score = float(np.dot(vec, np.asarray(emb, dtype=np.float32)))
        if score >= SIMILARITY_THRESHOLD:
            candidates.append((score, row))
    # Best score first, then apply the polarity guard — skip candidates that flip
    # meaning (encode/decode, X->Y vs Y->X, negation) and return the best safe one.
    candidates.sort(key=lambda c: c[0], reverse=True)
    for score, row in candidates:
        if query_text is not None:
            candidate_text = embedding_text(str(row.get("prompt_text", "")))
            if not is_safe_match(query_text, candidate_text):
                continue
            # Opt-in NLI stage (default off). Offload the sync CPU call so the
            # event loop is never blocked.
            if nli_enabled() and await asyncio.to_thread(
                is_contradiction, query_text, candidate_text
            ):
                continue
        return (
            score,
            str(row["prompt_hash"]),
            str(row["response_json"]).encode("utf-8"),
        )
    return None


async def _l2_lookup(
    vec: np.ndarray, model: Optional[str] = None, query_text: Optional[str] = None
) -> Optional[tuple[float, str, bytes]]:
    """Async L2 lookup — VS query off-loop through circuit breaker with deadline.

    When Databricks Vector Search is not configured, falls back to a local
    brute-force cosine search over the SQLite store so semantic caching still
    works standalone.

    Converts np.ndarray to list[float] before passing to VS SDK (Pitfall 3).
    Returns None on TimeoutError / CircuitOpenError / any exception.
    """
    if _get_vs_index() is None:
        return await _local_l2_lookup(vec, model, query_text)

    query_vec = vec.tolist()  # Pitfall 3: VS SDK wants list[float], not np.ndarray

    async def _call():
        return await asyncio.to_thread(_vs_similarity_search_sync, query_vec, model, query_text)

    try:
        # Deadline enforced INSIDE the breaker so a slow/hanging endpoint is
        # recorded as a failure and can trip the circuit (see CircuitBreaker.call).
        return await circuit_breaker.call(_call, timeout=L2_DEADLINE_S)
    except (asyncio.TimeoutError, TimeoutError):
        _log.warning("l2_timeout", deadline_s=L2_DEADLINE_S)
        return None
    except Exception as exc:
        # CircuitOpenError or any other Databricks failure → treat as miss
        _log.warning("l2_error", err=str(exc), err_type=type(exc).__name__)
        return None


# ---------------------------------------------------------------------------
# Hit response builder
# ---------------------------------------------------------------------------

def _wants_stream(body: bytes) -> bool:
    """Check if the client requested stream=true."""
    try:
        return bool(json.loads(body).get("stream"))
    except Exception:
        return False


def _build_hit_response(raw: bytes, body: bytes) -> Response:
    """Reconstruct a client-appropriate response from the stored buffer.

    The cache stores whatever upstream returned: raw SSE bytes for streaming
    misses, a JSON message body for non-streaming misses. Either stored form
    is replayed to either client mode via arca.cache_replay conversions.
    """
    wants_stream = _wants_stream(body)
    is_json = raw.lstrip().startswith(b"{")
    if wants_stream:
        return Response(
            content=message_json_to_sse(raw) if is_json else raw,
            media_type="text/event-stream",
            status_code=200,
            headers={"x-arca-cache": "hit"},
        )
    return Response(
        content=raw if is_json else sse_to_message_json(raw),
        media_type="application/json",
        status_code=200,
        headers={"x-arca-cache": "hit"},
    )


# ---------------------------------------------------------------------------
# Pre-request hook (CACHE-05 / CACHE-07)
# ---------------------------------------------------------------------------

async def _pre_request(request: Request, headers: dict, body: bytes) -> Optional[Response]:
    """L1 → L2 → None. Returns a Response on hit (short-circuits proxy), None on miss.

    Called by arca.proxy on every incoming request. Fast path (cache disabled
    or non-/v1/messages path) returns None in <50 µs.
    """
    if not _cache_enabled():  # CACHE-07: read env fresh every call
        return None
    # Only cache POST /v1/messages — everything else is pass-through
    if request.method != "POST" or not request.url.path.endswith("/messages"):
        return None

    t0 = time.monotonic()
    request.state.t0 = t0  # expose to _post_response for miss-latency calc
    try:
        canonical = canonicalize(body)
    except ValueError:
        # Malformed JSON — let Anthropic reject it
        return None
    key = prompt_hash(canonical)

    # --- L1 lookup ---
    raw = _l1_get(key)
    if raw is not None:
        elapsed_ms = (time.monotonic() - t0) * 1000
        _log.info("cache_hit", tier="L1", latency_ms=round(elapsed_ms, 2))

        # OBS-01: log L1 hit event
        model = _extract_model(canonical) or "unknown"
        input_tokens, output_tokens = extract_tokens(raw)
        cost_usd, cost_saved_usd = _compute_costs(
            cache_hit=True, model=model,
            input_tokens=input_tokens, output_tokens=output_tokens,
        )
        await log_usage_event(
            cache_hit=True, model=model,
            input_tokens=input_tokens, output_tokens=output_tokens,
            cost_usd=cost_usd, cost_saved_usd=cost_saved_usd,
            latency_ms=int(elapsed_ms), similarity_score=1.0,
        )
        return _build_hit_response(raw, body)

    # --- L2 lookup ---
    # Embed the conversation content, NOT the canonical JSON: the shared JSON
    # envelope inflates cosine similarity between unrelated prompts (measured:
    # unrelated questions crossed 0.95 when the envelope was embedded).
    try:
        query_text = embedding_text(canonical)
        vec = await embed(query_text)
    except Exception as exc:
        _log.warning("embed_failed", err=str(exc))
        return None

    request_model = _extract_model(canonical)
    hit = await _l2_lookup(vec, request_model, query_text)
    elapsed_ms = (time.monotonic() - t0) * 1000
    if hit is not None:
        score, found_hash, raw = hit
        # Populate L1 with the L2 hit so next exact repeat is <5ms
        _l1_put(key, raw)
        _log.info(
            "cache_hit",
            tier="L2",
            similarity_score=round(score, 4),
            latency_ms=round(elapsed_ms, 2),
        )

        # OBS-01: log L2 hit event
        model = _extract_model(canonical) or "unknown"
        input_tokens, output_tokens = extract_tokens(raw)
        cost_usd, cost_saved_usd = _compute_costs(
            cache_hit=True, model=model,
            input_tokens=input_tokens, output_tokens=output_tokens,
        )
        await log_usage_event(
            cache_hit=True, model=model,
            input_tokens=input_tokens, output_tokens=output_tokens,
            cost_usd=cost_usd, cost_saved_usd=cost_saved_usd,
            latency_ms=int(elapsed_ms), similarity_score=float(score),
        )
        return _build_hit_response(raw, body)

    _log.info("cache_miss", latency_ms=round(elapsed_ms, 2))
    # Stash canonical + vector on request.state so the post-hook can reuse them
    # without redoing the work. (request.state is the standard FastAPI carrier.)
    request.state.cache_canonical = canonical
    request.state.cache_prompt_hash = key
    request.state.cache_vector = vec
    return None


# ---------------------------------------------------------------------------
# SQLite fallback accessor (testable via patch)
# ---------------------------------------------------------------------------

def _get_sqlite_fallback():
    """Return the SQLiteFallback instance from app.state, or None if not seeded."""
    return getattr(app.state, "sqlite_fallback", None)


# ---------------------------------------------------------------------------
# Post-response hook + write-back (CACHE-04)
# ---------------------------------------------------------------------------

async def _post_response(request: Request, raw: bytes) -> None:
    """Fire-and-forget cache write. Fires only after message_stop (Phase 1 tee).

    Returns immediately; actual Delta + VS upserts run in the background.
    Never blocks the client response path.
    """
    if not _cache_enabled():
        return
    if not hasattr(request.state, "cache_canonical"):
        # Pre-hook decided not to cache (disabled / wrong path / bad JSON / no state)
        return

    canonical: str = request.state.cache_canonical
    key: str = request.state.cache_prompt_hash
    vec: np.ndarray = request.state.cache_vector

    # Populate L1 immediately (O(1), never blocks)
    _l1_put(key, raw)

    # OBS-01: log miss event (fire-and-forget INSERT)
    t0 = getattr(request.state, "t0", None)
    latency_ms = int((time.monotonic() - t0) * 1000) if t0 is not None else 0
    model = _extract_model(canonical) or "unknown"
    input_tokens, output_tokens = extract_tokens(raw)
    cost_usd, cost_saved_usd = _compute_costs(
        cache_hit=False, model=model,
        input_tokens=input_tokens, output_tokens=output_tokens,
    )
    await log_usage_event(
        cache_hit=False, model=model,
        input_tokens=input_tokens, output_tokens=output_tokens,
        cost_usd=cost_usd, cost_saved_usd=cost_saved_usd,
        latency_ms=latency_ms, similarity_score=None,
    )

    # Fire-and-forget background write to Delta + VS
    task = asyncio.create_task(_write_back(key, canonical, vec, raw))
    # Pitfall 4: log any unhandled exception from the background task
    task.add_done_callback(
        lambda t: t.exception() and _log.warning(
            "write_back_task_failed", err=str(t.exception())
        )
    )


async def _write_back(key: str, canonical: str, vec: np.ndarray, raw: bytes) -> None:
    """Parallel Delta + VS upsert. Falls back to SQLite on Delta failure.

    Uses asyncio.gather(..., return_exceptions=True) so one Databricks failure
    doesn't cancel the other (CACHE-04).
    """
    row_id = key  # prompt_hash IS the primary key (deterministic across retries)
    response_str = raw.decode("utf-8", errors="replace")
    embedding_list = vec.tolist()

    async def _enqueue_local() -> None:
        fallback = _get_sqlite_fallback()
        if fallback is not None:
            await fallback.enqueue({
                "id": row_id,
                "prompt_hash": key,
                "prompt_text": canonical,
                "embedding": embedding_list,
                "response_json": response_str,
                "model": _extract_model(canonical),
                "cost_usd": 0.0,
            })

    async def _delta_write():
        if getattr(app.state, "sql", None) is None:
            # Local mode: no Databricks connection. The SQLite store is the
            # primary store here (it also backs _local_l2_lookup).
            await _enqueue_local()
            return
        try:
            await asyncio.to_thread(
                _insert_cache_store_sync,
                row_id, key, canonical, embedding_list, response_str,
            )
        except Exception as exc:
            _log.warning("delta_write_failed_fallback_sqlite", err=str(exc))
            await _enqueue_local()

    async def _vs_upsert():
        try:
            await asyncio.to_thread(
                _vs_upsert_sync,
                row_id, key, canonical, embedding_list, response_str,
            )
        except Exception as exc:
            _log.warning("vs_upsert_failed", err=str(exc))

    results = await asyncio.gather(
        _delta_write(),
        _vs_upsert(),
        return_exceptions=True,  # one failure doesn't cancel the other
    )
    for r in results:
        if isinstance(r, Exception):
            _log.warning("write_back_partial_failure", err=str(r))


def _insert_cache_store_sync(
    row_id: str,
    prompt_hash_val: str,
    prompt_text: str,
    embedding: list[float],
    response_json: str,
) -> None:
    """Single-row INSERT into the configured cache_store table.

    Uses the long-lived SQL connection on app.state.sql (seeded by lifespan in Plan 03).
    Serialized by app.state.sql_lock (threading.Lock — connector is NOT thread-safe).

    Lock discipline (Pitfall 5): sql_lock is threading.Lock, NOT asyncio.Lock.
    Always acquire via `with app.state.sql_lock:` inside this sync function.
    If app.state.sql is None (Databricks env vars absent), skip and log.
    """
    conn = getattr(app.state, "sql", None)
    if conn is None:
        _log.warning("delta_write_skipped_no_sql_connection")
        return
    lock = app.state.sql_lock
    # Pitfall 5: threading.Lock — use `with lock:`, never `async with`
    with lock:
        cur = conn.cursor()
        # Named parameters (databricks-sql-connector 3.x native paramstyle).
        # The embedding array is bound as JSON and parsed server-side because
        # the connector does not support ARRAY-typed parameters directly.
        cur.execute(
            f"""INSERT INTO {CACHE_TABLE}
                (id, prompt_hash, prompt_text, embedding, response_json,
                 model, input_tokens, output_tokens, cost_usd, hit_count,
                 created_at, last_hit_at)
                VALUES (:id, :prompt_hash, :prompt_text,
                        from_json(:embedding, 'array<float>'), :response_json,
                        NULL, NULL, NULL, NULL, 0,
                        current_timestamp(), current_timestamp())""",
            {
                "id": row_id,
                "prompt_hash": prompt_hash_val,
                "prompt_text": prompt_text,
                "embedding": json.dumps(embedding),
                "response_json": response_json,
            },
        )
        cur.close()


def _vs_upsert_sync(
    row_id: str,
    prompt_hash_val: str,
    prompt_text: str,
    embedding: list[float],
    response_json: str,
) -> None:
    """Upsert one row into the VS index.

    Handles both Direct Access (index.upsert) and Delta Sync TRIGGERED (Pitfall 1):
    if upsert raises "Cannot upsert to DELTA_SYNC" or "DELTA_SYNC", fall back to
    index.sync() so the data reaches the index via the underlying Delta table.
    """
    index = _get_vs_index()
    if index is None:
        return  # Databricks VS not configured — local mode, nothing to upsert
    try:
        index.upsert([{
            "id": row_id,
            "prompt_hash": prompt_hash_val,
            "prompt_text": prompt_text,
            "embedding": embedding,
            "response_json": response_json,
            "model": _extract_model(prompt_text) or "unknown",
            "cost_usd": 0.0,
        }])
    except Exception as exc:
        msg = str(exc)
        if "DELTA_SYNC" in msg or "Cannot upsert" in msg.lower() or "delta_sync" in msg.lower():
            # Delta Sync index — data reaches index via underlying Delta table + sync()
            try:
                index.sync()
            except Exception:
                pass
        else:
            raise


# _extract_model lives in arca.observability (imported above) so cache + obs
# can share it without a circular import.


# ---------------------------------------------------------------------------
# Hook registration — happens at import time
# ---------------------------------------------------------------------------
register_hooks(pre=_pre_request, post=_post_response)
