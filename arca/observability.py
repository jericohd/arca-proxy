"""OBS-01..04 observability surfaces.

OBS-01 (this module, Plan 04-01):
  - extract_tokens(raw_sse_bytes) -> (input_tokens, output_tokens)
  - calculate_cost(model, in_tok, out_tok) -> USD
  - log_usage_event(...) fire-and-forget INSERT into the usage_log Delta table
  - _insert_usage_log_sync(...) sync helper run under app.state.sql_lock

OBS-02 / 04 wiring stubs:
  - app.state.tail_queue (asyncio.Queue)     — populated here, consumed in OBS-04
  - app.state.metrics_accumulator (dict)     — populated here, flushed in OBS-02

See .planning/phases/04-observability-cost-analytics/04-RESEARCH.md.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import statistics
import time
import uuid
from typing import Optional

import structlog

from arca.config import get_settings
from arca.proxy import app

_log = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
def _usage_table() -> str:
    """Fully-qualified usage_log table name from ARCA_CATALOG/ARCA_SCHEMA env."""
    return get_settings().usage_table

# OBS-02 — MLflow session tracking.
# Tracking URI "databricks" reads DATABRICKS_HOST + DATABRICKS_TOKEN from env.
MLFLOW_TRACKING_URI = "databricks"


def _mlflow_experiment_path() -> str:
    """Return the MLflow experiment path.

    Priority: ARCA_MLFLOW_EXPERIMENT env var → /Users/{current_user}/arca via
    Databricks SDK → /arca (workspace-level fallback).
    """
    env = os.environ.get("ARCA_MLFLOW_EXPERIMENT")
    if env:
        return env
    try:
        from databricks.sdk import WorkspaceClient
        email = WorkspaceClient().current_user.me().user_name
        return f"/Users/{email}/arca"
    except Exception:
        return "/arca"

# Anthropic pricing (USD per 1M tokens). Verified 2026-04-20.
# Keys are MODEL-FAMILY prefixes; matched via startswith() after
# lowercasing and replacing "." with "-" in the incoming model string.
MODEL_COSTS_PER_MTOK: dict[str, tuple[float, float]] = {
    # Current generation (model ids: claude-<family>-<major>-<minor>)
    "claude-fable-5":    (10.0, 50.0),
    "claude-opus-4-8":   (5.0,  25.0),
    "claude-opus-4-7":   (5.0,  25.0),
    "claude-opus-4-6":   (5.0,  25.0),
    "claude-opus-4-5":   (5.0,  25.0),
    "claude-opus-4-1":   (15.0, 75.0),
    "claude-opus-4":     (15.0, 75.0),   # claude-opus-4-0 / claude-opus-4-20250514
    "claude-sonnet-4":   (3.0,  15.0),   # sonnet 4.0 through 4.6 share rates
    "claude-haiku-4":    (1.0,   5.0),   # haiku 4.5
    # Legacy generation (model ids: claude-<major>-<minor>-<family>)
    "claude-3-7-sonnet": (3.0,  15.0),
    "claude-3-5-sonnet": (3.0,  15.0),
    "claude-3-5-haiku":  (0.80,  4.0),
    "claude-3-sonnet":   (3.0,  15.0),
    "claude-3-haiku":    (0.25,  1.25),
    "claude-3-opus":     (15.0, 75.0),
    "default":           (3.0,  15.0),
}

_DATA_LINE = re.compile(rb'^data: (.+)$', re.MULTILINE)


# ---------------------------------------------------------------------------
# Token extraction
# ---------------------------------------------------------------------------
def extract_tokens(raw: bytes) -> tuple[int, int]:
    """Extract (input_tokens, output_tokens) from a buffered Anthropic SSE byte stream.

    - input_tokens: from the first ``message_start`` event's ``message.usage.input_tokens``.
    - output_tokens: from the last ``message_delta`` event's ``usage.output_tokens``
      (Anthropic sends cumulative counts; last one wins).
    - Malformed ``data:`` lines are silently skipped — never raise on bad input.
    - Missing events default to 0.
    """
    input_tokens = 0
    output_tokens = 0
    stripped = raw.lstrip()
    if stripped.startswith(b"{"):
        # Non-streaming /v1/messages body: usage lives on the message itself.
        try:
            obj = json.loads(stripped)
            usage = (obj.get("usage") or {}) if isinstance(obj, dict) else {}
            return int(usage.get("input_tokens") or 0), int(usage.get("output_tokens") or 0)
        except (ValueError, TypeError):
            return 0, 0
    for m in _DATA_LINE.finditer(raw):
        payload = m.group(1)
        try:
            obj = json.loads(payload)
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue
        etype = obj.get("type")
        if etype == "message_start":
            usage = (obj.get("message") or {}).get("usage") or {}
            val = usage.get("input_tokens", 0) or 0
            try:
                input_tokens = int(val)
            except (TypeError, ValueError):
                input_tokens = 0
        elif etype == "message_delta":
            usage = obj.get("usage") or {}
            val = usage.get("output_tokens")
            if val is not None:
                try:
                    output_tokens = int(val)
                except (TypeError, ValueError):
                    pass
    return input_tokens, output_tokens


# ---------------------------------------------------------------------------
# Cost calculation
# ---------------------------------------------------------------------------
def calculate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Deterministic USD cost for a (model, input, output) triple.

    Matches model family by prefix (case-insensitive, dots normalized to dashes)
    against MODEL_COSTS_PER_MTOK. Falls back to the "default" tier.
    """
    model_normalized = (model or "").lower().replace(".", "-")
    in_rate, out_rate = MODEL_COSTS_PER_MTOK["default"]
    # Longest prefix wins: "claude-opus-4-5" must match its own rate, not
    # the shorter "claude-opus-4" (a different price point).
    for prefix in sorted(
        (k for k in MODEL_COSTS_PER_MTOK if k != "default"), key=len, reverse=True
    ):
        if model_normalized.startswith(prefix):
            in_rate, out_rate = MODEL_COSTS_PER_MTOK[prefix]
            break
    return (input_tokens * in_rate + output_tokens * out_rate) / 1_000_000


def _compute_costs(
    cache_hit: bool, model: str, input_tokens: int, output_tokens: int
) -> tuple[float, float]:
    """Return (cost_usd, cost_saved_usd) for a call.

    cost_saved_usd == cost_usd on a hit (you paid $0 and would have paid cost_usd)
    cost_saved_usd == 0.0 on a miss (you paid full price)
    """
    cost_usd = calculate_cost(model, input_tokens, output_tokens)
    cost_saved_usd = cost_usd if cache_hit else 0.0
    return cost_usd, cost_saved_usd


def _extract_model(canonical: str) -> Optional[str]:
    """Extract the ``model`` field from a canonical JSON request body.

    Lives here (not in arca.cache) so both the cache and the observability
    layer can reuse it without a circular import.
    """
    try:
        return json.loads(canonical).get("model")
    except Exception:
        return None


# ---------------------------------------------------------------------------
# log_usage_event — the OBS-01 entry point
# ---------------------------------------------------------------------------
async def log_usage_event(
    cache_hit: bool,
    model: str,
    input_tokens: int,
    output_tokens: int,
    cost_usd: float,
    cost_saved_usd: float,
    latency_ms: int,
    similarity_score: Optional[float],
) -> str:
    """Fire-and-forget write to the usage_log Delta table.

    Returns the row_id (uuid4) the row will be written with. Never blocks
    the caller on the DB write (uses asyncio.create_task + asyncio.to_thread).

    Graceful when ``app.state.sql`` is ``None`` (Databricks env absent):
    logs a warning, skips the INSERT, still returns a row_id.

    Also pushes the event into:
      - ``app.state.tail_queue``          (OBS-04 live tail)
      - ``app.state.metrics_accumulator`` (OBS-02 MLflow session)
    if those attributes are set on app.state.
    """
    row_id = str(uuid.uuid4())
    session_id = getattr(app.state, "session_id", None)

    # Push to tail queue (OBS-04) if present — drop on full (prefer recency of
    # new events is complex; simplest "drop-new" is fine for a demo).
    tail_queue = getattr(app.state, "tail_queue", None)
    if tail_queue is not None:
        event = {
            "cache_hit": cache_hit,
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_usd": cost_usd,
            "cost_saved_usd": cost_saved_usd,
            "latency_ms": latency_ms,
            "similarity_score": similarity_score,
            "ts": time.time(),
        }
        try:
            tail_queue.put_nowait(event)
        except asyncio.QueueFull:
            pass

    # Accumulate for MLflow session metrics (OBS-02).
    acc = getattr(app.state, "metrics_accumulator", None)
    if acc is not None:
        acc["total_calls"] = acc.get("total_calls", 0) + 1
        if cache_hit:
            acc["hit_count"] = acc.get("hit_count", 0) + 1
        acc["cost_usd_total"] = acc.get("cost_usd_total", 0.0) + cost_usd
        acc["cost_saved_usd_total"] = acc.get("cost_saved_usd_total", 0.0) + cost_saved_usd
        latencies = acc.setdefault("latencies_ms", [])
        latencies.append(latency_ms)
        if len(latencies) > 10_000:  # bound memory for long-lived sessions
            del latencies[: len(latencies) - 10_000]

    conn = getattr(app.state, "sql", None)
    if conn is None:
        _log.info("usage_log_skipped_no_sql", row_id=row_id)
        return row_id

    task = asyncio.create_task(
        asyncio.to_thread(
            _insert_usage_log_sync,
            row_id,
            session_id,
            cache_hit,
            model,
            input_tokens,
            output_tokens,
            cost_usd,
            cost_saved_usd,
            latency_ms,
            similarity_score,
        )
    )

    def _on_done(t: asyncio.Task) -> None:
        exc = t.exception()
        if exc:
            _log.warning("usage_log_write_failed", err=str(exc), row_id=row_id)

    task.add_done_callback(_on_done)
    return row_id


# ---------------------------------------------------------------------------
# Sync INSERT — runs inside asyncio.to_thread, guarded by app.state.sql_lock
# ---------------------------------------------------------------------------
def _insert_usage_log_sync(
    row_id: str,
    session_id: Optional[str],
    cache_hit: bool,
    model: str,
    input_tokens: int,
    output_tokens: int,
    cost_usd: float,
    cost_saved_usd: float,
    latency_ms: int,
    similarity_score: Optional[float],
) -> None:
    """Single-row INSERT into the configured usage_log table.

    Serialized by ``app.state.sql_lock`` (threading.Lock — databricks-sql-connector
    is NOT thread-safe). Silently returns if ``app.state.sql`` is None.
    """
    conn = getattr(app.state, "sql", None)
    if conn is None:
        return
    lock = app.state.sql_lock
    # Pitfall 5: threading.Lock — use `with lock:`, never `async with`
    with lock:
        cur = conn.cursor()
        cur.execute(
            f"""INSERT INTO {_usage_table()}
                (id, session_id, cache_hit, model,
                 input_tokens, output_tokens,
                 cost_usd, cost_saved_usd,
                 latency_ms, similarity_score, created_at)
                VALUES (:id, :session_id, :cache_hit, :model,
                        :input_tokens, :output_tokens,
                        :cost_usd, :cost_saved_usd,
                        :latency_ms, :similarity_score, current_timestamp())""",
            {
                "id": row_id,
                "session_id": session_id,
                "cache_hit": cache_hit,
                "model": model,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost_usd": cost_usd,
                "cost_saved_usd": cost_saved_usd,
                "latency_ms": latency_ms,
                "similarity_score": similarity_score,
            },
        )
        cur.close()


# ---------------------------------------------------------------------------
# OBS-02 — session lifecycle + metric flush
# ---------------------------------------------------------------------------
def _p95(latencies_ms: list[int]) -> float:
    """Return the 95th percentile latency in ms.

    - Empty list → 0.0 (no samples yet is distinct from "fast")
    - Single sample → that sample IS its own p95 (quantiles requires ≥2 points)
    - ≥2 samples → 19th vigintile from ``statistics.quantiles(..., n=20)``
    """
    if not latencies_ms:
        return 0.0
    if len(latencies_ms) == 1:
        return float(latencies_ms[0])
    return float(statistics.quantiles(latencies_ms, n=20)[18])


async def start_session(session_id: str) -> Optional[str]:
    """Create one MLflow run with ``run_name = session_id`` and return its run_id.

    The run is ended immediately; subsequent metric writes use
    ``MlflowClient().log_metric(run_id, ...)`` which is always safe and avoids
    the nested-run pitfall (RESEARCH.md Pitfall 5).

    Graceful degrade: returns ``None`` if Databricks/MLflow is unreachable or
    credentials are missing. Never raises.
    """
    def _start() -> str:
        import mlflow

        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(_mlflow_experiment_path())
        run = mlflow.start_run(run_name=session_id)
        run_id = run.info.run_id
        mlflow.end_run()
        return run_id

    try:
        return await asyncio.to_thread(_start)
    except Exception as exc:  # noqa: BLE001
        _log.warning("mlflow_start_failed", err=str(exc), session_id=session_id)
        return None


async def flush_session_metrics(run_id: Optional[str], accumulator: dict) -> None:
    """Flush accumulator metrics (cost_usd, hit_rate, latency_p95, total_calls)
    to the MLflow run identified by ``run_id``.

    No-op when ``run_id`` is None (MLflow unavailable at startup). Never raises —
    network or auth errors log a warning and return.
    """
    if run_id is None:
        return

    total_calls = accumulator.get("total_calls", 0)
    hit_count = accumulator.get("hit_count", 0)
    cost_usd_total = accumulator.get("cost_usd_total", 0.0)
    latencies_ms = list(accumulator.get("latencies_ms", []))
    hit_rate = (hit_count / total_calls) if total_calls else 0.0
    p95 = _p95(latencies_ms)

    def _flush() -> None:
        from mlflow.tracking import MlflowClient

        client = MlflowClient()
        client.log_metric(run_id, "cost_usd", float(cost_usd_total))
        client.log_metric(run_id, "hit_rate", float(hit_rate))
        client.log_metric(run_id, "latency_p95", float(p95))
        client.log_metric(run_id, "total_calls", float(total_calls))

    try:
        await asyncio.to_thread(_flush)
    except Exception as exc:  # noqa: BLE001
        _log.warning("mlflow_flush_failed", err=str(exc), run_id=run_id)


async def end_session(run_id: Optional[str]) -> None:
    """Terminate the MLflow run. No-op when ``run_id`` is None."""
    if run_id is None:
        return

    def _end() -> None:
        from mlflow.tracking import MlflowClient

        MlflowClient().set_terminated(run_id, status="FINISHED")

    try:
        await asyncio.to_thread(_end)
    except Exception as exc:  # noqa: BLE001
        _log.warning("mlflow_end_failed", err=str(exc), run_id=run_id)
