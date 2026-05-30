"""arca/proxy.py — Phase 1: transparent pass-through proxy.

Set ANTHROPIC_BASE_URL=http://localhost:8082 and Claude Code routes through this
proxy. Phase 1 is pure pass-through with <10ms added latency. Phase 3 will plug
its semantic cache into the pre-request and post-response hook points exposed
by this module (see register_hooks).

Key extension points:
    _pre_request_hook   — called before forwarding; may short-circuit with a Response
    _post_response_hook — called after a complete SSE stream (message_stop seen)
    circuit_breaker     — module-level singleton; Phase 3 wraps Databricks calls with it

Bind address is hardcoded to 127.0.0.1 (PRXY-06 — never exposed to LAN).
Port defaults to 8082; ARCA_PORT environment variable overrides.
"""
from __future__ import annotations

import asyncio
import json
import os
import threading
import time
import uuid
from contextlib import asynccontextmanager
from enum import Enum, auto
from typing import AsyncGenerator, Awaitable, Callable, TypeVar

import httpx
from databricks import sql as _dbsql
from fastapi import FastAPI, Request
from fastapi.responses import Response, StreamingResponse

from arca.fallback import SQLiteFallback

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ANTHROPIC_BASE = "https://api.anthropic.com"

T = TypeVar("T")

# Used by _forward_headers to strip incoming client headers before forwarding.
HOP_BY_HOP_REQUEST = frozenset({
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
    "host",
    "content-encoding",
    "content-length",
    "accept-encoding",
})

# Used to strip headers from the httpx-built upstream request.
# Must NOT include host or content-length — httpx computes those correctly
# from the URL and body; stripping them breaks HTTP/1.1.
# accept-encoding IS stripped so Anthropic returns plain-text SSE — required
# for _stream_and_buffer to detect b"event: message_stop" in raw bytes.
_HOP_BY_HOP_UPSTREAM = frozenset({
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
    "accept-encoding",
})

HOP_BY_HOP_RESPONSE = frozenset({
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
})


# ---------------------------------------------------------------------------
# Circuit Breaker (PRXY-04) — wraps Databricks calls in Phase 3, not Anthropic
# ---------------------------------------------------------------------------
class CBState(Enum):
    CLOSED = auto()
    OPEN = auto()
    HALF_OPEN = auto()


class CircuitOpenError(Exception):
    """Raised when the circuit breaker is in OPEN state."""


class CircuitBreaker:
    """Async circuit breaker for wrapping Databricks calls (PRXY-04).

    Not used in Phase 1 — instantiated as a module-level singleton so Phase 3
    can call ``circuit_breaker.call(databricks_fn, ...)`` without import changes.
    """

    def __init__(
        self,
        failure_threshold: int = 3,
        window_seconds: float = 30.0,
        reset_timeout: float = 60.0,
    ) -> None:
        self._threshold = failure_threshold
        self._window = window_seconds
        self._reset_timeout = reset_timeout
        self._state = CBState.CLOSED
        self._failure_times: list[float] = []
        self._opened_at: float = 0.0
        self._lock = asyncio.Lock()

    @property
    def state(self) -> CBState:
        return self._state

    async def call(self, fn: Callable[..., Awaitable[T]], *args, **kwargs) -> T:
        """Execute ``fn``, tripping the breaker on repeated failures.

        Raises ``CircuitOpenError`` if the breaker is OPEN and the reset
        timeout has not elapsed. Otherwise propagates the original exception
        on failure (after recording it) or returns the result on success.
        """
        async with self._lock:
            now = time.monotonic()
            if self._state == CBState.OPEN:
                if now - self._opened_at >= self._reset_timeout:
                    self._state = CBState.HALF_OPEN
                else:
                    raise CircuitOpenError(
                        "Circuit breaker is OPEN — Databricks call skipped"
                    )

        try:
            result = await fn(*args, **kwargs)
        except Exception:
            async with self._lock:
                now = time.monotonic()
                # Prune failures outside the rolling window
                self._failure_times = [
                    t for t in self._failure_times if now - t < self._window
                ]
                self._failure_times.append(now)
                if len(self._failure_times) >= self._threshold:
                    self._state = CBState.OPEN
                    self._opened_at = now
                    self._failure_times.clear()
            raise
        else:
            async with self._lock:
                if self._state == CBState.HALF_OPEN:
                    # Successful probe — reset to closed
                    self._state = CBState.CLOSED
                    self._failure_times.clear()
            return result


# Module-level singleton — Phase 3 wraps its Databricks calls with this
circuit_breaker = CircuitBreaker(
    failure_threshold=3,
    window_seconds=30.0,
    reset_timeout=60.0,
)


# ---------------------------------------------------------------------------
# Phase 3 Hook Points (PRXY-05) — no-ops in Phase 1
# ---------------------------------------------------------------------------
PreRequestHook = Callable[[Request, dict, bytes], Awaitable["Response | None"]]
PostResponseHook = Callable[[Request, bytes], Awaitable[None]]


async def _noop_pre(request: Request, headers: dict, body: bytes) -> Response | None:
    return None


async def _noop_post(request: Request, raw: bytes) -> None:
    return None


_pre_request_hook: PreRequestHook = _noop_pre
_post_response_hook: PostResponseHook = _noop_post


def register_hooks(
    pre: PreRequestHook | None = None,
    post: PostResponseHook | None = None,
) -> None:
    """Replace the module-level hooks. Phase 3 calls this at import time to
    wire in the semantic cache lookup (pre) and cache write (post).
    """
    global _pre_request_hook, _post_response_hook
    if pre is not None:
        _pre_request_hook = pre
    if post is not None:
        _post_response_hook = post


# ---------------------------------------------------------------------------
# Header helpers
# ---------------------------------------------------------------------------
def _forward_headers(request: Request) -> dict[str, str]:
    """Return forwardable request headers, stripping hop-by-hop entries and
    any header names listed in the incoming ``Connection`` value.
    """
    connection_extras = {
        h.strip().lower()
        for h in request.headers.get("connection", "").split(",")
        if h.strip()
    }
    skip = HOP_BY_HOP_REQUEST | connection_extras
    return {k: v for k, v in request.headers.items() if k.lower() not in skip}


def _filter_response_headers(headers: httpx.Headers) -> dict[str, str]:
    """Return upstream response headers safe to forward to the client."""
    return {k: v for k, v in headers.items() if k.lower() not in HOP_BY_HOP_RESPONSE}


# ---------------------------------------------------------------------------
# Lifespan (PRXY-01)
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.client = httpx.AsyncClient(
        base_url=ANTHROPIC_BASE,
        timeout=httpx.Timeout(None, connect=10.0),  # no read timeout for SSE streams
    )
    from arca.embeddings import warm_up  # local import keeps arca.proxy cheap to import (tests never trigger model download)
    asyncio.create_task(asyncio.to_thread(warm_up))

    # Phase 3 wiring: importing arca.cache registers pre/post hooks at module level.
    import arca.cache  # noqa: F401 — import for side effect (register_hooks)

    # Shared Databricks resources for arca.cache's write-back path.
    app.state.sql = None
    app.state.sql_lock = threading.Lock()  # threading.Lock — sync driver (Pitfall 5)
    app.state.sqlite_fallback = SQLiteFallback()
    await app.state.sqlite_fallback.start()
    try:
        host = os.environ["DATABRICKS_HOST"].replace("https://", "").replace("http://", "")
        app.state.sql = await asyncio.to_thread(
            _dbsql.connect,
            server_hostname=host,
            http_path=os.environ["DATABRICKS_HTTP_PATH"],
            access_token=os.environ["DATABRICKS_TOKEN"],
        )
    except KeyError:
        # PRXY-05: Databricks env missing → app.state.sql stays None;
        # arca.cache write-back logs + skips Delta INSERT; queries fall through to Anthropic.
        pass
    except Exception:
        # Any other connection failure: degrade gracefully, proxy still serves pass-through.
        pass

    # Phase 4 OBS-02: observability session lifecycle
    # Local import (matches `warm_up` pattern). We also import the module so
    # monkeypatching arca.observability.{start_session,...} in tests takes effect.
    from arca.observability import start_session, flush_session_metrics, end_session  # noqa: F401
    from arca import observability as _obs
    app.state.session_id = str(uuid.uuid4())
    app.state.metrics_accumulator = {
        "total_calls": 0,
        "hit_count": 0,
        "cost_usd_total": 0.0,
        "cost_saved_usd_total": 0.0,
        "latencies_ms": [],
    }
    app.state.mlflow_run_id = await _obs.start_session(app.state.session_id)

    # Phase 4 OBS-04: live tail queue. Created INSIDE lifespan (after loop starts,
    # Pitfall 1 avoided). log_usage_event pushes events here when present.
    app.state.tail_queue = asyncio.Queue(maxsize=1000)

    async def _periodic_flush() -> None:
        try:
            while True:
                await asyncio.sleep(30.0)
                if app.state.mlflow_run_id:
                    await _obs.flush_session_metrics(
                        app.state.mlflow_run_id,
                        app.state.metrics_accumulator,
                    )
        except asyncio.CancelledError:
            return

    app.state.flush_task = asyncio.create_task(_periodic_flush())

    try:
        yield
    finally:
        # Phase 4 OBS-02: stop periodic flusher, final flush, end MLflow run.
        flush_task = getattr(app.state, "flush_task", None)
        if flush_task is not None:
            flush_task.cancel()
            try:
                await flush_task
            except asyncio.CancelledError:
                pass
        run_id = getattr(app.state, "mlflow_run_id", None)
        if run_id:
            try:
                await _obs.flush_session_metrics(run_id, app.state.metrics_accumulator)
            except Exception:
                pass
            try:
                await _obs.end_session(run_id)
            except Exception:
                pass

        # Phase 4 OBS-04: drain tail queue so pending tasks don't warn on shutdown.
        tq = getattr(app.state, "tail_queue", None)
        if tq is not None:
            while not tq.empty():
                try:
                    tq.get_nowait()
                except asyncio.QueueEmpty:
                    break

        await app.state.client.aclose()
        if app.state.sql is not None:
            try:
                await asyncio.to_thread(app.state.sql.close)
            except Exception:
                pass
        await app.state.sqlite_fallback.stop()


app = FastAPI(lifespan=lifespan)


# ---------------------------------------------------------------------------
# SSE tee (PRXY-03)
# ---------------------------------------------------------------------------
async def _stream_and_buffer(
    upstream: httpx.Response,
    request: Request,
) -> AsyncGenerator[bytes, None]:
    """Tee: yield every raw chunk to client AND accumulate in a buffer.

    Invokes ``_post_response_hook`` with the full bytes only when the stream
    completes with an ``event: message_stop`` SSE event. Discards the buffer
    on client disconnect or any exception (never cache incomplete responses).
    """
    buffer: list[bytes] = []
    complete = False
    try:
        if upstream.is_stream_consumed:
            # Mock transports (used in tests) return pre-buffered responses
            # whose raw stream is already consumed. Yield the materialised
            # content as a single chunk so the tee pattern still applies.
            chunk = upstream.content
            if chunk:
                yield chunk
                buffer.append(chunk)
                if b"event: message_stop" in chunk:
                    complete = True
        else:
            async for chunk in upstream.aiter_raw():
                yield chunk
                buffer.append(chunk)
                if b"event: message_stop" in chunk:
                    complete = True
                    break
        if complete and buffer:
            await _post_response_hook(request, b"".join(buffer))
    except Exception:
        # Discard partial buffer — never cache incomplete responses
        pass
    finally:
        await upstream.aclose()


# ---------------------------------------------------------------------------
# OBS-04 — live tail SSE endpoint
# ---------------------------------------------------------------------------
@app.get("/tail")
async def tail(request: Request):
    """OBS-04: SSE stream of live cache events for `arca tail`.

    Emits `data: {json}\\n\\n` per usage event and `: keep-alive\\n\\n` every
    ~1s of inactivity so long-lived HTTP connections don't time out.
    """
    queue: asyncio.Queue = request.app.state.tail_queue

    async def event_gen():
        while True:
            if await request.is_disconnected():
                break
            try:
                event = await asyncio.wait_for(queue.get(), timeout=1.0)
                yield f"data: {json.dumps(event)}\n\n"
            except asyncio.TimeoutError:
                yield ": keep-alive\n\n"

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/stats")
async def get_stats() -> dict:
    """Return session metrics from in-memory accumulator (OBS-02 data).

    Consumed by `arca stats` CLI command (CLI-03). Falls back to Delta SQL
    when proxy is not running (handled CLI-side, not here).
    """
    acc = getattr(app.state, "metrics_accumulator", {}) or {}
    total = int(acc.get("total_calls", 0))
    hits = int(acc.get("hit_count", 0))
    misses = total - hits
    hit_rate = round(hits / total * 100, 1) if total > 0 else 0.0
    return {
        "total_calls": total,
        "cache_hits": hits,
        "cache_misses": misses,
        "hit_rate_pct": hit_rate,
        "cost_saved_usd": float(acc.get("cost_saved_usd_total", 0.0)),
    }


@app.get("/health")
async def get_health() -> dict:
    """Liveness probe consumed by `arca doctor` routing check (CLI-04)."""
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Catch-all proxy route (PRXY-02)
# ---------------------------------------------------------------------------
@app.api_route(
    "/v1/{path:path}",
    methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
)
async def proxy(path: str, request: Request) -> Response:
    headers = _forward_headers(request)
    body = await request.body()

    # Phase 3 pre-request hook (cache lookup). If it returns a Response, short-circuit.
    cache_hit = await _pre_request_hook(request, headers, body)
    if cache_hit is not None:
        return cache_hit

    client: httpx.AsyncClient = request.app.state.client
    # Use an absolute URL so the request works regardless of whether the
    # injected client has ``base_url`` set (tests inject a client without one).
    upstream_url = f"{ANTHROPIC_BASE}/v1/{path}"
    upstream_req = client.build_request(
        method=request.method,
        url=upstream_url,
        headers=headers,
        content=body,
        params=dict(request.query_params),
    )
    # Strip connection-scope headers from the httpx-built request.
    # Use _HOP_BY_HOP_UPSTREAM (not HOP_BY_HOP_REQUEST) so that host,
    # content-length, and content-encoding — which httpx computes correctly
    # from the target URL and body — are left intact.
    for _hop in _HOP_BY_HOP_UPSTREAM:
        if _hop in upstream_req.headers:
            del upstream_req.headers[_hop]

    try:
        upstream_resp = await client.send(upstream_req, stream=True)
    except httpx.HTTPError as exc:
        # Anthropic upstream error — propagate as 502 Bad Gateway.
        # The circuit breaker is NOT engaged here (CB wraps Databricks only).
        return Response(
            content=f"Upstream error: {exc}".encode("utf-8"),
            status_code=502,
            media_type="text/plain",
        )

    return StreamingResponse(
        _stream_and_buffer(upstream_resp, request),
        status_code=upstream_resp.status_code,
        headers=_filter_response_headers(upstream_resp.headers),
        media_type=upstream_resp.headers.get(
            "content-type", "application/octet-stream"
        ),
    )


# ---------------------------------------------------------------------------
# Entry point (PRXY-06)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("ARCA_PORT", "8082"))
    uvicorn.run(
        "arca.proxy:app",
        host="127.0.0.1",  # PRXY-06: never bind to LAN
        port=port,
        log_level="info",
    )
