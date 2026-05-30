"""tests/test_proxy.py — Phase 1 proxy test suite.

All tests are RED until arca/proxy.py is implemented (Plan 02).
asyncio_mode=auto is set in pyproject.toml — no @pytest.mark.asyncio needed.
"""
from __future__ import annotations

import importlib
import time
from collections.abc import AsyncIterator
from typing import Iterator

import httpx
import pytest
from httpx import ASGITransport, AsyncClient


# ---------------------------------------------------------------------------
# MockTransport — records upstream requests, returns scripted responses
# ---------------------------------------------------------------------------
class MockTransport(httpx.AsyncBaseTransport):
    """Intercepts httpx requests; records them; returns scripted responses.

    Pass a list of httpx.Response objects to the constructor. Each call to
    handle_async_request pops the next response. When the list is exhausted
    every subsequent call returns a 200 OK with content b'{"mocked":true}'.
    """

    def __init__(self, responses: list[httpx.Response] | None = None) -> None:
        self.requests: list[httpx.Request] = []
        self._responses: Iterator[httpx.Response] = iter(responses or [])

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        try:
            return next(self._responses)
        except StopIteration:
            return httpx.Response(200, content=b'{"mocked":true}')


# SSE response body that ends with message_stop — used for streaming tests
_SSE_BODY = (
    b"event: message_start\r\ndata: {}\r\n\r\n"
    b"event: content_block_delta\r\ndata: {}\r\n\r\n"
    b"event: message_stop\r\ndata: {}\r\n\r\n"
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
async def mock_transport() -> MockTransport:
    return MockTransport()


@pytest.fixture
async def proxy_app(mock_transport: MockTransport):
    """FastAPI app with a mock httpx client injected into app.state.

    Bypasses lifespan so we control the client directly.
    Does NOT start uvicorn — tests use ASGITransport.
    """
    from arca.proxy import app  # noqa: PLC0415

    client = httpx.AsyncClient(transport=mock_transport)
    app.state.client = client
    yield app
    await client.aclose()


@pytest.fixture
async def test_client(proxy_app) -> AsyncClient:
    """Async httpx client pointed at the proxy via in-process ASGI transport."""
    async with AsyncClient(
        transport=ASGITransport(app=proxy_app),
        base_url="http://test",
    ) as client:
        yield client


@pytest.fixture
async def sse_app():
    """proxy_app variant whose mock upstream returns a valid SSE stream."""
    from arca.proxy import app  # noqa: PLC0415

    transport = MockTransport(
        responses=[
            httpx.Response(
                200,
                content=_SSE_BODY,
                headers={"content-type": "text/event-stream"},
            )
        ]
    )
    client = httpx.AsyncClient(transport=transport)
    app.state.client = client
    yield app, transport
    await client.aclose()


# ---------------------------------------------------------------------------
# PRXY-01: App structure and lifespan
# ---------------------------------------------------------------------------
async def test_lifespan_client(proxy_app):
    """app.state.client is an AsyncClient (injected by fixture)."""
    assert isinstance(proxy_app.state.client, httpx.AsyncClient)


async def test_port_default():
    """Port 8082 is the default in the proxy module (ARCA_PORT override)."""
    import inspect
    import arca.proxy as proxy_mod
    source = inspect.getsource(proxy_mod)
    # The default port must be 8082
    assert "8082" in source


# ---------------------------------------------------------------------------
# PRXY-02: Header forwarding and filtering
# ---------------------------------------------------------------------------
async def test_header_forwarding(test_client, mock_transport):
    """Application-level headers are forwarded to upstream."""
    await test_client.post(
        "/v1/messages",
        content=b'{"model":"claude-3-5-haiku-20241022"}',
        headers={
            "x-api-key": "sk-test-key",
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
    )
    assert mock_transport.requests, "No upstream request was made"
    upstream_headers = {k.lower(): v for k, v in mock_transport.requests[0].headers.items()}
    assert upstream_headers.get("x-api-key") == "sk-test-key"
    assert upstream_headers.get("anthropic-version") == "2023-06-01"


async def test_hop_by_hop_stripped(test_client, mock_transport):
    """Hop-by-hop headers must NOT appear in the upstream request."""
    hop_headers = {
        "connection": "keep-alive",
        "transfer-encoding": "chunked",
        "te": "trailers",
        "trailers": "x-trailer",
        "upgrade": "websocket",
        "keep-alive": "timeout=5",
        "x-api-key": "sk-test",
    }
    await test_client.post("/v1/messages", content=b"{}", headers=hop_headers)
    assert mock_transport.requests
    upstream_headers = {k.lower() for k in mock_transport.requests[0].headers}
    for hop in ("connection", "transfer-encoding", "te", "trailers", "upgrade", "keep-alive"):
        assert hop not in upstream_headers, f"Hop-by-hop header '{hop}' was forwarded"


async def test_host_stripped(test_client, mock_transport):
    """The client's original host must NOT reach the upstream — httpx sets host from the target URL.

    After the live-integration fix (ea71791), host is NOT in HOP_BY_HOP_UPSTREAM so httpx sets
    it correctly to api.anthropic.com. The test asserts the correct behavior: host is either
    absent or set to the Anthropic target — never to the proxy client's address (testclient).
    """
    await test_client.post("/v1/messages", content=b"{}", headers={"x-api-key": "sk-test"})
    assert mock_transport.requests
    upstream_headers = {k.lower(): v for k, v in mock_transport.requests[0].headers.items()}
    if "host" in upstream_headers:
        assert "anthropic" in upstream_headers["host"], (
            f"host header forwarded client address instead of Anthropic: {upstream_headers['host']!r}"
        )


async def test_xapikey_forwarded(test_client, mock_transport):
    """x-api-key is forwarded verbatim — Claude Code authentication passthrough."""
    await test_client.post(
        "/v1/messages",
        content=b"{}",
        headers={"x-api-key": "sk-ant-api03-verysecret"},
    )
    assert mock_transport.requests
    upstream_headers = {k.lower(): v for k, v in mock_transport.requests[0].headers.items()}
    assert upstream_headers.get("x-api-key") == "sk-ant-api03-verysecret"


async def test_accept_encoding_stripped(test_client, mock_transport):
    """accept-encoding is stripped to force plaintext from Anthropic (per CONTEXT.md)."""
    await test_client.post(
        "/v1/messages",
        content=b"{}",
        headers={"x-api-key": "sk-test", "accept-encoding": "gzip, br"},
    )
    assert mock_transport.requests
    upstream_headers = {k.lower() for k in mock_transport.requests[0].headers}
    assert "accept-encoding" not in upstream_headers


# ---------------------------------------------------------------------------
# PRXY-03: SSE tee pattern
# ---------------------------------------------------------------------------
async def test_non_streaming_passthrough(test_client):
    """Non-streaming POST returns the upstream body unchanged with status 200."""
    response = await test_client.post(
        "/v1/messages",
        content=b'{"model":"claude-3-5-haiku-20241022"}',
        headers={"x-api-key": "sk-test", "content-type": "application/json"},
    )
    assert response.status_code == 200
    assert response.content == b'{"mocked":true}'


async def test_sse_buffer_on_message_stop(sse_app):
    """Post-response hook receives the complete SSE bytes after message_stop."""
    app, _transport = sse_app
    from arca.proxy import register_hooks  # noqa: PLC0415

    received: list[bytes] = []

    async def capture_post(request, raw: bytes) -> None:
        received.append(raw)

    register_hooks(post=capture_post)

    try:
        # Use AsyncClient+ASGITransport — buffers full response but hook still fires after message_stop
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            resp = await client.post(
                "/v1/messages",
                content=b"{}",
                headers={"x-api-key": "sk-test"},
            )
        # Hook must have been called with the full SSE body
        assert received, "Post-response hook was not called"
        full = b"".join(received)
        assert b"event: message_stop" in full
        assert resp.status_code == 200
    finally:
        from arca.proxy import _noop_pre, _noop_post  # noqa: PLC0415
        register_hooks(pre=_noop_pre, post=_noop_post)


async def test_disconnect_discards_buffer(proxy_app):
    """If stream ends without message_stop, post-response hook is NOT called."""
    from arca.proxy import register_hooks, _noop_pre, _noop_post  # noqa: PLC0415

    no_stop_transport = MockTransport(
        responses=[
            httpx.Response(
                200,
                content=b"event: content_block_delta\r\ndata: {}\r\n\r\n",
                headers={"content-type": "text/event-stream"},
            )
        ]
    )
    no_stop_client = httpx.AsyncClient(transport=no_stop_transport)
    proxy_app.state.client = no_stop_client

    called: list[bool] = []

    async def record_post(request, raw: bytes) -> None:
        called.append(True)

    register_hooks(post=record_post)
    try:
        async with AsyncClient(
            transport=ASGITransport(app=proxy_app),
            base_url="http://test",
        ) as tc:
            await tc.post(
                "/v1/messages",
                content=b"{}",
                headers={"x-api-key": "sk-test"},
            )
        assert not called, "Post-response hook must NOT be called when message_stop is absent"
    finally:
        await no_stop_client.aclose()
        register_hooks(pre=_noop_pre, post=_noop_post)


# ---------------------------------------------------------------------------
# PRXY-04: Circuit breaker (unit tests — no HTTP layer)
# ---------------------------------------------------------------------------
async def test_circuit_opens_after_threshold():
    """CB trips to OPEN after failure_threshold failures within the window."""
    from arca.proxy import CircuitBreaker, CBState  # noqa: PLC0415

    cb = CircuitBreaker(failure_threshold=3, window_seconds=30.0, reset_timeout=60.0)

    async def failing():
        raise RuntimeError("simulated downstream failure")

    for _ in range(3):
        with pytest.raises(RuntimeError):
            await cb.call(failing)

    assert cb.state == CBState.OPEN


async def test_circuit_stays_open():
    """A call to an OPEN circuit raises CircuitOpenError immediately."""
    from arca.proxy import CircuitBreaker, CBState, CircuitOpenError  # noqa: PLC0415

    cb = CircuitBreaker(failure_threshold=3, window_seconds=30.0, reset_timeout=60.0)

    async def failing():
        raise RuntimeError("downstream failure")

    for _ in range(3):
        with pytest.raises(RuntimeError):
            await cb.call(failing)

    assert cb.state == CBState.OPEN

    with pytest.raises(CircuitOpenError):
        await cb.call(failing)


async def test_anthropic_errors_propagate(proxy_app):
    """Anthropic upstream errors return 5xx and do NOT trip the circuit breaker (PRXY-04).

    The CB wraps Databricks calls only — Anthropic errors must propagate transparently.
    """
    from arca.proxy import circuit_breaker, CBState  # noqa: PLC0415

    class ErrorTransport(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("upstream unreachable")

    error_client = httpx.AsyncClient(transport=ErrorTransport())
    proxy_app.state.client = error_client
    try:
        async with AsyncClient(
            transport=ASGITransport(app=proxy_app),
            base_url="http://test",
        ) as tc:
            resp = await tc.post("/v1/messages", content=b"{}", headers={"x-api-key": "sk-test"})
        assert resp.status_code >= 500, f"Expected 5xx from upstream error, got {resp.status_code}"
        assert circuit_breaker.state == CBState.CLOSED, (
            "Circuit breaker must not be tripped by Anthropic upstream errors"
        )
    finally:
        await error_client.aclose()


# ---------------------------------------------------------------------------
# PRXY-05: Hook points
# ---------------------------------------------------------------------------
async def test_pre_hook_shortcircuit(proxy_app, mock_transport):
    """Pre-request hook can return a Response to short-circuit upstream forwarding."""
    from arca.proxy import register_hooks, _noop_pre, _noop_post  # noqa: PLC0415
    from fastapi.responses import JSONResponse

    async def cache_hit_hook(request, headers, body):
        return JSONResponse({"source": "cache"}, status_code=200)

    register_hooks(pre=cache_hit_hook)
    try:
        async with AsyncClient(
            transport=ASGITransport(app=proxy_app),
            base_url="http://test",
        ) as client:
            response = await client.post(
                "/v1/messages",
                content=b"{}",
                headers={"x-api-key": "sk-test"},
            )
        assert response.status_code == 200
        assert response.json() == {"source": "cache"}
        # Upstream must NOT have been called
        assert not mock_transport.requests, "Upstream was called despite cache hit"
    finally:
        register_hooks(pre=_noop_pre, post=_noop_post)


# ---------------------------------------------------------------------------
# PRXY-06: Bind address
# ---------------------------------------------------------------------------
async def test_bind_address():
    """Proxy module hardcodes host='127.0.0.1' — never binds to 0.0.0.0."""
    import inspect
    import arca.proxy as proxy_mod
    source = inspect.getsource(proxy_mod)
    assert "127.0.0.1" in source, "127.0.0.1 bind address not found in proxy module"
    assert "0.0.0.0" not in source, "0.0.0.0 must never appear in proxy module"
