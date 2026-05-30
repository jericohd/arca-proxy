"""Tests for GET /stats and GET /health endpoints added for CLI-03 and CLI-04."""
from __future__ import annotations

import pytest
import httpx
from httpx import ASGITransport

from arca.proxy import app


@pytest.fixture
def client_factory():
    """Build an httpx.AsyncClient wired to the FastAPI app in-process."""
    def _make() -> httpx.AsyncClient:
        transport = ASGITransport(app=app)
        return httpx.AsyncClient(transport=transport, base_url="http://test")
    return _make


@pytest.mark.asyncio
async def test_health_endpoint(client_factory):
    async with client_factory() as c:
        r = await c.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


@pytest.mark.asyncio
async def test_stats_endpoint_empty(client_factory):
    app.state.metrics_accumulator = {}
    async with client_factory() as c:
        r = await c.get("/stats")
    data = r.json()
    assert r.status_code == 200
    assert data["total_calls"] == 0
    assert data["cache_hits"] == 0
    assert data["cache_misses"] == 0
    assert data["hit_rate_pct"] == 0.0
    assert data["cost_saved_usd"] == 0.0


@pytest.mark.asyncio
async def test_stats_endpoint_populated(client_factory):
    app.state.metrics_accumulator = {
        "total_calls": 10,
        "hit_count": 7,
        "cost_saved_usd_total": 0.42,
    }
    async with client_factory() as c:
        r = await c.get("/stats")
    data = r.json()
    assert data["total_calls"] == 10
    assert data["cache_hits"] == 7
    assert data["cache_misses"] == 3
    assert data["hit_rate_pct"] == 70.0
    assert data["cost_saved_usd"] == 0.42


@pytest.mark.asyncio
async def test_stats_endpoint_missing_state(client_factory):
    # Forcibly remove the attribute if present
    if hasattr(app.state, "metrics_accumulator"):
        delattr(app.state, "metrics_accumulator")
    async with client_factory() as c:
        r = await c.get("/stats")
    assert r.status_code == 200
    data = r.json()
    assert data["total_calls"] == 0
    assert data["hit_rate_pct"] == 0.0


@pytest.mark.asyncio
async def test_stats_hit_rate_rounded(client_factory):
    app.state.metrics_accumulator = {"total_calls": 3, "hit_count": 2}
    async with client_factory() as c:
        r = await c.get("/stats")
    assert r.json()["hit_rate_pct"] == 66.7
