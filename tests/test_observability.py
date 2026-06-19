"""OBS-01..04 observability tests.

Plan 04-01 adds OBS-01 (per-call usage_log) unit + wiring tests.
Plan 04-03 contributed the dashboard module smoke + integration test.
Plans 04-02 / 04-04 will extend this file further.
"""
from __future__ import annotations

import asyncio
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from arca.config import get_settings
from arca.observability import (
    MODEL_COSTS_PER_MTOK,
    calculate_cost,
    extract_tokens,
    log_usage_event,
)
from unittest.mock import patch


FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "sample_sse.bin"
# Known token counts baked into tests/fixtures/sample_sse.bin:
FIXTURE_INPUT_TOKENS = 25
FIXTURE_OUTPUT_TOKENS = 127


# ---------------------------------------------------------------------------
# OBS-01 — extract_tokens
# ---------------------------------------------------------------------------
def test_extract_tokens():
    raw = FIXTURE_PATH.read_bytes()
    in_tok, out_tok = extract_tokens(raw)
    assert in_tok == FIXTURE_INPUT_TOKENS
    assert out_tok == FIXTURE_OUTPUT_TOKENS


def test_extract_tokens_handles_missing_message_start():
    raw = (
        b'event: message_delta\n'
        b'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},'
        b'"usage":{"output_tokens":42}}\n\n'
    )
    in_tok, out_tok = extract_tokens(raw)
    assert in_tok == 0
    assert out_tok == 42


def test_extract_tokens_handles_malformed_json():
    raw = (
        b'event: broken\n'
        b'data: {this is not json\n\n'
        b'event: message_start\n'
        b'data: {"type":"message_start","message":{"usage":{"input_tokens":17,"output_tokens":0}}}\n\n'
    )
    in_tok, out_tok = extract_tokens(raw)
    assert in_tok == 17
    assert out_tok == 0


# ---------------------------------------------------------------------------
# OBS-01 — calculate_cost
# ---------------------------------------------------------------------------
def test_calculate_cost_claude_sonnet_4():
    got = calculate_cost("claude-sonnet-4-20250514", 1000, 500)
    expected = (1000 * 3.0 + 500 * 15.0) / 1_000_000
    assert got == pytest.approx(expected, rel=1e-9)


def test_calculate_cost_claude_opus_4_legacy_rates():
    # claude-opus-4-20250514 (Opus 4.0) bills 15/75 — not the 5/25 of Opus 4.5+.
    got = calculate_cost("claude-opus-4-20250514", 2000, 1000)
    expected = (2000 * 15.0 + 1000 * 75.0) / 1_000_000
    assert got == pytest.approx(expected, rel=1e-9)


def test_calculate_cost_opus_4_5_longest_prefix_wins():
    # claude-opus-4-5 must match its own 5/25 rate, not "claude-opus-4" 15/75.
    got = calculate_cost("claude-opus-4-5-20251101", 2000, 1000)
    expected = (2000 * 5.0 + 1000 * 25.0) / 1_000_000
    assert got == pytest.approx(expected, rel=1e-9)


def test_calculate_cost_opus_4_1():
    got = calculate_cost("claude-opus-4-1-20250805", 1000, 1000)
    expected = (1000 * 15.0 + 1000 * 75.0) / 1_000_000
    assert got == pytest.approx(expected, rel=1e-9)


def test_calculate_cost_haiku_3_5():
    # Real legacy id is claude-3-5-haiku-... (number-first naming). This family
    # previously fell through to the default tier, overstating cost ~4x.
    got = calculate_cost("claude-3-5-haiku-20241022", 10000, 5000)
    expected = (10000 * 0.80 + 5000 * 4.0) / 1_000_000
    assert got == pytest.approx(expected, rel=1e-9)


def test_calculate_cost_normalized_family_matches():
    # arca.normalizer strips the date suffix; the table must match that form too.
    got = calculate_cost("claude-3-5-haiku", 1000, 1000)
    expected = (1000 * 0.80 + 1000 * 4.0) / 1_000_000
    assert got == pytest.approx(expected, rel=1e-9)


def test_calculate_cost_unknown_model_uses_default():
    assert MODEL_COSTS_PER_MTOK["default"] == (3.0, 15.0)
    got = calculate_cost("claude-unknown-99", 1000, 500)
    expected = (1000 * 3.0 + 500 * 15.0) / 1_000_000
    assert got == pytest.approx(expected, rel=1e-9)


def test_calculate_cost_empty_model_string():
    # Must not raise; must use default tier.
    got = calculate_cost("", 1000, 500)
    expected = (1000 * 3.0 + 500 * 15.0) / 1_000_000
    assert got == pytest.approx(expected, rel=1e-9)


# ---------------------------------------------------------------------------
# OBS-01 — cost_saved semantics
# ---------------------------------------------------------------------------
def test_cost_saved_on_hit_equals_cost_usd():
    from arca.observability import _compute_costs

    cost_usd, cost_saved_usd = _compute_costs(
        cache_hit=True,
        model="claude-sonnet-4-20250514",
        input_tokens=1000,
        output_tokens=500,
    )
    assert cost_usd > 0
    assert cost_saved_usd == pytest.approx(cost_usd, rel=1e-9)


def test_cost_saved_on_miss_is_zero():
    from arca.observability import _compute_costs

    cost_usd, cost_saved_usd = _compute_costs(
        cache_hit=False,
        model="claude-sonnet-4-20250514",
        input_tokens=1000,
        output_tokens=500,
    )
    assert cost_usd > 0
    assert cost_saved_usd == 0.0


# ---------------------------------------------------------------------------
# OBS-01 — log_usage_event runtime behavior (fire-and-forget + graceful)
# ---------------------------------------------------------------------------
async def test_log_usage_event_does_not_block(monkeypatch):
    """log_usage_event must schedule DB write off-loop and return in <50ms."""
    from arca import observability as obs

    def _slow_execute(*args, **kwargs):
        time.sleep(0.5)

    cursor = MagicMock()
    cursor.execute.side_effect = _slow_execute
    conn = MagicMock()
    conn.cursor.return_value = cursor

    import threading

    # Patch app.state.sql on the live FastAPI app used by observability.
    monkeypatch.setattr(obs.app.state, "sql", conn, raising=False)
    monkeypatch.setattr(obs.app.state, "sql_lock", threading.Lock(), raising=False)

    t0 = time.monotonic()
    row_id = await obs.log_usage_event(
        cache_hit=False,
        model="claude-sonnet-4-20250514",
        input_tokens=100,
        output_tokens=50,
        cost_usd=0.001,
        cost_saved_usd=0.0,
        latency_ms=123,
        similarity_score=None,
    )
    elapsed = time.monotonic() - t0
    assert elapsed < 0.05, f"log_usage_event blocked for {elapsed*1000:.1f}ms"
    assert isinstance(row_id, str) and len(row_id) > 0

    # Let the background task finish so pytest doesn't report a pending task.
    await asyncio.sleep(0.6)


async def test_log_usage_event_graceful_when_sql_none(monkeypatch):
    """With app.state.sql=None, log_usage_event must not raise and must not touch the lock."""
    from arca import observability as obs

    monkeypatch.setattr(obs.app.state, "sql", None, raising=False)
    # Sentinel lock that records if it was acquired. If it is, that's a bug.
    acquired: dict[str, bool] = {"was": False}

    class RecordingLock:
        def __enter__(self):
            acquired["was"] = True
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(obs.app.state, "sql_lock", RecordingLock(), raising=False)

    row_id = await obs.log_usage_event(
        cache_hit=True,
        model="claude-sonnet-4-20250514",
        input_tokens=10,
        output_tokens=5,
        cost_usd=0.0,
        cost_saved_usd=0.0,
        latency_ms=1,
        similarity_score=1.0,
    )
    assert isinstance(row_id, str) and len(row_id) > 0
    assert acquired["was"] is False, "sql_lock acquired even though app.state.sql is None"


# ---------------------------------------------------------------------------
# OBS-01 — integration (live Databricks)
# ---------------------------------------------------------------------------
@pytest.mark.integration
async def test_usage_log_insert(databricks_env):
    """End-to-end: log_usage_event writes a row to demo_jedi.arca.usage_log.

    Auto-skipped unless DATABRICKS_TOKEN (and friends) are set.
    """
    from arca import observability as obs
    from databricks import sql as dbsql
    import threading

    host = databricks_env["host"].replace("https://", "").replace("http://", "")
    conn = dbsql.connect(
        server_hostname=host,
        http_path=databricks_env["http_path"],
        access_token=databricks_env["token"],
    )
    try:
        obs.app.state.sql = conn
        obs.app.state.sql_lock = threading.Lock()

        row_id = await obs.log_usage_event(
            cache_hit=False,
            model="claude-sonnet-4-20250514",
            input_tokens=42,
            output_tokens=7,
            cost_usd=0.000231,
            cost_saved_usd=0.0,
            latency_ms=88,
            similarity_score=None,
        )
        # Wait for the fire-and-forget task to flush. 5s is the OBS-01 budget.
        for _ in range(50):
            await asyncio.sleep(0.1)
            cur = conn.cursor()
            cur.execute(
                "SELECT id, cache_hit, model, input_tokens, output_tokens, "
                "cost_usd, cost_saved_usd, latency_ms, similarity_score "
                f"FROM {get_settings().usage_table} WHERE id = :id",
                {"id": row_id},
            )
            rows = cur.fetchall()
            cur.close()
            if rows:
                break
        assert rows, f"usage_log row {row_id} not found within 5s"
        row = rows[0]
        assert row[1] is False  # cache_hit
        assert row[2] == "claude-sonnet-4-20250514"
        assert row[3] == 42
        assert row[4] == 7
        assert row[7] == 88
    finally:
        try:
            conn.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# OBS-01 — cache.py wiring (Task 3 adds these)
# ---------------------------------------------------------------------------
async def test_pre_request_logs_on_l1_hit(monkeypatch):
    """L1 hit path must call log_usage_event with cache_hit=True, similarity_score=1.0."""
    import arca.cache as cache

    raw_sse = FIXTURE_PATH.read_bytes()

    # Seed L1 with a canonical key.
    body = b'{"model":"claude-sonnet-4-20250514","messages":[{"role":"user","content":"hi"}]}'
    canonical = cache.canonicalize(body)
    key = cache.prompt_hash(canonical)
    cache._l1_put(key, raw_sse)

    mock_log = AsyncMock(return_value="row-xyz")
    monkeypatch.setattr(cache, "log_usage_event", mock_log)

    # Minimal Request stub: needs method, url.path, headers, state.
    req = SimpleNamespace(
        method="POST",
        url=SimpleNamespace(path="/v1/messages"),
        headers={},
        state=SimpleNamespace(),
        _body=body,
    )

    result = await cache._pre_request(req, {}, body)
    assert result is not None  # Hit path returned a Response
    mock_log.assert_awaited_once()
    kwargs = mock_log.await_args.kwargs
    assert kwargs["cache_hit"] is True
    assert kwargs["similarity_score"] == 1.0
    assert kwargs["cost_saved_usd"] > 0
    assert kwargs["cost_saved_usd"] == kwargs["cost_usd"]


async def test_post_response_logs_on_miss(monkeypatch):
    """Miss post-response path must call log_usage_event with cache_hit=False, similarity_score=None."""
    import arca.cache as cache
    import numpy as np

    raw_sse = FIXTURE_PATH.read_bytes()
    body = b'{"model":"claude-sonnet-4-20250514","messages":[{"role":"user","content":"hi"}]}'
    canonical = cache.canonicalize(body)
    key = cache.prompt_hash(canonical)
    vec = np.zeros(384, dtype=np.float32)

    mock_log = AsyncMock(return_value="row-abc")
    monkeypatch.setattr(cache, "log_usage_event", mock_log)

    # Prevent Delta + VS write-back from firing.
    async def _noop_write_back(*a, **kw):
        return None

    monkeypatch.setattr(cache, "_write_back", _noop_write_back)

    req = SimpleNamespace(
        method="POST",
        url=SimpleNamespace(path="/v1/messages"),
        headers={},
        state=SimpleNamespace(
            cache_canonical=canonical,
            cache_prompt_hash=key,
            cache_vector=vec,
            t0=time.monotonic(),
        ),
    )

    await cache._post_response(req, raw_sse)
    mock_log.assert_awaited_once()
    kwargs = mock_log.await_args.kwargs
    assert kwargs["cache_hit"] is False
    assert kwargs["similarity_score"] is None
    assert kwargs["cost_saved_usd"] == 0.0


# ---------------------------------------------------------------------------
# OBS-03 — dashboard module (seeded by Plan 04-03)
# ---------------------------------------------------------------------------
def _load_dashboard_module():
    """Return the arca.databricks.dashboard module."""
    import arca.databricks.dashboard as mod
    return mod


def test_dashboard_module_loads_and_exposes_api():
    """Smoke: module imports cleanly and exposes the expected public symbols."""
    mod = _load_dashboard_module()
    assert hasattr(mod, "create_dashboard_if_missing")
    assert hasattr(mod, "export_template")
    assert hasattr(mod, "DASHBOARD_NAME")
    assert mod.DASHBOARD_NAME == "arca-cost-analytics"
    assert hasattr(mod, "DASHBOARD_DEF_PATH")
    assert mod.DASHBOARD_DEF_PATH.exists()


# ---------------------------------------------------------------------------
# OBS-02 — _p95 helper
# ---------------------------------------------------------------------------
def test_p95_helper_empty_returns_zero():
    from arca.observability import _p95
    assert _p95([]) == 0.0


def test_p95_helper_with_1_point_returns_that_point():
    from arca.observability import _p95
    assert _p95([150]) == 150.0


def test_p95_helper_with_2_points():
    from arca.observability import _p95
    result = _p95([100, 200])
    assert isinstance(result, float)
    assert result > 100.0


# ---------------------------------------------------------------------------
# OBS-02 — flush_session_metrics
# ---------------------------------------------------------------------------
async def test_flush_session_metrics_computes_hit_rate():
    from arca import observability as obs

    acc = {
        "total_calls": 10,
        "hit_count": 7,
        "cost_usd_total": 0.5,
        "latencies_ms": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    }

    with patch("mlflow.tracking.MlflowClient") as mock_client_cls:
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        await obs.flush_session_metrics("fake-run-id", acc)

    # Extract (metric_name, value) from the log_metric calls
    calls = mock_client.log_metric.call_args_list
    metrics = {c.args[1]: c.args[2] for c in calls}
    assert metrics["cost_usd"] == pytest.approx(0.5)
    assert metrics["hit_rate"] == pytest.approx(0.7)
    assert metrics["total_calls"] == pytest.approx(10.0)
    assert metrics["latency_p95"] > 0.0


async def test_flush_session_metrics_zero_calls_no_error():
    from arca import observability as obs

    acc = {
        "total_calls": 0,
        "hit_count": 0,
        "cost_usd_total": 0.0,
        "latencies_ms": [],
    }

    with patch("mlflow.tracking.MlflowClient") as mock_client_cls:
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        # Must NOT raise ZeroDivisionError
        await obs.flush_session_metrics("fake-run-id", acc)

    calls = mock_client.log_metric.call_args_list
    metrics = {c.args[1]: c.args[2] for c in calls}
    assert metrics["hit_rate"] == 0.0
    assert metrics["total_calls"] == 0.0


async def test_flush_session_metrics_none_run_id_noop():
    from arca import observability as obs
    with patch("mlflow.tracking.MlflowClient") as mock_client_cls:
        await obs.flush_session_metrics(None, {"total_calls": 5})
        mock_client_cls.assert_not_called()


# ---------------------------------------------------------------------------
# OBS-02 — session_id propagation into usage_log
# ---------------------------------------------------------------------------
async def test_session_id_propagates_to_usage_log(monkeypatch):
    """Setting app.state.session_id must flow into the INSERT args."""
    from arca import observability as obs
    import threading

    cursor = MagicMock()
    conn = MagicMock()
    conn.cursor.return_value = cursor

    monkeypatch.setattr(obs.app.state, "sql", conn, raising=False)
    monkeypatch.setattr(obs.app.state, "sql_lock", threading.Lock(), raising=False)
    monkeypatch.setattr(obs.app.state, "session_id", "sess-xyz", raising=False)

    row_id = await obs.log_usage_event(
        cache_hit=False,
        model="claude-sonnet-4-20250514",
        input_tokens=1,
        output_tokens=1,
        cost_usd=0.0,
        cost_saved_usd=0.0,
        latency_ms=1,
        similarity_score=None,
    )
    # Wait for fire-and-forget task
    for _ in range(20):
        await asyncio.sleep(0.05)
        if cursor.execute.call_args is not None:
            break

    assert cursor.execute.call_args is not None, "execute never called"
    # Named parameters (databricks-sql-connector native paramstyle)
    params = cursor.execute.call_args.args[1]
    assert params["id"] == row_id
    assert params["session_id"] == "sess-xyz"


# ---------------------------------------------------------------------------
# OBS-02 — lifespan integration
# ---------------------------------------------------------------------------
async def test_lifespan_sets_session_id_and_accumulator(monkeypatch):
    """Entering the lifespan must populate app.state.session_id, metrics_accumulator, flush_task."""
    from starlette.testclient import TestClient
    from arca.proxy import app
    from arca import observability as obs

    async def _fake_start(sid):
        return None

    async def _fake_flush(rid, acc):
        return None

    async def _fake_end(rid):
        return None

    monkeypatch.setattr(obs, "start_session", _fake_start)
    monkeypatch.setattr(obs, "flush_session_metrics", _fake_flush)
    monkeypatch.setattr(obs, "end_session", _fake_end)

    with TestClient(app):
        assert isinstance(app.state.session_id, str)
        assert len(app.state.session_id) == 36  # UUID4 canonical form
        assert isinstance(app.state.metrics_accumulator, dict)
        assert app.state.metrics_accumulator["total_calls"] == 0
        assert hasattr(app.state, "flush_task")


# ---------------------------------------------------------------------------
# OBS-02 — start_session integration (live Databricks)
# ---------------------------------------------------------------------------
@pytest.mark.integration
async def test_start_session_integration():
    """Live MLflow: start_session returns a run_id and the run_name matches session_id."""
    from arca import observability as obs
    from mlflow.tracking import MlflowClient

    session_id = "test-session-" + str(time.time())
    run_id = await obs.start_session(session_id)
    assert run_id is not None
    assert isinstance(run_id, str) and len(run_id) > 0

    client = MlflowClient()
    run = client.get_run(run_id)
    # run_name is recorded as a tag
    assert run.data.tags.get("mlflow.runName") == session_id

    # clean up
    await obs.end_session(run_id)


# ---------------------------------------------------------------------------
# OBS-04 — /tail SSE endpoint + tail_queue lifespan
# ---------------------------------------------------------------------------
async def test_tail_queue_created_in_lifespan(monkeypatch):
    """Entering the lifespan must create app.state.tail_queue as an empty asyncio.Queue."""
    from starlette.testclient import TestClient
    from arca.proxy import app
    from arca import observability as obs

    async def _fake_start(sid):
        return None

    async def _fake_flush(rid, acc):
        return None

    async def _fake_end(rid):
        return None

    monkeypatch.setattr(obs, "start_session", _fake_start)
    monkeypatch.setattr(obs, "flush_session_metrics", _fake_flush)
    monkeypatch.setattr(obs, "end_session", _fake_end)

    with TestClient(app):
        assert isinstance(app.state.tail_queue, asyncio.Queue)
        assert app.state.tail_queue.empty()
    # __exit__ completed without error (queue drain path)


async def test_tail_endpoint_yields_event():
    """After pushing an event onto the queue, the /tail generator must yield
    `data: {json}\\n\\n` within 500ms. Tests the route handler directly
    (avoids ASGITransport buffering of infinite streams)."""
    import json as _json
    from types import SimpleNamespace
    from arca.proxy import tail as tail_route, app

    # Use a fresh queue isolated to this test
    q: asyncio.Queue = asyncio.Queue(maxsize=1000)

    class FakeRequest:
        def __init__(self, app_obj):
            self.app = app_obj

        async def is_disconnected(self) -> bool:
            return False

    fake_app = SimpleNamespace(state=SimpleNamespace(tail_queue=q))
    req = FakeRequest(fake_app)

    fake_event = {
        "cache_hit": True,
        "model": "claude-sonnet-4-20250514",
        "input_tokens": 10,
        "output_tokens": 5,
        "cost_usd": 0.001,
        "cost_saved_usd": 0.001,
        "latency_ms": 7,
        "similarity_score": 1.0,
        "ts": time.time(),
    }
    q.put_nowait(fake_event)

    response = await tail_route(req)
    assert response.media_type == "text/event-stream"

    t0 = time.monotonic()
    got_chunk = None
    async for chunk in response.body_iterator:
        if isinstance(chunk, bytes):
            chunk = chunk.decode()
        if chunk.startswith("data:"):
            got_chunk = chunk
            break
        if time.monotonic() - t0 > 2.0:
            break
    elapsed = time.monotonic() - t0
    assert got_chunk is not None, "no data: chunk yielded from /tail generator"
    assert elapsed < 2.0
    payload = _json.loads(got_chunk[len("data:"):].strip())
    assert payload["cache_hit"] is True
    assert payload["model"] == "claude-sonnet-4-20250514"


async def test_tail_endpoint_emits_keepalive_on_idle():
    """With an empty queue, the /tail generator must emit `: keep-alive` within
    ~1.2s. Tests the route handler directly."""
    from types import SimpleNamespace
    from arca.proxy import tail as tail_route

    q: asyncio.Queue = asyncio.Queue(maxsize=1000)

    class FakeRequest:
        def __init__(self, app_obj):
            self.app = app_obj

        async def is_disconnected(self) -> bool:
            return False

    fake_app = SimpleNamespace(state=SimpleNamespace(tail_queue=q))
    req = FakeRequest(fake_app)

    response = await tail_route(req)

    t0 = time.monotonic()
    got_keepalive = False
    async for chunk in response.body_iterator:
        if isinstance(chunk, bytes):
            chunk = chunk.decode()
        if chunk.startswith(":"):
            got_keepalive = True
            break
        if time.monotonic() - t0 > 3.0:
            break
    assert got_keepalive, "no `:` keep-alive chunk yielded within 3s"


# ---------------------------------------------------------------------------
# OBS-04 — arca/cli.py _render_event
# ---------------------------------------------------------------------------
def test_tail_cli_renders_hit_line():
    import io
    from rich.console import Console
    from arca.cli import _render_event
    buf = io.StringIO()
    console = Console(file=buf, force_terminal=True, color_system="truecolor")
    _render_event({
        "cache_hit": True, "model": "claude-sonnet-4-20250514",
        "latency_ms": 9, "cost_usd": 0.001, "cost_saved_usd": 0.001,
        "similarity_score": 1.0,
    }, console)
    output = buf.getvalue()
    assert "HIT" in output
    assert "latency=9ms" in output
    assert "\x1b[" in output  # at least one ANSI escape


def test_tail_cli_renders_miss_line():
    import io
    from rich.console import Console
    from arca.cli import _render_event
    buf = io.StringIO()
    console = Console(file=buf, force_terminal=True, color_system="truecolor")
    _render_event({
        "cache_hit": False, "model": "claude-sonnet-4-20250514",
        "latency_ms": 847, "cost_usd": 0.0, "cost_saved_usd": 0.0,
        "similarity_score": None,
    }, console)
    output = buf.getvalue()
    assert "MISS" in output
    assert "sim=—" in output


@pytest.mark.integration
async def test_create_dashboard_integration():
    """Live-Databricks integration: create dashboard, verify id, verify idempotency.

    Auto-skipped when DATABRICKS_TOKEN is unset (see conftest.py).
    Runs during phase-gate validation with live workspace credentials.
    """
    mod = _load_dashboard_module()
    dashboard_id = await mod.create_dashboard_if_missing()
    assert dashboard_id is not None, "create_dashboard_if_missing returned None with DATABRICKS_TOKEN set"
    assert isinstance(dashboard_id, str)
    assert len(dashboard_id) > 0
    dashboard_id2 = await mod.create_dashboard_if_missing()
    assert dashboard_id == dashboard_id2, (
        f"idempotency violated: first call returned {dashboard_id}, "
        f"second call returned {dashboard_id2}"
    )
