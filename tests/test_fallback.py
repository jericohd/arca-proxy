"""Unit tests for arca.fallback.SQLiteFallback (DB-04).

All tests are pure unit tests — no Databricks credentials required.
Tests use the `arca_home` fixture from conftest to isolate DB files.
asyncio_mode=auto (set in pyproject.toml) means no @pytest.mark.asyncio needed.
"""
from __future__ import annotations

import asyncio
import json
import sqlite3
from pathlib import Path

import pytest

from arca.fallback import SQLiteFallback, db_path


# ---------------------------------------------------------------------------
# Test 1: Importing arca.fallback does NOT create any files on disk.
# ---------------------------------------------------------------------------
def test_import_no_side_effects(arca_home):
    """Module import must not create pending.db or any file."""
    db = arca_home / "pending.db"
    assert not db.exists(), "import should not create pending.db"


# ---------------------------------------------------------------------------
# Test 2: await store.start() creates $ARCA_HOME/pending.db (parent dirs created).
# ---------------------------------------------------------------------------
async def test_start_creates_db_file(arca_home):
    store = SQLiteFallback()
    await store.start()
    assert store.path.exists(), "start() must create pending.db"
    await store.stop()


# ---------------------------------------------------------------------------
# Test 3: After start(), PRAGMA journal_mode returns 'wal' (lowercase).
# ---------------------------------------------------------------------------
async def test_start_creates_wal_db(arca_home):
    store = SQLiteFallback()
    await store.start()
    conn = sqlite3.connect(store.path)
    mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
    conn.close()
    assert mode.lower() == "wal", f"expected WAL mode, got: {mode}"
    await store.stop()


# ---------------------------------------------------------------------------
# Test 4: After start(), PRAGMA synchronous returns 1 (NORMAL) on the store connection.
# Note: PRAGMA synchronous is connection-scoped. A fresh sqlite3.connect() will show
# the default (2=FULL). We verify via the store's own internal connection.
# ---------------------------------------------------------------------------
async def test_start_synchronous_normal(arca_home):
    store = SQLiteFallback()
    await store.start()
    # Query via the store's internal connection (where NORMAL was set)
    sync_val = await asyncio.to_thread(
        lambda: store._conn.execute("PRAGMA synchronous").fetchone()[0]
    )
    # SQLite returns integer: 0=OFF, 1=NORMAL, 2=FULL, 3=EXTRA
    assert sync_val == 1, f"expected NORMAL synchronous (1), got: {sync_val}"
    await store.stop()


# ---------------------------------------------------------------------------
# Test 5: pending_cache table exists with all required columns.
# ---------------------------------------------------------------------------
async def test_pending_cache_table_columns(arca_home):
    store = SQLiteFallback()
    await store.start()
    conn = sqlite3.connect(store.path)
    cols = {row[1] for row in conn.execute("PRAGMA table_info(pending_cache)")}
    conn.close()
    required = {"id", "prompt_hash", "prompt_text", "embedding_json", "response_json", "model", "cost_usd", "enqueued_at"}
    missing = required - cols
    assert not missing, f"missing columns: {missing}"
    await store.stop()


# ---------------------------------------------------------------------------
# Test 6: idx_prompt_hash index exists on pending_cache(prompt_hash).
# ---------------------------------------------------------------------------
async def test_idx_prompt_hash_exists(arca_home):
    store = SQLiteFallback()
    await store.start()
    conn = sqlite3.connect(store.path)
    indexes = {row[1] for row in conn.execute("PRAGMA index_list(pending_cache)")}
    conn.close()
    assert "idx_prompt_hash" in indexes, f"idx_prompt_hash not found; indexes: {indexes}"
    await store.stop()


# ---------------------------------------------------------------------------
# Test 7: enqueue round-trip — embedding (384 floats) round-trips correctly.
# ---------------------------------------------------------------------------
async def test_enqueue_roundtrip(arca_home):
    store = SQLiteFallback()
    await store.start()
    row = {
        "id": "r-1",
        "prompt_hash": "h1",
        "prompt_text": "hello world",
        "embedding": [0.1] * 384,
        "response_json": '{"text": "response"}',
        "model": "claude-3",
        "cost_usd": 0.01,
    }
    await store.enqueue(row)
    rows = await store.fetch_all()
    assert len(rows) == 1
    r = rows[0]
    assert r["prompt_hash"] == "h1"
    assert r["prompt_text"] == "hello world"
    assert r["response_json"] == '{"text": "response"}'
    assert r["embedding"] == [0.1] * 384, "embedding round-trip mismatch"
    await store.stop()


# ---------------------------------------------------------------------------
# Test 8: enqueue same id twice (INSERT OR REPLACE) leaves exactly one row.
# ---------------------------------------------------------------------------
async def test_enqueue_upsert_deduplication(arca_home):
    store = SQLiteFallback()
    await store.start()
    row_v1 = {
        "id": "dup-1",
        "prompt_hash": "h-dup",
        "prompt_text": "original",
        "embedding": [0.0] * 384,
        "response_json": "{}",
        "model": None,
        "cost_usd": None,
    }
    row_v2 = {**row_v1, "prompt_text": "updated"}
    await store.enqueue(row_v1)
    await store.enqueue(row_v2)
    rows = await store.fetch_all()
    assert len(rows) == 1, f"expected 1 row after upsert, got {len(rows)}"
    assert rows[0]["prompt_text"] == "updated", "upsert did not overwrite row"
    await store.stop()


# ---------------------------------------------------------------------------
# Test 9: Two concurrent enqueues serialize via asyncio.Lock — both rows present.
# ---------------------------------------------------------------------------
async def test_concurrent_enqueue_serialization(arca_home):
    store = SQLiteFallback()
    await store.start()

    row_a = {
        "id": "concurrent-a",
        "prompt_hash": "ha",
        "prompt_text": "prompt a",
        "embedding": [0.1] * 384,
        "response_json": "{}",
        "model": None,
        "cost_usd": None,
    }
    row_b = {
        "id": "concurrent-b",
        "prompt_hash": "hb",
        "prompt_text": "prompt b",
        "embedding": [0.2] * 384,
        "response_json": "{}",
        "model": None,
        "cost_usd": None,
    }

    await asyncio.gather(store.enqueue(row_a), store.enqueue(row_b))
    rows = await store.fetch_all()
    ids = {r["id"] for r in rows}
    assert {"concurrent-a", "concurrent-b"} <= ids, f"both rows should be present; got: {ids}"
    await store.stop()


# ---------------------------------------------------------------------------
# Test 10: File path honors $ARCA_HOME env var when set (via arca_home fixture).
# ---------------------------------------------------------------------------
async def test_db_path_honors_arca_home_env(arca_home):
    """db_path() must return $ARCA_HOME/pending.db when ARCA_HOME is set."""
    import os
    expected = Path(os.environ["ARCA_HOME"]) / "pending.db"
    assert db_path() == expected, f"db_path() returned {db_path()}, expected {expected}"

    store = SQLiteFallback()
    await store.start()
    assert store.path == expected
    assert store.path.exists()
    await store.stop()
