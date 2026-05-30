"""SQLite WAL fallback store for Arca (DB-04).

When Delta writes are disabled or failing, cache rows are enqueued locally at
$ARCA_HOME/pending.db (default: ~/.arca/pending.db). A later phase drains the queue
back to Delta.

Invariants:
- journal_mode = WAL (crash-safe + concurrent readers)
- synchronous = NORMAL (WAL-appropriate durability)
- Single writer serialized by asyncio.Lock (matches proxy's single-process model)
- All sqlite3 calls wrapped in asyncio.to_thread (DB-03: no blocking on hot path)

Requirements satisfied: DB-04
"""
from __future__ import annotations

import asyncio
import json
import os
import sqlite3
from pathlib import Path
from typing import Any

_DDL = """
CREATE TABLE IF NOT EXISTS pending_cache (
  id              TEXT PRIMARY KEY,
  prompt_hash     TEXT NOT NULL,
  prompt_text     TEXT NOT NULL,
  embedding_json  TEXT NOT NULL,
  response_json   TEXT NOT NULL,
  model           TEXT,
  cost_usd        REAL,
  enqueued_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_prompt_hash ON pending_cache(prompt_hash);
"""


def _arca_home() -> Path:
    env = os.environ.get("ARCA_HOME")
    return Path(env) if env else Path.home() / ".arca"


def db_path() -> Path:
    return _arca_home() / "pending.db"


def _init_sync(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, isolation_level=None, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.executescript(_DDL)
    return conn


class SQLiteFallback:
    """Single-process async wrapper around a SQLite WAL file.

    Usage::

        store = SQLiteFallback()
        await store.start()          # creates DB, sets WAL + NORMAL
        await store.enqueue(row)     # INSERT OR REPLACE
        rows = await store.fetch_all()
        await store.stop()
    """

    def __init__(self, path: Path | None = None) -> None:
        self._path = path or db_path()
        self._lock = asyncio.Lock()
        self._conn: sqlite3.Connection | None = None

    @property
    def path(self) -> Path:
        return self._path

    async def start(self) -> None:
        """Create (or open) the SQLite DB with WAL mode and create schema."""
        self._conn = await asyncio.to_thread(_init_sync, self._path)

    async def stop(self) -> None:
        """Close the connection gracefully."""
        if self._conn is not None:
            await asyncio.to_thread(self._conn.close)
            self._conn = None

    async def enqueue(self, row: dict[str, Any]) -> None:
        """Insert-or-replace a pending cache row.

        ``row['embedding']`` must be a list[float] of length 384; it is
        JSON-serialized on write and deserialized transparently on ``fetch_all``.
        """
        if self._conn is None:
            raise RuntimeError("SQLiteFallback.start() must be awaited first")
        async with self._lock:
            await asyncio.to_thread(self._insert_sync, row)

    def _insert_sync(self, row: dict[str, Any]) -> None:
        assert self._conn is not None
        self._conn.execute(
            "INSERT OR REPLACE INTO pending_cache "
            "(id, prompt_hash, prompt_text, embedding_json, response_json, model, cost_usd) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                row["id"],
                row["prompt_hash"],
                row["prompt_text"],
                json.dumps(row["embedding"]),
                row["response_json"],
                row.get("model"),
                row.get("cost_usd"),
            ),
        )

    async def fetch_all(self) -> list[dict[str, Any]]:
        """Return all pending rows as dicts (embedding deserialized from JSON)."""
        if self._conn is None:
            raise RuntimeError("SQLiteFallback.start() must be awaited first")
        async with self._lock:
            rows = await asyncio.to_thread(self._fetch_sync)
        return rows

    def _fetch_sync(self) -> list[dict[str, Any]]:
        assert self._conn is not None
        cur = self._conn.execute(
            "SELECT id, prompt_hash, prompt_text, embedding_json, response_json, "
            "model, cost_usd, enqueued_at FROM pending_cache ORDER BY enqueued_at"
        )
        cols = [d[0] for d in cur.description]
        result = []
        for row in cur.fetchall():
            d = dict(zip(cols, row))
            d["embedding"] = json.loads(d.pop("embedding_json"))
            result.append(d)
        return result
