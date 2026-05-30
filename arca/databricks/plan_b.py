"""Arca Plan B: dormant brute-force cosine fallback over demo_jedi.arca.cache_store.

Activates ONLY when the VS endpoint (arca-vs-endpoint) circuit breaker trips.
Under normal operation this module is imported but never called.

IMPORTANT: This module performs NO Databricks call on import — safe to import
unconditionally in the proxy startup path.

Requirements: DB-02 (fallback path)
"""
from __future__ import annotations

import math
from typing import Any

# ---------------------------------------------------------------------------
# Constants (mirror 00_bootstrap.py — do not import from there to keep
# this module standalone / importable without the full bootstrap installed)
# ---------------------------------------------------------------------------
CACHE_TABLE = "demo_jedi.arca.cache_store"
ENDPOINT = "arca-vs-endpoint"
INDEX = "demo_jedi.arca.prompt_index"


def _cosine(a: list[float], b: list[float]) -> float:
    """Pure-Python cosine similarity. Uses math.sqrt — no numpy dependency."""
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(y * y for y in b))
    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0
    return dot / (mag_a * mag_b)


def brute_force_cosine(
    query_vector: list[float],
    *,
    top_k: int = 1,
    limit: int = 10000,
    conn: Any = None,  # optional pre-opened databricks.sql connection
) -> list[dict]:
    """Brute-force cosine search over demo_jedi.arca.cache_store.embedding.

    Activates when VS endpoint is unreachable. Reads up to `limit` rows via
    databricks-sql-connector, computes cosine in Python, returns top-k rows with
    id, prompt_hash, response_json, cost_usd, similarity.

    Not intended for hot path — O(N*384) per query. Fallback only.

    Parameters
    ----------
    query_vector:
        The query embedding (list of 384 floats).
    top_k:
        Number of results to return.
    limit:
        Maximum number of rows to fetch from Delta (default 10 000).
    conn:
        Optional pre-opened ``databricks.sql`` connection. If None, a new
        connection is opened using DATABRICKS_HOST, DATABRICKS_TOKEN, and
        DATABRICKS_HTTP_PATH from the environment.

    Returns
    -------
    list[dict]
        Top-k rows sorted by descending cosine similarity, each with keys:
        id, prompt_hash, prompt_text, response_json, cost_usd, similarity.
    """
    import os

    _own_conn = conn is None
    if _own_conn:
        from databricks import sql as dbsql

        host = os.environ.get("DATABRICKS_HOST", "")
        token = os.environ.get("DATABRICKS_TOKEN", "")
        http_path = os.environ.get("DATABRICKS_HTTP_PATH", "")
        server_hostname = host.replace("https://", "").replace("http://", "")
        conn = dbsql.connect(
            server_hostname=server_hostname,
            http_path=http_path,
            access_token=token,
        )

    try:
        cur = conn.cursor()
        cur.execute(
            f"SELECT id, prompt_hash, prompt_text, embedding, response_json, cost_usd "
            f"FROM {CACHE_TABLE} LIMIT {limit}"
        )
        rows = cur.fetchall()
    finally:
        if _own_conn:
            conn.close()

    scored: list[dict] = []
    for row in rows:
        row_id, prompt_hash, prompt_text, embedding, response_json, cost_usd = row
        if embedding is None:
            continue
        # embedding is returned as a list of floats by the SQL connector
        sim = _cosine(query_vector, list(embedding))
        scored.append(
            {
                "id": row_id,
                "prompt_hash": prompt_hash,
                "prompt_text": prompt_text,
                "response_json": response_json,
                "cost_usd": cost_usd,
                "similarity": sim,
            }
        )

    scored.sort(key=lambda r: r["similarity"], reverse=True)
    return scored[:top_k]


if __name__ == "__main__":
    # Accidental direct execution is safe — print a guard message and exit 0.
    import sys

    print("plan_b dormant -- invoke via brute_force_cosine()")
    print("  This module activates only when the VS endpoint circuit breaker trips.")
    print(f"  Endpoint: {ENDPOINT}")
    print(f"  Index:    {INDEX}")
    sys.exit(0)
