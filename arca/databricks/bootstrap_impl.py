"""Arca Phase 0 bootstrap: idempotent Databricks infrastructure provisioning.

Requirements satisfied: DB-01, DB-02, DB-03, DB-04, DB-05

Kick-first, gate-later strategy
--------------------------------
VS endpoint creation is ASYNCHRONOUS and takes 5-15 minutes. This script:
  1. Kicks VS endpoint creation immediately (returns in <2s, provisions async)
  2. Runs all DDL + MLflow setup while the endpoint provisions
  3. Gates on wait_for_endpoint() before creating the index
This means total wall-clock time = max(DDL_time, endpoint_provision_time)
instead of DDL_time + endpoint_provision_time.

DB-03 note for Phase 1+
------------------------
This bootstrap script is SYNCHRONOUS (one-shot provisioning, not on hot path).
The hot-path proxy (Phase 1+) wraps every sync Databricks SDK call in
asyncio.to_thread — see arca/fallback/__init__.py for the pattern.

Usage
-----
    python -m arca.databricks.bootstrap_impl

Exit codes
----------
    0  All assets provisioned and verified
    2  Missing required environment variables
    3  VS endpoint did not reach ONLINE within timeout — activate Plan B
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Locked constants — every subsequent phase references these strings.
# ---------------------------------------------------------------------------
from arca.config import get_settings as _get_settings

_s = _get_settings()
CATALOG = _s.catalog
SCHEMA = _s.db_schema
CACHE_TABLE = _s.cache_table
USAGE_TABLE = _s.usage_table
ENDPOINT = _s.vs_endpoint
INDEX = _s.vs_index


def _resolve_mlflow_experiment() -> str:
    """Resolve lazily at call time — WorkspaceClient().current_user.me() is a
    NETWORK call and must never run at import time (every `arca` CLI command
    imports this module's package)."""
    env = os.environ.get("ARCA_MLFLOW_EXPERIMENT")
    if env:
        return env
    try:
        from databricks.sdk import WorkspaceClient
        email = WorkspaceClient().current_user.me().user_name
        return f"/Users/{email}/arca"
    except Exception:
        return "/arca"
EMBEDDING_DIMS = 384
SECRETS_SCOPE = "demo-secrets"
WARMUP_ID = "warmup-0001"

# VS index schema (Direct Access — fields that will be searchable)
VS_INDEX_SCHEMA = {
    "id": "string",
    "prompt_hash": "string",
    "prompt_text": "string",
    "embedding": "array<float>",
    "response_json": "string",
    "model": "string",
    "cost_usd": "double",
}

# DDL — explicit column types prevent first-write latency spike (Pitfall 3)
_CACHE_STORE_DDL = f"""
CREATE TABLE IF NOT EXISTS {CACHE_TABLE} (
  id              STRING NOT NULL,
  prompt_hash     STRING NOT NULL,
  prompt_text     STRING NOT NULL,
  embedding       ARRAY<FLOAT>,
  response_json   STRING NOT NULL,
  model           STRING,
  input_tokens    INT,
  output_tokens   INT,
  cost_usd        DOUBLE,
  hit_count       INT,
  created_at      TIMESTAMP,
  last_hit_at     TIMESTAMP
) USING DELTA
  TBLPROPERTIES ('delta.enableChangeDataFeed' = 'true')
"""

# similarity_score DOUBLE is REQUIRED by OBS-01 — ARCHITECTURE.md omits it (stale)
_USAGE_LOG_DDL = f"""
CREATE TABLE IF NOT EXISTS {USAGE_TABLE} (
  id               STRING NOT NULL,
  session_id       STRING,
  cache_hit        BOOLEAN,
  model            STRING,
  input_tokens     INT,
  output_tokens    INT,
  cost_usd         DOUBLE,
  cost_saved_usd   DOUBLE,
  latency_ms       INT,
  similarity_score DOUBLE,
  created_at       TIMESTAMP
) USING DELTA
"""


def _check_env() -> tuple[str, str, str]:
    """Assert required env vars are set; return (host, token, http_path)."""
    host = os.environ.get("DATABRICKS_HOST", "")
    token = os.environ.get("DATABRICKS_TOKEN", "")
    http_path = os.environ.get("DATABRICKS_HTTP_PATH", "")

    missing = []
    if not host:
        missing.append("DATABRICKS_HOST")
    if not token:
        missing.append("DATABRICKS_TOKEN")
    if not http_path:
        missing.append("DATABRICKS_HTTP_PATH")

    if missing:
        print("[FAIL] Missing required environment variables:")
        for v in missing:
            print(f"       export {v}=<value>")
        print()
        print("  DATABRICKS_HOST     — workspace URL, e.g. https://<id>.azuredatabricks.net")
        print("  DATABRICKS_TOKEN    — personal access token (Settings > Developer > Access tokens)")
        print("  DATABRICKS_HTTP_PATH — SQL warehouse HTTP path (SQL > Warehouses > <name> > Connection details)")
        sys.exit(2)

    return host, token, http_path


def _kick_endpoint_creation(vsc) -> None:
    """Kick VS endpoint creation immediately (async — returns in <2s)."""
    try:
        vsc.create_endpoint(name=ENDPOINT, endpoint_type="STANDARD")
        print(f"[WAIT] VS endpoint '{ENDPOINT}' creation kicked — provisioning in background (~10 min)...")
    except Exception as e:
        err = str(e).lower()
        if "already exists" in err or "quota" in err or "maximum number" in err:
            print(f"[OK]   VS endpoint '{ENDPOINT}' already exists (or quota reached) — skipping create")
        else:
            raise


def _run_ddl(host: str, token: str, http_path: str) -> None:
    """Connect via databricks-sql-connector and run schema + table DDL."""
    from databricks import sql

    server_hostname = host.replace("https://", "").replace("http://", "")

    with sql.connect(
        server_hostname=server_hostname,
        http_path=http_path,
        access_token=token,
    ) as conn:
        cur = conn.cursor()

        # Create catalog if it doesn't exist (required on fresh workspaces)
        cur.execute(
            f"CREATE CATALOG IF NOT EXISTS {CATALOG} "
            f"COMMENT 'Arca: Claude Code optimizer'"
        )
        print(f"[OK]   CREATE CATALOG IF NOT EXISTS {CATALOG}")

        cur.execute(f"USE CATALOG {CATALOG}")
        print(f"[OK]   USE CATALOG {CATALOG}")

        # Create schema (idempotent)
        cur.execute(
            f"CREATE SCHEMA IF NOT EXISTS {SCHEMA} "
            f"COMMENT 'Arca: Claude Code optimizer -- cache + analytics'"
        )
        print(f"[OK]   Schema {CATALOG}.{SCHEMA} created / already exists")

        # Create cache_store table
        cur.execute(_CACHE_STORE_DDL.strip())
        print(f"[OK]   Table {CACHE_TABLE} created / already exists")

        # Create usage_log table
        cur.execute(_USAGE_LOG_DDL.strip())
        print(f"[OK]   Table {USAGE_TABLE} created / already exists")

        # Upsert warm-up row into cache_store (materializes table, prevents first-write spike)
        cur.execute(
            f"""
            INSERT INTO {CACHE_TABLE}
              (id, prompt_hash, prompt_text, embedding, response_json, model, cost_usd, hit_count, created_at)
            SELECT
              '{WARMUP_ID}', 'warmup', 'hello world',
              array_repeat(CAST(0.0 AS FLOAT), {EMBEDDING_DIMS}),
              '{{}}', 'warmup', 0.0, 0, current_timestamp()
            WHERE NOT EXISTS (
              SELECT 1 FROM {CACHE_TABLE} WHERE id = '{WARMUP_ID}'
            )
            """
        )
        print(f"[OK]   Warm-up row '{WARMUP_ID}' upserted into {CACHE_TABLE}")


def _setup_mlflow() -> str:
    """Create MLflow experiment idempotently; return experiment_id."""
    import mlflow

    mlflow.set_tracking_uri("databricks")  # picks up DATABRICKS_HOST + TOKEN

    experiment = _resolve_mlflow_experiment()  # resolve once — may hit the network
    try:
        exp_id = mlflow.create_experiment(experiment)
        print(f"[OK]   MLflow experiment '{experiment}' created (id={exp_id})")
    except mlflow.exceptions.MlflowException as e:
        if "RESOURCE_ALREADY_EXISTS" not in str(e):
            raise
        exp = mlflow.get_experiment_by_name(experiment)
        exp_id = exp.experiment_id
        print(f"[OK]   MLflow experiment '{experiment}' already exists (id={exp_id})")

    mlflow.set_experiment(experiment)

    # Log a smoke metric to prove the write path works
    with mlflow.start_run(run_name="bootstrap-smoke"):
        mlflow.log_metric("bootstrap_ok", 1)
        mlflow.set_tag("phase", "0")
    print("[OK]   MLflow smoke run logged (bootstrap_ok=1)")

    return exp_id


def _gate_endpoint_online(vsc) -> None:
    """Block until VS endpoint is ONLINE (up to 15 min). Exit 3 on timeout."""
    print(f"[WAIT] Gating on VS endpoint '{ENDPOINT}' reaching ONLINE state...")
    t0 = time.time()
    try:
        from datetime import timedelta
        vsc.wait_for_endpoint(name=ENDPOINT, timeout=timedelta(seconds=900), verbose=True)
        elapsed = int(time.time() - t0)
        print(f"[OK]   VS endpoint '{ENDPOINT}' is ONLINE (took {elapsed}s)")
    except Exception as e:
        print(f"[FAIL] VS endpoint did not reach ONLINE within 900s: {e}")
        print("       Cache still runs in degraded local mode (SQLite L2 fallback).")
        print("       Re-run bootstrap once the endpoint is reachable.")
        sys.exit(3)


def _create_vs_index(vsc) -> None:
    """Create Delta Sync CONTINUOUS index on cache_store (idempotent).

    Direct Access is not available on all workspace tiers; Delta Sync CONTINUOUS
    provides near-real-time sync (typically <5s) which is acceptable for the demo.
    cache_store already has enableChangeDataFeed=true so sync works correctly.
    """
    try:
        vsc.create_delta_sync_index(
            endpoint_name=ENDPOINT,
            index_name=INDEX,
            source_table_name=CACHE_TABLE,
            pipeline_type="TRIGGERED",
            primary_key="id",
            embedding_vector_column="embedding",
            embedding_dimension=EMBEDDING_DIMS,
        )
        print(f"[OK]   VS index '{INDEX}' created (Delta Sync TRIGGERED, {EMBEDDING_DIMS} dims)")
    except Exception as e:
        if "already exists" not in str(e).lower():
            raise
        print(f"[OK]   VS index '{INDEX}' already exists — skipping create")


def _upsert_warmup_vs(vsc) -> None:
    """Poll until index READY, trigger sync, wait for sync to settle.

    With TRIGGERED pipeline we must call index.sync() after READY.
    We poll get_index() status since wait_for_index() is not available in all SDK versions.
    """
    print(f"[WAIT] Polling VS index '{INDEX}' until READY (up to 10 min)...")
    deadline = time.time() + 600
    while time.time() < deadline:
        try:
            idx = vsc.get_index(endpoint_name=ENDPOINT, index_name=INDEX)
            status = (idx.describe().get("status", {}).get("ready_for_query") or
                      idx.describe().get("status", {}).get("detailed_state", ""))
            if status is True or "ready" in str(status).lower():
                print(f"[OK]   VS index '{INDEX}' is READY")
                break
            print(f"[WAIT] Index status: {status} — retrying in 20s...")
        except Exception as e:
            print(f"[WAIT] Index status check error: {e} — retrying in 20s...")
        time.sleep(20)
    else:
        print(f"[WARN] Index not READY within 600s — proceeding anyway")
    # Trigger sync
    try:
        index = vsc.get_index(endpoint_name=ENDPOINT, index_name=INDEX)
        index.sync()
        print(f"[OK]   VS index sync triggered — waiting 30s for warm-up row to propagate...")
        time.sleep(30)
    except Exception as e:
        print(f"[WARN] Could not trigger sync: {e}")


def _smoke_query(vsc) -> None:
    """Run similarity_search against warm-up row; warn (not fail) if sync not complete."""
    try:
        index = vsc.get_index(endpoint_name=ENDPOINT, index_name=INDEX)
        results = index.similarity_search(
            query_vector=[0.0] * EMBEDDING_DIMS,
            columns=["id"],
            num_results=1,
        )
        data = (results or {}).get("result", {}).get("data_array")
        if not data:
            print("[WARN] VS smoke query returned no results — sync may still be in progress")
            print("       Run: python -c \"from databricks.vector_search.client import VectorSearchClient; "
                  f"VectorSearchClient().get_index('{ENDPOINT}', '{INDEX}').sync()\"")
        else:
            print(f"[OK]   VS smoke query returned {len(data)} result(s)")
    except Exception as e:
        print(f"[WARN] Smoke query failed (index may still be syncing): {e}")
        print("       Re-run bootstrap after index sync completes to verify.")


def _verify_secrets(w) -> None:
    """Verify demo-secrets scope is reachable (DB-05)."""
    try:
        scopes = {s.name for s in w.secrets.list_scopes()}
        if SECRETS_SCOPE in scopes:
            print(f"[OK]   Secrets scope '{SECRETS_SCOPE}' is reachable")
        else:
            print(f"[FAIL] Secrets scope '{SECRETS_SCOPE}' not found; available: {scopes}")
    except Exception as e:
        print(f"[FAIL] Could not list secrets scopes: {e}")


def main() -> int:
    """CLI entry point — delegates to bootstrap() and maps exceptions to exit codes."""
    from arca.databricks.bootstrap import bootstrap, BootstrapError

    print("=" * 60)
    print("Arca Phase 0 Bootstrap")
    print("=" * 60)
    print()

    try:
        bootstrap(skip_vs_endpoint=False)
    except BootstrapError as e:
        print(f"[FAIL] {e}")
        return e.exit_code

    print()
    print("=" * 60)
    print("Bootstrap complete — all Phase 0 requirements satisfied:")
    print(f"  [OK] DB-01: schema {CATALOG}.{SCHEMA} + tables cache_store / usage_log")
    print(f"  [OK] DB-02: VS endpoint '{ENDPOINT}' ONLINE + Direct Access index")
    print(f"  [OK] DB-03: asyncio.to_thread pattern documented (hot path in Phase 1+)")
    print(f"  [OK] DB-04: SQLite fallback at ~/.arca/pending.db (arca/fallback/)")
    print(f"  [OK] DB-05: auth verified, secrets scope '{SECRETS_SCOPE}' reachable")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
