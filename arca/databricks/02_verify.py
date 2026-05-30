"""Arca Phase 0 verification: standalone smoke check for all bootstrapped assets.

Seed for `arca doctor` (Phase 5). Runs READ-ONLY checks — never creates
or destroys Databricks state.

Requirements verified: DB-01, DB-02, DB-03, DB-04, DB-05

Usage
-----
    python -m arca.databricks.02_verify
    # or
    python arca/databricks/02_verify.py

Exit codes
----------
    0   All checks pass
    1   One or more checks failed (details printed to stdout)
    2   Missing required environment variables
"""
from __future__ import annotations

import os
import sys
from typing import Any

# ---------------------------------------------------------------------------
# Locked constants (must match 00_bootstrap.py exactly)
# ---------------------------------------------------------------------------
CATALOG = "demo_jedi"
SCHEMA = "arca"
CACHE_TABLE = "demo_jedi.arca.cache_store"
USAGE_TABLE = "demo_jedi.arca.usage_log"
ENDPOINT = "arca-vs-endpoint"
INDEX = "demo_jedi.arca.prompt_index"
def _resolve_mlflow_experiment() -> str:
    env = os.environ.get("ARCA_MLFLOW_EXPERIMENT")
    if env:
        return env
    try:
        from databricks.sdk import WorkspaceClient
        email = WorkspaceClient().current_user.me().user_name
        return f"/Users/{email}/arca"
    except Exception:
        return "/arca"

MLFLOW_EXPERIMENT = _resolve_mlflow_experiment()
EMBEDDING_DIMS = 384
SECRETS_SCOPE = "demo-secrets"

# Required columns for cache_store (name -> expected type)
CACHE_STORE_COLUMNS = {
    "id": "string",
    "prompt_hash": "string",
    "prompt_text": "string",
    "embedding": "array<float>",
    "response_json": "string",
    "model": "string",
    "input_tokens": "int",
    "output_tokens": "int",
    "cost_usd": "double",
    "hit_count": "int",
    "created_at": "timestamp",
    "last_hit_at": "timestamp",
}


def _check_env() -> tuple[str, str, str]:
    host = os.environ.get("DATABRICKS_HOST", "")
    token = os.environ.get("DATABRICKS_TOKEN", "")
    http_path = os.environ.get("DATABRICKS_HTTP_PATH", "")
    missing = [v for v, val in [
        ("DATABRICKS_HOST", host),
        ("DATABRICKS_TOKEN", token),
        ("DATABRICKS_HTTP_PATH", http_path),
    ] if not val]
    if missing:
        print("[FAIL] Missing env vars:", ", ".join(missing))
        sys.exit(2)
    return host, token, http_path


def _sql_connect(host: str, token: str, http_path: str):
    from databricks import sql as dbsql
    server_hostname = host.replace("https://", "").replace("http://", "")
    return dbsql.connect(
        server_hostname=server_hostname,
        http_path=http_path,
        access_token=token,
    )


def check_auth(results: list[tuple[str, bool, str]]) -> None:
    """Check 1: SDK auth — WorkspaceClient().current_user.me()"""
    from databricks.sdk import WorkspaceClient
    try:
        w = WorkspaceClient()
        me = w.current_user.me()
        results.append(("Auth (DB-05)", True, f"authenticated as {me.user_name}"))
    except Exception as e:
        results.append(("Auth (DB-05)", False, str(e)))


def check_schema_and_tables(host: str, token: str, http_path: str,
                             results: list[tuple[str, bool, str]]) -> None:
    """Check 2: SHOW TABLES IN demo_jedi.arca returns cache_store + usage_log."""
    try:
        with _sql_connect(host, token, http_path) as conn:
            cur = conn.cursor()
            cur.execute(f"SHOW TABLES IN {CATALOG}.{SCHEMA}")
            tables = {row[1] for row in cur.fetchall()}
        missing = {"cache_store", "usage_log"} - tables
        if missing:
            results.append(("Schema + tables (DB-01)", False, f"missing tables: {missing}"))
        else:
            results.append(("Schema + tables (DB-01)", True, f"found: {tables}"))
    except Exception as e:
        results.append(("Schema + tables (DB-01)", False, str(e)))


def check_cache_store_columns(host: str, token: str, http_path: str,
                               results: list[tuple[str, bool, str]]) -> None:
    """Check 3: DESCRIBE TABLE cache_store — all columns + correct types."""
    try:
        with _sql_connect(host, token, http_path) as conn:
            cur = conn.cursor()
            cur.execute(f"DESCRIBE TABLE {CACHE_TABLE}")
            cols = {
                r[0]: r[1]
                for r in cur.fetchall()
                if r[0] and not r[0].startswith("#")
            }

        wrong: list[str] = []
        for name, expected_type in CACHE_STORE_COLUMNS.items():
            actual = cols.get(name)
            if actual != expected_type:
                wrong.append(f"{name}: got {actual!r}, want {expected_type!r}")

        if wrong:
            results.append(("cache_store columns (DB-01)", False, "; ".join(wrong)))
        else:
            results.append(("cache_store columns (DB-01)", True,
                            f"all {len(CACHE_STORE_COLUMNS)} columns correct"))
    except Exception as e:
        results.append(("cache_store columns (DB-01)", False, str(e)))


def check_usage_log_has_similarity_score(host: str, token: str, http_path: str,
                                          results: list[tuple[str, bool, str]]) -> None:
    """Check 3b: usage_log must have similarity_score DOUBLE (OBS-01 dependency)."""
    try:
        with _sql_connect(host, token, http_path) as conn:
            cur = conn.cursor()
            cur.execute(f"DESCRIBE TABLE {USAGE_TABLE}")
            cols = {
                r[0]: r[1]
                for r in cur.fetchall()
                if r[0] and not r[0].startswith("#")
            }
        if cols.get("similarity_score") == "double":
            results.append(("usage_log.similarity_score (OBS-01)", True, "type=double"))
        else:
            results.append(("usage_log.similarity_score (OBS-01)", False,
                            f"got {cols.get('similarity_score')!r}, want 'double'"))
    except Exception as e:
        results.append(("usage_log.similarity_score (OBS-01)", False, str(e)))


def check_vs_endpoint_online(results: list[tuple[str, bool, str]]) -> None:
    """Check 4: VS endpoint state == ONLINE."""
    from databricks.vector_search.client import VectorSearchClient
    try:
        vsc = VectorSearchClient()
        ep = vsc.get_endpoint(ENDPOINT)
        state = ep.get("endpoint_status", {}).get("state")
        if state == "ONLINE":
            results.append(("VS endpoint ONLINE (DB-02)", True, f"state={state}"))
        else:
            results.append(("VS endpoint ONLINE (DB-02)", False, f"state={state}"))
    except Exception as e:
        results.append(("VS endpoint ONLINE (DB-02)", False, str(e)))


def check_vs_index_and_smoke(results: list[tuple[str, bool, str]]) -> None:
    """Check 5: VS index is DIRECT_ACCESS + similarity_search returns warm-up row."""
    from databricks.vector_search.client import VectorSearchClient
    try:
        vsc = VectorSearchClient()
        index = vsc.get_index(endpoint_name=ENDPOINT, index_name=INDEX)
        desc = index.describe()
        idx_type = desc.get("index_type")
        if idx_type != "DIRECT_ACCESS":
            results.append(("VS index type (DB-02)", False,
                            f"expected DIRECT_ACCESS, got {idx_type!r}"))
            return
        results.append(("VS index type (DB-02)", True, "DIRECT_ACCESS"))

        # Smoke similarity_search query
        r = index.similarity_search(
            query_vector=[0.0] * EMBEDDING_DIMS,
            columns=["id"],
            num_results=1,
        )
        data = (r or {}).get("result", {}).get("data_array")
        if data:
            results.append(("VS similarity_search (DB-02)", True,
                            f"warm-up row found ({len(data)} result)"))
        else:
            results.append(("VS similarity_search (DB-02)", False,
                            "similarity_search returned empty — warm-up row missing?"))
    except Exception as e:
        results.append(("VS index + smoke query (DB-02)", False, str(e)))


def check_mlflow_experiment(results: list[tuple[str, bool, str]]) -> None:
    """Check 6: MLflow experiment exists and is active."""
    import mlflow
    try:
        mlflow.set_tracking_uri("databricks")
        exp = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT)
        if exp is None:
            results.append(("MLflow experiment (DB-05)", False, "experiment not found"))
        elif exp.lifecycle_stage != "active":
            results.append(("MLflow experiment (DB-05)", False,
                            f"lifecycle_stage={exp.lifecycle_stage!r}"))
        else:
            results.append(("MLflow experiment (DB-05)", True,
                            f"id={exp.experiment_id}, stage=active"))
    except Exception as e:
        results.append(("MLflow experiment (DB-05)", False, str(e)))


def check_secrets_scope(results: list[tuple[str, bool, str]]) -> None:
    """Check 7: demo-secrets scope reachable (DB-05)."""
    from databricks.sdk import WorkspaceClient
    try:
        w = WorkspaceClient()
        scopes = {s.name for s in w.secrets.list_scopes()}
        if SECRETS_SCOPE in scopes:
            results.append((f"Secrets scope '{SECRETS_SCOPE}' (DB-05)", True,
                            "reachable"))
        else:
            results.append((f"Secrets scope '{SECRETS_SCOPE}' (DB-05)", False,
                            f"not found; available: {scopes}"))
    except Exception as e:
        results.append((f"Secrets scope '{SECRETS_SCOPE}' (DB-05)", False, str(e)))


def main() -> int:
    """Run all verification checks. Returns 0 on all-green, 1 on any failure."""
    print("=" * 60)
    print("Arca Phase 0 — Asset Verification")
    print("=" * 60)
    print()

    host, token, http_path = _check_env()

    results: list[tuple[str, bool, str]] = []

    check_auth(results)
    check_schema_and_tables(host, token, http_path, results)
    check_cache_store_columns(host, token, http_path, results)
    check_usage_log_has_similarity_score(host, token, http_path, results)
    check_vs_endpoint_online(results)
    check_vs_index_and_smoke(results)
    check_mlflow_experiment(results)
    check_secrets_scope(results)

    print()
    failures = 0
    for name, ok, detail in results:
        status = "[OK]  " if ok else "[FAIL]"
        print(f"  {status} {name}")
        if detail:
            print(f"         {detail}")
        if not ok:
            failures += 1

    print()
    if failures == 0:
        print(f"All {len(results)} checks passed.")
        return 0
    else:
        print(f"{failures}/{len(results)} check(s) FAILED.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
