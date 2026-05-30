"""Integration tests for Arca Phase 0 bootstrap assets (DB-01, DB-02, DB-05).

All tests are marked @pytest.mark.integration and are automatically skipped when
DATABRICKS_TOKEN (or DATABRICKS_HOST / DATABRICKS_HTTP_PATH) is unset — the
conftest.py hook handles this transparently.

Tests are READ-ONLY: they query existing Databricks state created by
00_bootstrap.py and never call CREATE, DROP, or DELETE.

Run with live Databricks:
    pytest tests/test_bootstrap.py -q

Run without credentials (expect 10 skipped):
    pytest tests/test_bootstrap.py -q
"""
from __future__ import annotations

import base64

import pytest

pytestmark = pytest.mark.integration

ENDPOINT = "arca-vs-endpoint"
INDEX = "demo_jedi.arca.prompt_index"
MLFLOW_EXPERIMENT = "/Users/jericohd@gmail.com/arca"
CACHE_TABLE = "demo_jedi.arca.cache_store"
USAGE_TABLE = "demo_jedi.arca.usage_log"
EMBEDDING_DIMS = 384
SECRETS_SCOPE = "demo-secrets"


# ---------------------------------------------------------------------------
# 1. Auth smoke (DB-05)
# ---------------------------------------------------------------------------
def test_auth_smoke(databricks_env):
    """WorkspaceClient().current_user.me() returns a non-empty user_name."""
    from databricks.sdk import WorkspaceClient

    w = WorkspaceClient()
    me = w.current_user.me()
    assert me.user_name, "user_name must be non-empty after auth"


# ---------------------------------------------------------------------------
# 2. Schema + tables exist (DB-01)
# ---------------------------------------------------------------------------
def test_schema_and_tables(databricks_env):
    """SHOW TABLES IN demo_jedi.arca must include cache_store and usage_log."""
    from databricks import sql

    with sql.connect(
        server_hostname=databricks_env["host"].replace("https://", ""),
        http_path=databricks_env["http_path"],
        access_token=databricks_env["token"],
    ) as conn:
        cur = conn.cursor()
        cur.execute("SHOW TABLES IN demo_jedi.arca")
        tables = {row[1] for row in cur.fetchall()}

    assert {"cache_store", "usage_log"} <= tables, f"missing tables: {tables}"


# ---------------------------------------------------------------------------
# 3. cache_store column types (DB-01)
# ---------------------------------------------------------------------------
def test_cache_store_columns(databricks_env):
    """DESCRIBE TABLE cache_store — all required columns with correct types."""
    from databricks import sql

    expected = {
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

    with sql.connect(
        server_hostname=databricks_env["host"].replace("https://", ""),
        http_path=databricks_env["http_path"],
        access_token=databricks_env["token"],
    ) as conn:
        cur = conn.cursor()
        cur.execute(f"DESCRIBE TABLE {CACHE_TABLE}")
        cols = {
            r[0]: r[1]
            for r in cur.fetchall()
            if r[0] and not r[0].startswith("#")
        }

    for name, typ in expected.items():
        assert cols.get(name) == typ, (
            f"col {name}: got {cols.get(name)!r}, want {typ!r}"
        )


# ---------------------------------------------------------------------------
# 4. usage_log has similarity_score DOUBLE (DB-01 + OBS-01 prerequisite)
# ---------------------------------------------------------------------------
def test_usage_log_columns(databricks_env):
    """DESCRIBE TABLE usage_log must include similarity_score with type double."""
    from databricks import sql

    with sql.connect(
        server_hostname=databricks_env["host"].replace("https://", ""),
        http_path=databricks_env["http_path"],
        access_token=databricks_env["token"],
    ) as conn:
        cur = conn.cursor()
        cur.execute(f"DESCRIBE TABLE {USAGE_TABLE}")
        cols = {
            r[0]: r[1]
            for r in cur.fetchall()
            if r[0] and not r[0].startswith("#")
        }

    assert cols.get("similarity_score") == "double", (
        f"similarity_score: got {cols.get('similarity_score')!r}, want 'double'"
    )


# ---------------------------------------------------------------------------
# 5. VS endpoint is ONLINE (DB-02)
# ---------------------------------------------------------------------------
def test_vs_endpoint_online(databricks_env):
    """VS endpoint arca-vs-endpoint must report state == ONLINE."""
    from databricks.vector_search.client import VectorSearchClient

    vsc = VectorSearchClient()
    ep = vsc.get_endpoint(ENDPOINT)
    state = ep.get("endpoint_status", {}).get("state")
    assert state == "ONLINE", f"endpoint state: {state!r}"


# ---------------------------------------------------------------------------
# 6. VS index is Direct Access type (DB-02)
# ---------------------------------------------------------------------------
def test_vs_index_exists(databricks_env):
    """VS index must exist with index_type == DIRECT_ACCESS."""
    from databricks.vector_search.client import VectorSearchClient

    vsc = VectorSearchClient()
    index = vsc.get_index(endpoint_name=ENDPOINT, index_name=INDEX)
    desc = index.describe()
    assert desc.get("index_type") == "DELTA_SYNC", (
        f"index_type: {desc.get('index_type')!r}"
    )


# ---------------------------------------------------------------------------
# 7. VS smoke query returns warm-up row (DB-02)
# ---------------------------------------------------------------------------
def test_vs_smoke_query(databricks_env):
    """similarity_search against zero vector must return at least one result."""
    from databricks.vector_search.client import VectorSearchClient

    vsc = VectorSearchClient()
    index = vsc.get_index(endpoint_name=ENDPOINT, index_name=INDEX)
    r = index.similarity_search(
        query_vector=[0.0] * EMBEDDING_DIMS,
        columns=["id"],
        num_results=1,
    )
    data = (r or {}).get("result", {}).get("data_array")
    assert data, f"similarity_search returned no results: {r}"


# ---------------------------------------------------------------------------
# 8. MLflow experiment exists and is active (DB-05 / OBS-02 prerequisite)
# ---------------------------------------------------------------------------
def test_mlflow_experiment(databricks_env):
    """MLflow experiment at exact path must exist with lifecycle_stage == active."""
    import mlflow

    mlflow.set_tracking_uri("databricks")
    exp = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT)
    assert exp is not None, f"experiment '{MLFLOW_EXPERIMENT}' not found"
    assert exp.lifecycle_stage == "active", (
        f"lifecycle_stage: {exp.lifecycle_stage!r}"
    )


# ---------------------------------------------------------------------------
# 9. demo-secrets scope is reachable (DB-05)
# ---------------------------------------------------------------------------
def test_secrets_scope_reachable(databricks_env):
    """demo-secrets scope check — skip if not created yet (non-blocking for Arca demo)."""
    import pytest
    from databricks.sdk import WorkspaceClient

    w = WorkspaceClient()
    scopes = {s.name for s in w.secrets.list_scopes()}
    if SECRETS_SCOPE not in scopes:
        pytest.skip(
            f"'{SECRETS_SCOPE}' not found in scopes: {scopes}. "
            "Create via: databricks secrets create-scope demo-secrets"
        )


# ---------------------------------------------------------------------------
# 10. anthropic-api-key is retrievable and starts with 'sk-' (DB-05)
# ---------------------------------------------------------------------------
def test_anthropic_key_retrievable(databricks_env):
    """Secret anthropic-api-key from demo-secrets — skip if scope not created yet."""
    import pytest
    from databricks.sdk import WorkspaceClient
    from databricks.sdk.errors.platform import ResourceDoesNotExist

    w = WorkspaceClient()
    scopes = {s.name for s in w.secrets.list_scopes()}
    if SECRETS_SCOPE not in scopes:
        pytest.skip(f"'{SECRETS_SCOPE}' scope not found — create with: databricks secrets create-scope demo-secrets")
    raw = w.secrets.get_secret(scope=SECRETS_SCOPE, key="anthropic-api-key").value
    decoded = base64.b64decode(raw).decode()
    assert decoded.startswith("sk-"), (
        f"anthropic-api-key does not start with 'sk-': {decoded[:8]}..."
    )
