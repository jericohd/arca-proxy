"""Typer-safe bootstrap entry point — wraps bootstrap_impl helpers without sys.exit.

Used by:
  - arca/cli.py  ->  arca init           (raises exceptions; Typer handles)
  - arca/databricks/bootstrap_impl.py    (CLI-style; main() calls this + sys.exit)
"""
from __future__ import annotations


class BootstrapError(RuntimeError):
    """Bootstrap failure — caller decides how to present / exit."""

    def __init__(self, message: str, exit_code: int = 1) -> None:
        super().__init__(message)
        self.exit_code = exit_code


def bootstrap(skip_vs_endpoint: bool = False) -> dict:
    """Run idempotent Delta + MLflow (+ optional VS endpoint) provisioning.

    Args:
        skip_vs_endpoint: if True, do NOT create or gate the VS endpoint.
            `arca init` passes True (endpoint is already ONLINE from Phase 0;
            re-provisioning would take 10-15 min). `python -m ...bootstrap_impl`
            passes False to do full end-to-end provisioning.

    Returns:
        dict with keys: {experiment_id: str, endpoint_online: bool|None,
                         index_created: bool|None}

    Raises:
        BootstrapError(exit_code=2): Missing required environment variables.
        BootstrapError(exit_code=3): VS endpoint timeout.
        Any other exception from DDL / MLflow / VS: propagated unchanged.
    """
    import os
    from arca.databricks import _bootstrap_impl as _impl  # alias configured in package __init__

    # Env var check — previously sys.exit(2)
    host = os.environ.get("DATABRICKS_HOST", "")
    token = os.environ.get("DATABRICKS_TOKEN", "")
    http_path = os.environ.get("DATABRICKS_HTTP_PATH", "")
    missing = [
        v
        for v, val in (
            ("DATABRICKS_HOST", host),
            ("DATABRICKS_TOKEN", token),
            ("DATABRICKS_HTTP_PATH", http_path),
        )
        if not val
    ]
    if missing:
        raise BootstrapError(
            f"Missing required environment variables: {', '.join(missing)}. "
            "Set DATABRICKS_HOST, DATABRICKS_TOKEN, DATABRICKS_HTTP_PATH before running.",
            exit_code=2,
        )

    result: dict = {"experiment_id": None, "endpoint_online": None, "index_created": None}

    # DDL — reuses _run_ddl from bootstrap_impl module
    _impl._run_ddl(host, token, http_path)

    # MLflow — reuses _setup_mlflow
    result["experiment_id"] = _impl._setup_mlflow()

    if skip_vs_endpoint:
        return result

    # Full provision path (module CLI)
    from databricks.vector_search.client import VectorSearchClient

    vsc = VectorSearchClient()
    _impl._kick_endpoint_creation(vsc)
    try:
        _impl._gate_endpoint_online(vsc)
        result["endpoint_online"] = True
    except SystemExit as e:  # legacy bootstrap_impl may still raise SystemExit
        raise BootstrapError(
            "VS endpoint did not reach ONLINE within 900s. "
            "Cache runs in degraded local mode (SQLite L2 fallback); "
            "re-run bootstrap once the endpoint is reachable.",
            exit_code=3,
        ) from e

    _impl._create_vs_index(vsc)
    result["index_created"] = True
    return result
