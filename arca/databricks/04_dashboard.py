"""OBS-03: Databricks Lakeview (AI/BI) dashboard for Arca cost analytics.

Provides:
    - export_template(template_dashboard_id): one-shot export of a UI-built
      template dashboard to arca/databricks/dashboard_definition.json
    - create_dashboard_if_missing(): idempotent create + publish of the
      production dashboard named "arca-cost-analytics"
    - main(): CLI entry — runs create_dashboard_if_missing() and prints
      DASHBOARD_URL=<url>

Legacy DBSQL dashboards were deprecated 2026-01-12; this module uses the
Lakeview (/api/2.0/lakeview/) API exclusively.

Filename note: this module is named ``04_dashboard.py`` to match the Phase 0
bootstrap naming convention (``00_bootstrap.py``). Because the leading digit
makes ``from arca.databricks.04_dashboard import ...`` invalid Python syntax,
callers that need programmatic access should load it via ``importlib``
(see ``tests/test_observability.py::test_create_dashboard_integration``).
"""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import structlog

_log = structlog.get_logger(__name__)

DASHBOARD_NAME = "arca-cost-analytics"
DASHBOARD_DEF_PATH = Path(__file__).parent / "dashboard_definition.json"


def export_template(
    template_dashboard_id: str,
    out_path: str | os.PathLike[str] = DASHBOARD_DEF_PATH,
) -> None:
    """Export a UI-built template dashboard's JSON definition to disk.

    Run this once after manually building the template dashboard in the
    Databricks UI (name it with a ``-template`` suffix). The exported JSON
    is then committed so :func:`create_dashboard_if_missing` can reproduce
    the dashboard programmatically.
    """
    from databricks.sdk import WorkspaceClient

    w = WorkspaceClient()
    d = w.lakeview.get(template_dashboard_id)
    # serialized_dashboard is a JSON string; parse and re-dump indented for diffability
    parsed = json.loads(d.serialized_dashboard)
    out_path = Path(out_path)
    with open(out_path, "w") as f:
        json.dump(parsed, f, indent=2, sort_keys=True)
    print(f"Exported {out_path} ({len(d.serialized_dashboard)} bytes)")


def _load_def() -> str:
    """Read dashboard_definition.json and return it as a JSON string.

    We deliberately return the string form (not a dict) because the Lakeview
    API's ``serialized_dashboard`` parameter REQUIRES a JSON string, not a
    dict (RESEARCH.md Pitfall 6 — double-encoding trap).
    """
    with open(DASHBOARD_DEF_PATH) as f:
        return f.read()


def _get_warehouse_id() -> str:
    """Pick the first available SQL warehouse; env override ARCA_WAREHOUSE_ID wins."""
    override = os.environ.get("ARCA_WAREHOUSE_ID")
    if override:
        return override
    from databricks.sdk import WorkspaceClient

    w = WorkspaceClient()
    warehouses = list(w.warehouses.list())
    if not warehouses:
        raise RuntimeError("No SQL warehouse available; set ARCA_WAREHOUSE_ID")
    return warehouses[0].id


def _create_sync() -> str | None:
    """Synchronous core of create_dashboard_if_missing (wrapped in to_thread)."""
    from databricks.sdk import WorkspaceClient

    w = WorkspaceClient()

    # Resolve parent_path: env override → current user home → /Shared fallback
    parent_path = os.environ.get("ARCA_DASHBOARD_PARENT")
    if not parent_path:
        try:
            email = w.current_user.me().user_name
            parent_path = f"/Users/{email}"
        except Exception:
            parent_path = "/Shared"

    # Idempotency check — return existing id if a dashboard with DASHBOARD_NAME exists
    for d in w.lakeview.list():
        if d.display_name == DASHBOARD_NAME:
            _log.info("dashboard_already_exists", dashboard_id=d.dashboard_id)
            return d.dashboard_id
    # Create (load the JSON string; validate it parses)
    serialized = _load_def()
    try:
        json.loads(serialized)  # validates it's valid JSON
    except json.JSONDecodeError:
        # Defense in depth: if file got stored as something exotic, re-serialize
        serialized = json.dumps(json.loads(serialized))
    warehouse_id = _get_warehouse_id()
    result = w.lakeview.create(
        display_name=DASHBOARD_NAME,
        serialized_dashboard=serialized,
        warehouse_id=warehouse_id,
        parent_path=parent_path,
    )
    # Publish so it's viewable at /sql/dashboardsv3/<id>
    w.lakeview.publish(dashboard_id=result.dashboard_id)
    _log.info(
        "dashboard_created",
        dashboard_id=result.dashboard_id,
        warehouse_id=warehouse_id,
    )
    return result.dashboard_id


async def create_dashboard_if_missing() -> str | None:
    """Idempotently create (or find) the arca-cost-analytics Lakeview dashboard.

    Returns the dashboard_id on success, ``None`` on failure (e.g. missing
    Databricks credentials). Never raises — all errors are logged as warnings
    so the proxy bootstrap path can degrade gracefully.
    """
    try:
        return await asyncio.to_thread(_create_sync)
    except Exception as exc:  # noqa: BLE001 — graceful degrade
        _log.warning("dashboard_create_failed", err=str(exc))
        return None


async def main() -> None:
    """CLI entry: create/find dashboard and print DASHBOARD_URL=<url>."""
    dashboard_id = await create_dashboard_if_missing()
    if dashboard_id:
        host = os.environ.get("DATABRICKS_HOST", "<DATABRICKS_HOST>").rstrip("/")
        url = f"{host}/sql/dashboardsv3/{dashboard_id}"
        print(f"DASHBOARD_URL={url}")
    else:
        print("DASHBOARD_URL=<failed; check logs>")


if __name__ == "__main__":
    asyncio.run(main())
