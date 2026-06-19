"""Arca CLI commands. Entry point: `arca`.

Phase 4 adds: `arca tail` (OBS-04 — live cache event stream).
Phase 5 adds: arca start, arca stop, arca init.
Phase 5 (Plan 03) will add: arca stats, arca doctor.
"""
from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import signal
import socket
import subprocess
import sys
from pathlib import Path

import typer
import httpx

from arca.config import get_settings
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from rich.console import Console
from rich.table import Table
from rich.text import Text

# Module-level imports (exposed so tests can patch arca.cli.<name>).
# Best-effort: if the dependency is missing at import time we set the
# attribute to None — the runtime guards handle the missing-dep case.
try:
    from arca.databricks.bootstrap import bootstrap as bootstrap  # noqa: F401
except Exception:
    bootstrap = None  # type: ignore

# Lazy: importing sentence-transformers pulls torch (~2-4s). The CLI must
# start fast for `arca stats` / `arca tail`; only `arca init` needs the model.
# Module-level None so tests can patch arca.cli.SentenceTransformer.
SentenceTransformer = None  # type: ignore

# For arca stats (Plan 03) Delta fallback — expose at module level for patching
try:
    from databricks.sql import connect as databricks_sql_connect  # noqa: F401
except Exception:
    databricks_sql_connect = None  # type: ignore


app = typer.Typer(help="Arca — local Claude Code proxy with Databricks-backed semantic cache.")
_console = Console()


# ---------------------------------------------------------------------------
# PID file helpers
# ---------------------------------------------------------------------------
_PID_DIR_NAME = ".arca"
_PID_FILE_NAME = "arca.pid"


def _pid_file_path() -> Path:
    """Compute PID file path — respects HOME env var (tests monkeypatch HOME)."""
    return Path.home() / _PID_DIR_NAME / _PID_FILE_NAME


def _check_torch_guard() -> None:
    """Exit 1 with install instruction if torch is not installed."""
    if importlib.util.find_spec("torch") is None:
        _console.print(
            "[red]torch is not installed. "
            "Run: pip install torch==2.4.* --index-url https://download.pytorch.org/whl/cpu[/red]",
            soft_wrap=True,
        )
        raise typer.Exit(1)


def _render_event(event: dict, console: Console) -> None:
    """Render one cache event as a colorized one-line summary."""
    hit = bool(event.get("cache_hit"))
    color = "green" if hit else "yellow"
    label = "HIT " if hit else "MISS"
    t = Text()
    t.append(f"[{label}] ", style=f"bold {color}")
    latency = event.get("latency_ms")
    if latency is not None:
        t.append(f"latency={latency}ms  ")
    sim = event.get("similarity_score")
    if sim is not None:
        try:
            t.append(f"sim={float(sim):.3f}  ")
        except (TypeError, ValueError):
            t.append("sim=—  ")
    else:
        t.append("sim=—  ")
    cost = float(event.get("cost_usd") or 0.0)
    saved = float(event.get("cost_saved_usd") or 0.0)
    t.append(f"cost=${cost:.6f}  ")
    t.append(f"saved=${saved:.6f}", style="bold green" if saved > 0 else "")
    model = event.get("model") or ""
    if model:
        t.append(f"  {model}", style="dim")
    console.print(t)


@app.command()
def tail(
    url: str = typer.Option(
        "http://localhost:8082/tail",
        help="Proxy /tail endpoint. Override if proxy runs on a non-default port.",
    ),
) -> None:
    """Stream live cache events from the running proxy."""
    _console.print(f"[dim]Connecting to {url}...[/dim]")
    try:
        with httpx.stream("GET", url, timeout=None) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line:
                    continue
                if line.startswith(":"):
                    continue  # keep-alive
                if line.startswith("data:"):
                    try:
                        event = json.loads(line[5:].strip())
                    except json.JSONDecodeError:
                        continue
                    _render_event(event, _console)
    except KeyboardInterrupt:
        _console.print("\n[dim]Disconnected.[/dim]")
        sys.exit(0)
    except httpx.ConnectError:
        _console.print("[red]Could not connect to proxy. Is `arca start` running?[/red]")
        sys.exit(1)


@app.command()
def start(
    port: int = typer.Option(8082, help="Port for the proxy to bind on localhost."),
) -> None:
    """Launch the Arca proxy as a detached background process."""
    _check_torch_guard()

    pid_file = _pid_file_path()
    pid_file.parent.mkdir(parents=True, exist_ok=True)

    # Stale-PID guard
    if pid_file.exists():
        try:
            existing_pid = int(pid_file.read_text().strip())
        except ValueError:
            existing_pid = -1
        if existing_pid > 0:
            try:
                os.kill(existing_pid, 0)  # probe existence (signal 0)
                _console.print(
                    f"[yellow]Arca already running (PID {existing_pid}). "
                    f"Use 'arca stop' first.[/yellow]",
                    soft_wrap=True,
                )
                raise typer.Exit(0)
            except ProcessLookupError:
                pid_file.unlink(missing_ok=True)  # stale — proceed

    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "arca.proxy:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--workers",
            "1",
            "--log-level",
            "warning",
        ],
        start_new_session=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    pid_file.write_text(str(proc.pid))

    # Poll port for up to 3s
    import time as _time

    deadline = _time.time() + 3.0
    booted = False
    while _time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.1):
                booted = True
                break
        except OSError:
            _time.sleep(0.1)

    if not booted:
        _console.print(
            f"[red]Proxy failed to start. Check that port {port} is free and torch is installed.[/red]",
            soft_wrap=True,
        )
        pid_file.unlink(missing_ok=True)
        raise typer.Exit(1)

    _console.print(f"[green]Arca started (PID {proc.pid})[/green]")
    _console.print(f"  export ANTHROPIC_BASE_URL=http://localhost:{port}")


@app.command()
def stop() -> None:
    """Gracefully shut down the Arca proxy via PID file."""
    pid_file = _pid_file_path()
    if not pid_file.exists():
        _console.print(
            f"Arca is not running (no PID file found at {pid_file}).",
            soft_wrap=True,
        )
        raise typer.Exit(1)
    try:
        pid = int(pid_file.read_text().strip())
    except ValueError:
        pid_file.unlink(missing_ok=True)
        _console.print("PID file was corrupt — removed.")
        raise typer.Exit(1)

    try:
        os.kill(pid, signal.SIGTERM)
        _console.print(f"Arca stopped (PID {pid}).")
    except ProcessLookupError:
        _console.print(
            f"Process {pid} not found — proxy may have already stopped. PID file removed.",
            soft_wrap=True,
        )
    finally:
        pid_file.unlink(missing_ok=True)


@app.command()
def init() -> None:
    """Provision Databricks schema, VS index, and download the embedding model."""
    _check_torch_guard()

    _console.print("Provisioning Delta schema and tables...", soft_wrap=True)
    try:
        bootstrap(skip_vs_endpoint=True)
    except Exception as exc:
        _console.print(
            f"[red]Init failed during schema provisioning: {exc}[/red]",
            soft_wrap=True,
        )
        raise typer.Exit(1)

    # VS endpoint already ONLINE from Phase 0; MLflow experiment is provisioned
    # inside bootstrap(). Print the UI-SPEC step lines so the user sees what happened.
    _console.print(
        "Creating Vector Search index (Direct Access, 384 dims)...",
        soft_wrap=True,
    )
    _console.print("Setting up MLflow experiment...", soft_wrap=True)
    _console.print(
        "Downloading embedding model (all-MiniLM-L6-v2) — first run only...",
        soft_wrap=True,
    )
    try:
        from arca.embeddings import EMBEDDING_MODEL
        st_cls = SentenceTransformer
        if st_cls is None:
            from sentence_transformers import SentenceTransformer as st_cls
        st_cls(EMBEDDING_MODEL, device="cpu")
    except ImportError:
        _check_torch_guard()
    except Exception as exc:
        _console.print(
            f"[yellow]Embedding model download failed ({exc}). "
            "Re-run 'arca init' with network access; the proxy will also "
            "download it on first start.[/yellow]",
            soft_wrap=True,
        )

    _console.print("Provisioning Lakeview cost-analytics dashboard...", soft_wrap=True)
    try:
        from arca.databricks import dashboard as _dash_mod
        dashboard_id = asyncio.run(_dash_mod.create_dashboard_if_missing())
        if dashboard_id:
            host = os.environ.get("DATABRICKS_HOST", "").rstrip("/")
            _console.print(
                f"  Dashboard: {host}/sql/dashboardsv3/{dashboard_id}",
                soft_wrap=True,
            )
        else:
            _console.print(
                "  [yellow]Dashboard skipped (check DATABRICKS_HOST / credentials).[/yellow]",
                soft_wrap=True,
            )
    except Exception as _e:
        _console.print(f"  [yellow]Dashboard skipped: {_e}[/yellow]", soft_wrap=True)

    _console.print(
        "[green]Init complete. Run 'arca doctor' to verify all checks pass.[/green]",
        soft_wrap=True,
    )


# ---------------------------------------------------------------------------
# Plan 03: arca stats
# ---------------------------------------------------------------------------
def _render_stats_table(data: dict, *, prefix_notice: str | None = None) -> None:
    if prefix_notice:
        _console.print(prefix_notice, soft_wrap=True)
    table = Table(title="Arca Session Stats", show_header=True, header_style="bold", padding=(0, 2))
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("Total calls", str(int(data.get("total_calls", 0))))
    table.add_row("Cache hits", str(int(data.get("cache_hits", 0))))
    table.add_row("Cache misses", str(int(data.get("cache_misses", 0))))
    table.add_row("Hit rate", f"{float(data.get('hit_rate_pct', 0.0))}%")
    table.add_row("Total cost saved", f"${float(data.get('cost_saved_usd', 0.0)):.6f}")
    _console.print(table)


def _stats_from_delta() -> dict | None:
    """Run Delta SQL aggregate over usage_log. Returns dict or None on failure."""
    host = os.environ.get("DATABRICKS_HOST", "").replace("https://", "").replace("http://", "")
    token = os.environ.get("DATABRICKS_TOKEN", "")
    http_path = os.environ.get("DATABRICKS_HTTP_PATH", "")
    # Allow Delta path even if env vars missing (tests patch databricks_sql_connect directly)
    try:
        conn = databricks_sql_connect(
            server_hostname=host, http_path=http_path, access_token=token
        )
    except Exception:
        return None
    try:
        with conn:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT
                    COUNT(*) AS total_calls,
                    SUM(CASE WHEN cache_hit THEN 1 ELSE 0 END) AS cache_hits,
                    SUM(CASE WHEN NOT cache_hit THEN 1 ELSE 0 END) AS cache_misses,
                    ROUND(100.0 * SUM(CASE WHEN cache_hit THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0), 1) AS hit_rate_pct,
                    COALESCE(SUM(cost_saved_usd), 0.0) AS cost_saved_usd
                FROM {table}
                """.format(table=get_settings().usage_table)
            )
            row = cur.fetchone()
            if row is None:
                return None
            cols = [d[0] for d in cur.description]
            rec = dict(zip(cols, row))
            return {
                "total_calls": rec.get("total_calls") or 0,
                "cache_hits": rec.get("cache_hits") or 0,
                "cache_misses": rec.get("cache_misses") or 0,
                "hit_rate_pct": rec.get("hit_rate_pct") or 0.0,
                "cost_saved_usd": rec.get("cost_saved_usd") or 0.0,
            }
    except Exception:
        return None


@app.command()
def stats(port: int = typer.Option(8082, help="Proxy port for live /stats endpoint.")) -> None:
    """Show session stats from running proxy or from all-time Delta aggregate."""
    url = f"http://localhost:{port}/stats"

    # Path A — live proxy
    live_attempted = False
    try:
        resp = httpx.get(url, timeout=2.0)
        resp.raise_for_status()
        data = resp.json()
        live_attempted = True
        if int(data.get("total_calls", 0)) == 0:
            delta = _stats_from_delta()
            if delta and int(delta.get("total_calls", 0)) > 0:
                _render_stats_table(delta, prefix_notice="Proxy not running — showing all-time Delta aggregates.")
                return
            _console.print("No usage data found. Has 'arca start' been run and requests sent?", soft_wrap=True)
            return
        _render_stats_table(data)
        return
    except httpx.ConnectError:
        pass
    except Exception:
        if live_attempted:
            # Live path returned but failed post-parse — still try Delta
            pass

    # Path B — Delta fallback
    delta = _stats_from_delta()
    if delta is None:
        _console.print("No usage data found. Has 'arca start' been run and requests sent?", soft_wrap=True)
        return
    if int(delta.get("total_calls", 0)) == 0:
        _console.print("No usage data found. Has 'arca start' been run and requests sent?", soft_wrap=True)
        return
    _render_stats_table(delta, prefix_notice="Proxy not running — showing all-time Delta aggregates.")


# ---------------------------------------------------------------------------
# Plan 03: arca doctor
# ---------------------------------------------------------------------------
def _doctor_databricks_auth(timeout: int = 5) -> str:
    """Raise on auth failure; return detail string on success."""
    from databricks.sdk import WorkspaceClient
    host = os.environ.get("DATABRICKS_HOST", "")
    w = WorkspaceClient()
    w.current_user.me()
    return f"workspace={host}"


def _doctor_vs_index(timeout: int = 5) -> str:
    from databricks.vector_search.client import VectorSearchClient
    settings = get_settings()
    vsc = VectorSearchClient()
    vsc.get_index(endpoint_name=settings.vs_endpoint, index_name=settings.vs_index)
    return f"index {settings.vs_index} reachable"


def _doctor_mlflow(timeout: int = 5) -> str:
    import mlflow
    from arca.observability import _mlflow_experiment_path
    mlflow.set_tracking_uri("databricks")
    path = _mlflow_experiment_path()
    exp = mlflow.get_experiment_by_name(path)
    if exp is None:
        raise RuntimeError(f"MLflow experiment not found at {path} — run 'arca init'")
    return f"experiment={path}"


def _doctor_proxy_routing(timeout: int = 5) -> str:
    port = int(os.environ.get("ARCA_PORT", "8082"))
    r = httpx.get(f"http://127.0.0.1:{port}/health", timeout=timeout)
    r.raise_for_status()
    return f"localhost:{port} reachable"


@app.command()
def doctor() -> None:
    """Validate all Arca integration points and print a pass/fail report."""
    checks = [
        ("Databricks auth",     _doctor_databricks_auth),
        ("Vector Search index", _doctor_vs_index),
        ("MLflow experiment",   _doctor_mlflow),
        ("Proxy routing",       _doctor_proxy_routing),
    ]

    table = Table(title="Arca Health Check", show_header=True, header_style="bold", padding=(0, 2))
    table.add_column("Check")
    table.add_column("Status")
    table.add_column("Detail")

    failed = 0
    for name, fn in checks:
        try:
            with ThreadPoolExecutor(max_workers=1) as ex:
                future = ex.submit(fn, 5)
                detail = future.result(timeout=5)
            table.add_row(name, "[green]PASS[/green]", str(detail))
        except FuturesTimeoutError:
            table.add_row(name, "[red]FAIL[/red]", "timed out after 5s")
            failed += 1
        except TimeoutError:
            table.add_row(name, "[red]FAIL[/red]", "timed out after 5s")
            failed += 1
        except Exception as exc:
            detail = str(exc)[:120] or type(exc).__name__
            table.add_row(name, "[red]FAIL[/red]", detail)
            failed += 1

    _console.print(table)
    if failed:
        _console.print(
            f"[red]{failed} check(s) failed. Run 'arca init' if not yet initialized.[/red]",
            soft_wrap=True,
        )
        raise typer.Exit(1)
    _console.print("[green]All checks passed.[/green]", soft_wrap=True)


if __name__ == "__main__":
    app()
