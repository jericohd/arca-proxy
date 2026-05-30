"""Phase 5 CLI tests — CLI-01..05 + PKG-01. RED until Plans 02/03 implement."""
from __future__ import annotations

import os
import signal
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from arca.cli import app

runner = CliRunner()


# ---------- CLI-01: arca start ----------

def test_start_writes_pid(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    fake_proc = MagicMock(pid=12345)
    with patch("arca.cli.subprocess.Popen", return_value=fake_proc) as popen, \
         patch("arca.cli.socket.create_connection", return_value=MagicMock(__enter__=lambda s: s, __exit__=lambda *a: None)), \
         patch("arca.cli.importlib.util.find_spec", return_value=object()):
        result = runner.invoke(app, ["start"])
    assert result.exit_code == 0, result.output
    pid_file = tmp_path / ".arca" / "arca.pid"
    assert pid_file.exists()
    assert pid_file.read_text().strip() == "12345"
    assert "Arca started (PID 12345)" in result.output
    assert "export ANTHROPIC_BASE_URL=http://localhost:8082" in result.output
    popen.assert_called_once()


def test_start_already_running(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    pid_file = tmp_path / ".arca" / "arca.pid"
    pid_file.parent.mkdir(parents=True)
    pid_file.write_text("4242")
    with patch("arca.cli.os.kill", return_value=None), \
         patch("arca.cli.importlib.util.find_spec", return_value=object()):
        result = runner.invoke(app, ["start"])
    assert result.exit_code == 0
    assert "Arca already running (PID 4242). Use 'arca stop' first." in result.output


def test_start_stale_pid(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    pid_file = tmp_path / ".arca" / "arca.pid"
    pid_file.parent.mkdir(parents=True)
    pid_file.write_text("99999999")
    fake_proc = MagicMock(pid=7777)

    def _kill(pid, sig):
        raise ProcessLookupError()

    with patch("arca.cli.os.kill", side_effect=_kill), \
         patch("arca.cli.subprocess.Popen", return_value=fake_proc), \
         patch("arca.cli.socket.create_connection", return_value=MagicMock(__enter__=lambda s: s, __exit__=lambda *a: None)), \
         patch("arca.cli.importlib.util.find_spec", return_value=object()):
        result = runner.invoke(app, ["start"])
    assert result.exit_code == 0
    assert pid_file.read_text().strip() == "7777"


def test_start_no_torch(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    with patch("arca.cli.importlib.util.find_spec", return_value=None):
        result = runner.invoke(app, ["start"])
    assert result.exit_code == 1
    assert "torch is not installed" in result.output
    assert "pip install torch==2.4.* --index-url https://download.pytorch.org/whl/cpu" in result.output


# ---------- CLI-02: arca stop ----------

def test_stop_running(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    pid_file = tmp_path / ".arca" / "arca.pid"
    pid_file.parent.mkdir(parents=True)
    pid_file.write_text("5555")
    with patch("arca.cli.os.kill") as kill:
        result = runner.invoke(app, ["stop"])
    assert result.exit_code == 0
    kill.assert_called_once_with(5555, signal.SIGTERM)
    assert not pid_file.exists()
    assert "Arca stopped (PID 5555)." in result.output


def test_stop_not_running(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    result = runner.invoke(app, ["stop"])
    assert result.exit_code == 1
    assert "Arca is not running (no PID file found" in result.output


def test_stop_stale_pid(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    pid_file = tmp_path / ".arca" / "arca.pid"
    pid_file.parent.mkdir(parents=True)
    pid_file.write_text("6666")
    with patch("arca.cli.os.kill", side_effect=ProcessLookupError()):
        result = runner.invoke(app, ["stop"])
    assert result.exit_code == 0
    assert not pid_file.exists()
    assert "Process 6666 not found" in result.output
    assert "PID file removed" in result.output


# ---------- CLI-03: arca stats ----------

def test_stats_live(monkeypatch):
    fake_resp = MagicMock(status_code=200)
    fake_resp.json.return_value = {
        "total_calls": 10, "cache_hits": 7, "cache_misses": 3,
        "hit_rate_pct": 70.0, "cost_saved_usd": 0.42,
    }
    fake_resp.raise_for_status = MagicMock()
    with patch("arca.cli.httpx.get", return_value=fake_resp):
        result = runner.invoke(app, ["stats"])
    assert result.exit_code == 0
    assert "Arca Session Stats" in result.output
    assert "Total calls" in result.output
    assert "Hit rate" in result.output
    assert "70.0%" in result.output or "70%" in result.output


def test_stats_delta_fallback(monkeypatch):
    import httpx as _httpx
    fake_cursor = MagicMock()
    fake_cursor.fetchone.return_value = (100, 55, 45, 55.0, 1.234567)
    fake_cursor.description = [("total_calls",), ("cache_hits",), ("cache_misses",), ("hit_rate_pct",), ("cost_saved_usd",)]
    fake_conn = MagicMock()
    fake_conn.cursor.return_value = fake_cursor
    fake_conn.__enter__ = lambda s: s
    fake_conn.__exit__ = lambda *a: None
    with patch("arca.cli.httpx.get", side_effect=_httpx.ConnectError("refused")), \
         patch("arca.cli.databricks_sql_connect", return_value=fake_conn):
        result = runner.invoke(app, ["stats"])
    assert result.exit_code == 0
    assert "Proxy not running — showing all-time Delta aggregates." in result.output
    assert "Arca Session Stats" in result.output


def test_stats_no_data(monkeypatch):
    import httpx as _httpx
    with patch("arca.cli.httpx.get", side_effect=_httpx.ConnectError("refused")), \
         patch("arca.cli.databricks_sql_connect", side_effect=Exception("no creds")):
        result = runner.invoke(app, ["stats"])
    assert "No usage data found. Has 'arca start' been run and requests sent?" in result.output


# ---------- CLI-04: arca doctor ----------

def test_doctor_all_pass(monkeypatch):
    with patch("arca.cli._doctor_databricks_auth", return_value="workspace=https://x"), \
         patch("arca.cli._doctor_vs_index", return_value="index ONLINE, 384 dims"), \
         patch("arca.cli._doctor_mlflow", return_value="experiment=/Users/jericohd@gmail.com/arca"), \
         patch("arca.cli._doctor_proxy_routing", return_value="localhost:8082 reachable"):
        result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0
    assert "Arca Health Check" in result.output
    assert "PASS" in result.output
    assert "All checks passed." in result.output


def test_doctor_one_fail(monkeypatch):
    with patch("arca.cli._doctor_databricks_auth", return_value="workspace=https://x"), \
         patch("arca.cli._doctor_vs_index", side_effect=RuntimeError("index OFFLINE")), \
         patch("arca.cli._doctor_mlflow", return_value="experiment=/Users/jericohd@gmail.com/arca"), \
         patch("arca.cli._doctor_proxy_routing", return_value="localhost:8082 reachable"):
        result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 1
    assert "FAIL" in result.output
    assert "index OFFLINE" in result.output
    assert "1 check(s) failed. Run 'arca init' if not yet initialized." in result.output


def test_doctor_timeout(monkeypatch):
    with patch("arca.cli._doctor_databricks_auth", side_effect=TimeoutError()), \
         patch("arca.cli._doctor_vs_index", return_value="index ONLINE, 384 dims"), \
         patch("arca.cli._doctor_mlflow", return_value="experiment=/Users/jericohd@gmail.com/arca"), \
         patch("arca.cli._doctor_proxy_routing", return_value="localhost:8082 reachable"):
        result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 1
    assert "timed out after 5s" in result.output


# ---------- CLI-05: arca init ----------

def test_init_sequence(monkeypatch):
    with patch("arca.cli.importlib.util.find_spec", return_value=object()), \
         patch("arca.cli.bootstrap") as bootstrap_fn, \
         patch("arca.cli.SentenceTransformer") as st:
        result = runner.invoke(app, ["init"])
    assert result.exit_code == 0
    bootstrap_fn.assert_called_once()
    st.assert_called_once_with("all-MiniLM-L6-v2", device="cpu")
    assert "Provisioning Delta schema and tables" in result.output
    assert "Creating Vector Search index" in result.output
    assert "Setting up MLflow experiment" in result.output
    assert "Downloading embedding model" in result.output
    assert "Init complete. Run 'arca doctor' to verify all checks pass." in result.output


def test_init_no_torch(monkeypatch):
    with patch("arca.cli.importlib.util.find_spec", return_value=None), \
         patch("arca.cli.bootstrap") as bootstrap_fn:
        result = runner.invoke(app, ["init"])
    assert result.exit_code == 1
    assert "torch is not installed" in result.output
    bootstrap_fn.assert_not_called()


def test_init_idempotent(monkeypatch):
    with patch("arca.cli.importlib.util.find_spec", return_value=object()), \
         patch("arca.cli.bootstrap", return_value=None), \
         patch("arca.cli.SentenceTransformer"):
        result = runner.invoke(app, ["init"])
    assert result.exit_code == 0


# ---------- PKG-01: entry point ----------

def test_cli_entrypoint():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for cmd in ("tail", "start", "stop", "stats", "init", "doctor"):
        assert cmd in result.output, f"{cmd} missing from --help output"


def test_start_help():
    result = runner.invoke(app, ["start", "--help"])
    assert result.exit_code == 0
