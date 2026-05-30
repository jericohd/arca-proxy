"""Shared pytest fixtures for Arca Phase 0 tests.

- Integration tests (marked `@pytest.mark.integration`) hit live Databricks and are
  automatically skipped when `DATABRICKS_TOKEN` is not set in the environment.
- `databricks_env` fixture returns a typed dict of the three env vars Phase 0 needs
  (host, token, http_path). If any is missing, the fixture skips the test.
"""
from __future__ import annotations

import os
from typing import TypedDict

import pytest


class DatabricksEnv(TypedDict):
    host: str
    token: str
    http_path: str


REQUIRED_ENV_VARS = ("DATABRICKS_HOST", "DATABRICKS_TOKEN", "DATABRICKS_HTTP_PATH")


def _missing_env_vars() -> list[str]:
    return [v for v in REQUIRED_ENV_VARS if not os.environ.get(v)]


def pytest_collection_modifyitems(config, items):
    """Auto-skip integration tests when Databricks env vars are unset."""
    missing = _missing_env_vars()
    if not missing:
        return
    skip_marker = pytest.mark.skip(
        reason=f"integration tests require env vars: {', '.join(missing)}"
    )
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip_marker)


@pytest.fixture(scope="session")
def databricks_env() -> DatabricksEnv:
    """Return the three Databricks env vars; skip the test if any are unset."""
    missing = _missing_env_vars()
    if missing:
        pytest.skip(f"missing env vars: {', '.join(missing)}")
    return DatabricksEnv(
        host=os.environ["DATABRICKS_HOST"],
        token=os.environ["DATABRICKS_TOKEN"],
        http_path=os.environ["DATABRICKS_HTTP_PATH"],
    )


@pytest.fixture
def arca_home(tmp_path, monkeypatch):
    """Point the SQLite fallback store at a tmp dir so tests don't pollute ~/.arca."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("ARCA_HOME", str(home / ".arca"))
    return home / ".arca"


@pytest.fixture(autouse=True)
def _clear_l1_between_tests():
    try:
        from arca.cache import _l1
        _l1.clear()
    except ModuleNotFoundError:
        pass
    yield
    try:
        from arca.cache import _l1
        _l1.clear()
    except ModuleNotFoundError:
        pass
