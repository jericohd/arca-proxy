"""Smoke test: proves the pytest harness + asyncio_mode=auto work end-to-end."""
import asyncio

import pytest


def test_imports_arca():
    import arca
    assert arca.__version__ == "0.1.0"


async def test_asyncio_mode_auto_works():
    await asyncio.sleep(0)
    assert True


@pytest.mark.integration
def test_integration_marker_is_registered(databricks_env):
    assert databricks_env["host"].startswith("https://")
    assert databricks_env["token"]
    assert databricks_env["http_path"].startswith("/sql/")
