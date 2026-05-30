"""Tests for scripts/demo_seed.py (DEMO-01). RED until Plan 04 implements."""
from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch, AsyncMock

import pytest


def _fresh_import_demo_seed():
    import importlib, sys
    if "demo_seed" in sys.modules:
        del sys.modules["demo_seed"]
    sys.path.insert(0, "scripts")
    return importlib.import_module("demo_seed")


def test_seed_inserts_10_rows(monkeypatch, capsys):
    fake_index = MagicMock()
    fake_vsc = MagicMock()
    fake_vsc.get_index.return_value = fake_index
    fake_anthropic_response = {"content": [{"type": "text", "text": "stub"}], "model": "claude-sonnet-4", "usage": {"input_tokens": 10, "output_tokens": 20}}

    with patch("arca.embeddings.embed", new=AsyncMock(return_value=MagicMock(tolist=lambda: [0.0]*384))), \
         patch("databricks.vector_search.client.VectorSearchClient", return_value=fake_vsc), \
         patch("arca.cli.importlib.util.find_spec", return_value=object()):
        mod = _fresh_import_demo_seed()
        with patch.object(mod, "_call_anthropic", new=AsyncMock(return_value=fake_anthropic_response)), \
             patch.object(mod, "_insert_delta", return_value=None):
            asyncio.run(mod.seed())

    assert fake_index.upsert.call_count == 10
    fake_index.sync.assert_called_once()


def test_seed_prints_progress(capsys, monkeypatch):
    fake_index = MagicMock()
    fake_vsc = MagicMock()
    fake_vsc.get_index.return_value = fake_index
    with patch("arca.embeddings.embed", new=AsyncMock(return_value=MagicMock(tolist=lambda: [0.0]*384))), \
         patch("databricks.vector_search.client.VectorSearchClient", return_value=fake_vsc):
        mod = _fresh_import_demo_seed()
        with patch.object(mod, "_call_anthropic", new=AsyncMock(return_value={"content": [], "model": "x", "usage": {"input_tokens": 1, "output_tokens": 1}})), \
             patch.object(mod, "_insert_delta", return_value=None):
            asyncio.run(mod.seed())
    out = capsys.readouterr().out
    for i in range(1, 11):
        assert f"[{i}/10] Seeding:" in out
    assert "Seed complete. 10 pairs written to Delta + VS index. Demo ready." in out


def test_seed_no_torch(monkeypatch):
    with patch("importlib.util.find_spec", return_value=None):
        mod = _fresh_import_demo_seed()
        with pytest.raises(SystemExit) as exc:
            mod._check_torch_guard()
        assert exc.value.code == 1
