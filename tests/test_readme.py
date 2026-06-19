"""DEMO-02 README smoke tests. RED until Plan 04 writes the README."""
from __future__ import annotations

import re
from pathlib import Path

import pytest

README = Path(__file__).parent.parent / "README.md"


@pytest.fixture(scope="module")
def readme_text() -> str:
    return README.read_text(encoding="utf-8")


def test_readme_has_tagline(readme_text):
    expected = "A developer using Claude Code pays $0 and waits milliseconds for questions they — or a teammate — have already asked."
    assert expected in readme_text


def test_readme_has_mermaid_block(readme_text):
    assert "```mermaid" in readme_text
    assert "flowchart LR" in readme_text


def test_readme_has_required_sections(readme_text):
    for heading in ("Quick Start (< 5 minutes)", "How It Works", "Databricks Setup"):
        assert heading in readme_text, f"Missing section heading: {heading}"


def test_readme_has_torch_warning(readme_text):
    assert "torch must be installed separately to avoid the CUDA wheel" in readme_text


def test_readme_mermaid_has_all_nodes(readme_text):
    for label in ("Developer", "Arca Proxy", "L1 LRU", "L2 Vector Search", "Anthropic API", "Delta Lake", "MLflow"):
        assert label in readme_text, f"Mermaid node missing: {label}"


def test_readme_quick_start_step_order(readme_text):
    # torch CPU wheel install MUST appear before `pip install arca-proxy`
    torch_idx = readme_text.find("pip install torch==2.4")
    arca_idx = readme_text.find("pip install arca-proxy")
    assert torch_idx != -1, "torch install command not found"
    assert arca_idx != -1, "pip install arca-proxy not found"
    assert torch_idx < arca_idx, "torch install must precede arca-proxy install"
