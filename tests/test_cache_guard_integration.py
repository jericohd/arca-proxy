"""Integration: the polarity guard is wired into the real L2 hit-path.

Proves the guard actually runs inside `arca.cache._l2_lookup` (via the Vector
Search path), not just as a standalone module — a candidate that flips meaning
is rejected even above the cosine threshold, so a wrong cached answer is never
served. The local-SQLite path shares the same guard call.
"""
from __future__ import annotations

import json
from unittest.mock import patch

import numpy as np

from arca.cache import _l2_lookup, _local_l2_lookup


def _canonical(text: str) -> str:
    # mirrors what _write_back stores in the prompt_text column (canonical JSON)
    return json.dumps({"messages": [{"role": "user", "content": [{"type": "text", "text": text}]}]})


def _vs_row(stored_text: str, score: float) -> dict:
    # VS row layout for columns [prompt_hash, response_json, model, prompt_text] + score
    return {"result": {"data_array": [[
        "h1", '{"content":[{"type":"text","text":"cached"}]}', "claude-x",
        _canonical(stored_text), score,
    ]]}}


async def test_l2_rejects_polarity_flip_above_threshold():
    """Query 'decode' vs stored 'encode' at cosine 0.97 -> guard rejects -> no hit."""
    with patch("arca.cache._get_vs_index") as m:
        m.return_value.similarity_search.return_value = _vs_row("how to encode base64 in python", 0.97)
        vec = np.zeros(384, dtype=np.float32)
        hit = await _l2_lookup(vec, None, "how to decode base64 in python")
        assert hit is None, "guard failed to reject an encode/decode flip in the hit-path"


async def test_l2_allows_true_paraphrase_above_threshold():
    """A genuine paraphrase at the same score is still served."""
    with patch("arca.cache._get_vs_index") as m:
        m.return_value.similarity_search.return_value = _vs_row("explain python decorators", 0.97)
        vec = np.zeros(384, dtype=np.float32)
        hit = await _l2_lookup(vec, None, "what is a python decorator")
        assert hit is not None and hit[1] == "h1", "guard wrongly rejected a real paraphrase"


class _FakeFallback:
    def __init__(self, rows):
        self._rows = rows

    async def fetch_all(self):
        return self._rows


async def test_local_path_rejects_polarity_flip():
    """The LOCAL SQLite L2 path (the no-Databricks demo path) also runs the guard.

    Stored vector == query vector (cosine 1.0, well above threshold), but the
    stored prompt is a polarity flip of the query → must NOT be served.
    """
    emb = (np.ones(384, dtype=np.float32) / np.sqrt(384)).tolist()
    rows = [{
        "embedding": emb, "prompt_hash": "h1",
        "response_json": '{"content":[{"type":"text","text":"cached"}]}',
        "model": None, "prompt_text": _canonical("how to encode base64 in python"),
    }]
    qvec = np.asarray(emb, dtype=np.float32)  # identical -> cosine 1.0
    with patch("arca.cache._get_vs_index", return_value=None), \
         patch("arca.cache._get_sqlite_fallback", return_value=_FakeFallback(rows)):
        flip = await _local_l2_lookup(qvec, None, "how to decode base64 in python")
        assert flip is None, "local path served an encode/decode flip"
        para = await _local_l2_lookup(qvec, None, "encode base64 in python please")
        assert para is not None and para[1] == "h1", "local path rejected a real match"
