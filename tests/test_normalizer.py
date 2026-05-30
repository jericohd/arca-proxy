"""RED test suite for arca/normalizer.py — covers CACHE-06 (canonicalization determinism + exclusions).

These tests lock the public API surface (canonicalize, prompt_hash) and every assertion
encodes a requirement from CACHE-06. They MUST fail with ImportError/ModuleNotFoundError
against the current repo because arca/normalizer.py does not exist yet.

Plan 02-02 will implement the module; once that lands, these tests flip GREEN.
"""
from __future__ import annotations

import json

import pytest

from arca.normalizer import canonicalize, prompt_hash


def _body(**kwargs) -> bytes:
    """Serialize kwargs as a JSON request body (bytes) the way Claude Code sends them."""
    return json.dumps(kwargs).encode("utf-8")


def test_excluded_sampling_params_do_not_change_canonical():
    """temperature / top_p / top_k do not affect semantic identity — they must be excluded."""
    base = dict(
        model="claude-3-5-haiku-20241022",
        messages=[{"role": "user", "content": "what is 2+2?"}],
        max_tokens=1024,
    )
    a = canonicalize(_body(**base, temperature=0.0))
    b = canonicalize(_body(**base, temperature=0.9, top_p=0.5))
    assert a == b


def test_stream_and_metadata_excluded():
    """stream, metadata, service_tier are transport/billing concerns — not semantic."""
    base = dict(
        model="claude-3-5-haiku-20241022",
        messages=[{"role": "user", "content": "what is 2+2?"}],
        max_tokens=1024,
    )
    a = canonicalize(_body(**base))
    b = canonicalize(_body(
        **base,
        stream=True,
        metadata={"user_id": "x"},
        service_tier="priority",
    ))
    assert a == b


def test_content_string_and_block_form_equivalent():
    """'content': 'hi' and 'content': [{'type':'text','text':'hi'}] are semantically identical."""
    a = canonicalize(_body(
        model="claude-3-5-haiku-20241022",
        max_tokens=1,
        messages=[{"role": "user", "content": "hi"}],
    ))
    b = canonicalize(_body(
        model="claude-3-5-haiku-20241022",
        max_tokens=1,
        messages=[{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
    ))
    assert a == b


def test_model_family_normalization():
    """Date-suffix variants and casing normalize to the same family key."""
    a = canonicalize(_body(
        model="claude-3-5-haiku-20241022",
        max_tokens=1,
        messages=[{"role": "user", "content": "hi"}],
    ))
    b = canonicalize(_body(
        model="claude-3-5-haiku-20250101",
        max_tokens=1,
        messages=[{"role": "user", "content": "hi"}],
    ))
    c = canonicalize(_body(
        model="claude-3-5-HAIKU",
        max_tokens=1,
        messages=[{"role": "user", "content": "hi"}],
    ))
    assert a == b
    assert a == c


def test_tools_included_in_canonical():
    """`tools` changes Claude's output distribution — it MUST be part of the canonical form."""
    base = dict(
        model="claude-3-5-haiku-20241022",
        max_tokens=1,
        messages=[{"role": "user", "content": "hi"}],
    )
    without = canonicalize(_body(**base))
    with_tool = canonicalize(_body(**base, tools=[{"name": "calc", "input_schema": {}}]))
    assert without != with_tool


def test_system_prompt_included():
    """`system` changes the model's behavior — it MUST be part of the canonical form."""
    base = dict(
        model="claude-3-5-haiku-20241022",
        max_tokens=1,
        messages=[{"role": "user", "content": "hi"}],
    )
    without = canonicalize(_body(**base))
    with_system = canonicalize(_body(**base, system="You are helpful"))
    assert without != with_system


def test_key_ordering_is_stable():
    """Two payloads with identical content but different top-level key order collapse byte-identical."""
    payload1 = _body(
        messages=[{"role": "user", "content": "hi"}],
        model="claude-3-5-haiku",
        max_tokens=1,
    )
    payload2 = _body(
        max_tokens=1,
        model="claude-3-5-haiku",
        messages=[{"role": "user", "content": "hi"}],
    )
    assert canonicalize(payload1) == canonicalize(payload2)


def test_content_block_order_preserved():
    """Content-block order is semantic — swapping blocks MUST produce different canonical output."""
    a = canonicalize(_body(
        model="claude-3-5-haiku-20241022",
        max_tokens=1,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": "first"},
                {"type": "text", "text": "second"},
            ],
        }],
    ))
    b = canonicalize(_body(
        model="claude-3-5-haiku-20241022",
        max_tokens=1,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": "second"},
                {"type": "text", "text": "first"},
            ],
        }],
    ))
    assert a != b


def test_cache_control_stripped_from_blocks():
    """`cache_control` is a prompt-caching hint — not semantic. It MUST be stripped from blocks."""
    with_cc = canonicalize(_body(
        model="claude-3-5-haiku-20241022",
        max_tokens=1,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": "hi", "cache_control": {"type": "ephemeral"}},
            ],
        }],
    ))
    without_cc = canonicalize(_body(
        model="claude-3-5-haiku-20241022",
        max_tokens=1,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": "hi"},
            ],
        }],
    ))
    assert with_cc == without_cc


def test_prompt_hash_is_hex_64():
    """prompt_hash returns a 64-char lowercase hex string (SHA-256)."""
    c = canonicalize(_body(
        model="x",
        max_tokens=1,
        messages=[{"role": "user", "content": "hi"}],
    ))
    h = prompt_hash(c)
    assert len(h) == 64
    assert all(ch in "0123456789abcdef" for ch in h)


def test_invalid_json_raises():
    """Non-JSON bytes raise ValueError (json.JSONDecodeError subclasses ValueError)."""
    with pytest.raises(ValueError):
        canonicalize(b"not json")
