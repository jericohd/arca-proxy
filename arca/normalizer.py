"""Deterministic canonicalization of Anthropic /v1/messages request bodies.

The output of ``canonicalize(body)`` is a stable UTF-8 JSON string suitable
for hashing (L1 cache key in Phase 3) and embedding (Phase 2). Two requests
that are semantically identical — even if they differ in key order, content
shorthand (string vs. single-text-block array), or in any excluded sampling
parameter — produce byte-identical canonical output.

Excluded (sampling / transport / metadata — not semantic):
  temperature, top_p, top_k, max_tokens, stop_sequences, stream,
  metadata, service_tier, cache_control, container, inference_geo,
  output_config, thinking (non-deterministic output budget)

Included (semantic — identifies what the model is being asked):
  model (normalized to family), system, messages, tools, tool_choice
"""
from __future__ import annotations

import hashlib
import json
import re
from typing import Any

# ---------------------------------------------------------------------------
# Policy constants

# Fields that DO NOT affect the semantic identity of the prompt.
# See Anthropic /v1/messages reference (verified 2026-04-19). CACHE-06.
_EXCLUDED_FIELDS: frozenset[str] = frozenset({
    "temperature",
    "top_p",
    "top_k",
    "max_tokens",
    "stop_sequences",
    "stream",
    "metadata",
    "service_tier",
    "cache_control",      # top-level only; per-block cache_control stripped separately
    "container",
    "inference_geo",
    "output_config",
    "thinking",           # budget_tokens is non-semantic output budget
})

# Strip the trailing -YYYYMMDD date suffix from Anthropic model IDs so that
# claude-3-5-haiku-20241022 and claude-3-5-haiku-20250101 normalize to the
# same family. Deliberate policy: cached responses from an older snapshot of
# the same family are considered equivalent for cache hits.
_MODEL_DATE_SUFFIX = re.compile(r"-\d{8}$")


# ---------------------------------------------------------------------------
# Helpers

def _normalize_model(model: str) -> str:
    """claude-3-5-HAIKU-20241022 -> claude-3-5-haiku"""
    return _MODEL_DATE_SUFFIX.sub("", model.strip().lower())


def _strip_block(block: Any) -> Any:
    """Remove non-semantic keys (cache_control) from a content block."""
    if not isinstance(block, dict):
        return block
    return {k: v for k, v in block.items() if k != "cache_control"}


def _coerce_content(content: Any) -> Any:
    """Normalize messages[].content to its array-of-blocks form.

    Anthropic accepts either a string (shorthand for a single text block) or
    an array of blocks. We ALWAYS normalize to the array form so the two
    request shapes produce identical canonical output. Block order is
    preserved — it's semantic (tool_result after tool_use != before).
    """
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    if isinstance(content, list):
        return [_strip_block(b) for b in content]
    # Unexpected shape — pass through. An un-parseable body produces a unique
    # canonical form and won't hit the cache, which is safe.
    return content


def _coerce_system(system: Any) -> Any:
    """Normalize the system field to its array-of-text-blocks form.

    None / missing / empty-string all map to None (omit the key entirely).
    """
    if system is None or system == "":
        return None
    if isinstance(system, str):
        return [{"type": "text", "text": system}]
    if isinstance(system, list):
        return [_strip_block(b) for b in system]
    return system


# ---------------------------------------------------------------------------
# Public API

def canonicalize(body: bytes) -> str:
    """Return a deterministic UTF-8 canonical JSON string for an Anthropic body.

    Raises:
        ValueError: on invalid JSON (json.JSONDecodeError subclasses ValueError)
            or when the JSON root is not an object.
    """
    parsed = json.loads(body)
    if not isinstance(parsed, dict):
        raise ValueError("Request body is not a JSON object")

    canonical: dict[str, Any] = {}

    # 1. Model family (required by CACHE-06)
    if "model" in parsed:
        canonical["model"] = _normalize_model(str(parsed["model"]))

    # 2. System prompt (required by CACHE-06)
    system = _coerce_system(parsed.get("system"))
    if system is not None:
        canonical["system"] = system

    # 3. Messages (required by CACHE-06)
    messages = parsed.get("messages", [])
    canonical["messages"] = [
        {"role": m.get("role"), "content": _coerce_content(m.get("content"))}
        for m in messages
    ]

    # 4. Tools — semantic (changes what Claude can do)
    if "tools" in parsed:
        canonical["tools"] = parsed["tools"]

    # 5. tool_choice with default elision:
    # if tools present and tool_choice == {"type":"auto"} (the Anthropic default),
    # drop tool_choice so explicit-auto and omitted-auto produce identical canonical.
    tc = parsed.get("tool_choice")
    if tc is not None and not (tc == {"type": "auto"} and "tools" in parsed):
        canonical["tool_choice"] = tc

    # Any field in _EXCLUDED_FIELDS is deliberately dropped above.

    # RFC 8785-style serialization for our float-free canonical subset:
    #   - sort_keys=True:       stable key ordering at every object level
    #   - separators:           no whitespace
    #   - ensure_ascii=False:   preserve UTF-8 verbatim
    return json.dumps(
        canonical,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def prompt_hash(canonical: str) -> str:
    """SHA-256 hex digest (64 lowercase hex chars) of the canonical UTF-8 bytes.

    Phase 3 uses this as the L1 cache key (CACHE-02). Exposed here so both
    normalizer and embeddings modules share one definition.
    """
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
