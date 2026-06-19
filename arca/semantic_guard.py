"""Polarity guard for the semantic (L2) cache — the precision stage of a
retrieve-then-verify pipeline.

Why this exists (measured, see benchmarks/EVAL_METRICS.md): bi-encoder
embeddings capture *topic* but not *direction/polarity*. "encode base64" and
"decode base64" sit at cosine ~0.93; "convert a list to a tuple" vs the reverse
at ~0.99 — a naive cosine cache would serve the wrong cached answer. An STS
cross-encoder does NOT fix this (it also rates them ~0.99); the difference is not
topical. So we gate accepted candidates with a deterministic check for the
signals that flip meaning.

Design note on overfitting (this is the important part): an earlier version used
a hand-curated antonym list. Measured on a HELD-OUT set it overfit badly (100%
precision on the tuning set → 79% held-out). So the guard is built from GENERAL,
generalizable signals instead of a memorized list:

  1. negation mismatch        — exactly one side negated ("delete" vs "don't delete")
  2. morphological opposite    — same stem, opposite/negating prefix
                                 (zip/unzip, publish/unpublish, encode/decode,
                                  enable/disable, import/export, increase/decrease)
  3. direction swap            — "convert X to Y" vs "convert Y to X"
  4. a SMALL set of common non-morphological opposites (read/write, add/remove,
     up/down, in/out, login/logout …) — the residual hand part, deliberately
     limited and general (not test-specific).

`is_safe_match(a, b)` returns False when any fires → candidate rejected (treated
as a miss) so a wrong answer is never served. Pure stdlib, <1ms.

HONEST LIMITATION: no deterministic guard reaches 100% precision on open-domain
adversarial inputs — there is always an out-of-vocabulary polarity flip
(buy/sell, promote/demote …). The held-out precision in EVAL_METRICS.md is the
real number. The general path to full coverage is an NLI contradiction model
(higher latency) on borderline candidates.
"""
from __future__ import annotations

import re

_WORD = re.compile(r"[a-z0-9]+")
_NEG = re.compile(r"\b(not|no|never|without|cannot|none|non)\b|n't")
_ARTICLES = {"a", "an", "the"}

# Prefixes that, added to a stem, negate/reverse it (zip -> unzip, lock -> unlock).
_NEGATING_PREFIXES = ("un", "de", "dis", "non", "ir", "il")
# Prefix PAIRS that invert meaning on a shared stem (enCODE/deCODE, imPORT/exPORT).
_OPPOSITE_PREFIX_PAIRS = frozenset(
    frozenset(p) for p in [("en", "de"), ("en", "dis"), ("im", "ex"),
                           ("in", "ex"), ("in", "de"), ("a", "de")]
)
_ALL_PREFIXES = ("un", "de", "dis", "non", "ir", "il", "im", "in", "en", "ex", "a", "re")

# Common opposites that are NOT prefix-derived. Deliberately small + general
# (everyday/programming English), not scraped from any eval set.
_OPPOSITES: frozenset[frozenset[str]] = frozenset(
    frozenset(p) for p in [
        ("read", "write"), ("add", "remove"), ("get", "set"), ("push", "pull"),
        ("open", "close"), ("start", "stop"), ("show", "hide"), ("up", "down"),
        ("in", "out"), ("on", "off"), ("left", "right"), ("min", "max"),
        ("first", "last"), ("before", "after"), ("asc", "desc"),
        ("ascending", "descending"), ("true", "false"), ("input", "output"),
        ("source", "target"), ("login", "logout"), ("upload", "download"),
        ("redo", "undo"), ("expand", "collapse"), ("accept", "reject"),
        ("allow", "deny"), ("commit", "rollback"), ("raise", "catch"),
        ("throw", "catch"), ("buy", "sell"), ("send", "receive"),
        ("publish", "consume"), ("master", "replica"),
    ]
)


def _tokens(s: str) -> list[str]:
    return _WORD.findall(s.lower())


def _has_negation(s: str) -> bool:
    return _NEG.search(s.lower()) is not None


def _split_prefix(w: str) -> tuple[str, str]:
    """Return (prefix, stem) if w starts with a known prefix leaving a stem of
    length >= 3, else ("", w)."""
    for p in _ALL_PREFIXES:
        if w.startswith(p) and len(w) - len(p) >= 3:
            return p, w[len(p):]
    return "", w


def _is_morphological_opposite(x: str, y: str) -> bool:
    # one is the other plus a negating prefix: zip/unzip, lock/unlock, publish/unpublish
    for p in _NEGATING_PREFIXES:
        if x == p + y or y == p + x:
            return True
    # same stem, different opposing prefixes: encode/decode, enable/disable, import/export
    px, sx = _split_prefix(x)
    py, sy = _split_prefix(y)
    if sx == sy and px != py and frozenset((px, py)) in _OPPOSITE_PREFIX_PAIRS:
        return True
    return False


def _direction_operands(tokens: list[str]) -> tuple[str, str] | None:
    """For 'convert X to Y' / 'cast X into Y', return (X, Y) using the LAST
    'to'/'into' so a leading 'how to' is ignored. None if no clear operands."""
    idx = None
    for i, t in enumerate(tokens):
        if t in ("to", "into"):
            idx = i
    if idx is None or idx == 0 or idx == len(tokens) - 1:
        return None
    before = next((t for t in reversed(tokens[:idx]) if t not in _ARTICLES), None)
    after = next((t for t in tokens[idx + 1:] if t not in _ARTICLES), None)
    if not before or not after:
        return None
    return (before, after)


def is_safe_match(a: str, b: str) -> bool:
    """True if a cached answer for ``a`` is safe to serve for ``b``.

    False when a polarity/direction/negation conflict is detected — the caller
    must then treat the candidate as a miss. Conservative: when in doubt about a
    flip, reject (a missed cache hit is cheap; a wrong answer is not).
    """
    if _has_negation(a) != _has_negation(b):
        return False

    ta, tb = _tokens(a), _tokens(b)
    only_a = set(ta) - set(tb)
    only_b = set(tb) - set(ta)
    for x in only_a:
        for y in only_b:
            if _is_morphological_opposite(x, y):
                return False
            if frozenset((x, y)) in _OPPOSITES:
                return False

    da, db = _direction_operands(ta), _direction_operands(tb)
    if da and db and da == (db[1], db[0]) and da[0] != da[1]:
        return False

    return True


if __name__ == "__main__":
    # ponytail: smallest runnable self-check. Cases are GENERAL (morphological /
    # common opposites / direction), not copied from any eval file.
    flips = [
        ("how to encode base64", "how to decode base64"),              # opposite prefix
        ("how to zip a folder", "how to unzip a folder"),              # negating prefix
        ("how to enable cors", "how to disable cors"),                 # opposite prefix
        ("convert celsius to fahrenheit", "convert fahrenheit to celsius"),  # direction
        ("how to read a config", "how to write a config"),            # common opposite
        ("delete a file", "do not delete a file"),                    # negation
    ]
    paras = [
        ("what is a python decorator", "explain python decorators"),
        ("how do I reverse a linked list", "reverse a linked list in code"),
    ]
    for a, b in flips:
        assert not is_safe_match(a, b), f"should reject: {a!r} ~ {b!r}"
    for a, b in paras:
        assert is_safe_match(a, b), f"should accept: {a!r} ~ {b!r}"
    print("semantic_guard self-check OK")
