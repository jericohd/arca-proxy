"""Polarity guard for the semantic (L2) cache — the precision stage of a
retrieve-then-verify pipeline.

Why this exists (measured, see benchmarks/EVAL_METRICS.md): bi-encoder
embeddings capture *topic* but not *direction/polarity*. So "encode base64" and
"decode base64" sit at cosine 0.93, and "convert a list to a tuple" vs the
reverse at 0.99 — a naive cosine cache would serve the wrong cached answer.
An STS cross-encoder does NOT fix this (it also rates them ~0.99); the
difference is not topical. So we gate accepted candidates with a cheap,
deterministic check for the three signals that flip meaning:

  1. negation mismatch  — exactly one side negated ("delete" vs "do not delete")
  2. antonym/opposite   — the two sides differ by a known opposite-operation word
  3. direction swap     — "convert X to Y" vs "convert Y to X"

`is_safe_match(a, b)` returns False when any fires → the candidate is rejected
(treated as a miss) so a wrong answer is never served. Pure stdlib, <1ms, keeps
the <50ms hit budget intact.

LIMITATION (honest): the antonym lexicon below is curated, not exhaustive — it
covers common programming opposites. The negation and direction-swap rules are
general; the antonym list is the part with long-tail/overfitting risk. The
generalization path is an NLI model (contradiction detection) at higher latency.
Validate on a held-out set before trusting the lexicon broadly.
"""
from __future__ import annotations

import re

_WORD = re.compile(r"[a-z0-9]+")
_NEG = re.compile(r"\b(not|no|never|without|cannot|none|non)\b|n't")
_ARTICLES = {"a", "an", "the"}

# Opposite-operation / antonym pairs. Curated for the programming domain; each is
# a pair of words that flips the meaning of an otherwise-identical request.
_ANTONYMS: frozenset[frozenset[str]] = frozenset(
    frozenset(p) for p in [
        ("enable", "disable"), ("encode", "decode"), ("encrypt", "decrypt"),
        ("compress", "decompress"), ("serialize", "deserialize"),
        ("install", "uninstall"), ("mount", "unmount"), ("lock", "unlock"),
        ("connect", "disconnect"), ("add", "remove"), ("insert", "delete"),
        ("push", "pop"), ("open", "close"), ("start", "stop"), ("show", "hide"),
        ("increase", "decrease"), ("increment", "decrement"), ("up", "down"),
        ("ascending", "descending"), ("asc", "desc"), ("max", "min"),
        ("maximum", "minimum"), ("read", "write"), ("import", "export"),
        ("expand", "collapse"), ("attach", "detach"), ("allow", "deny"),
        ("accept", "reject"), ("grant", "revoke"), ("raise", "catch"),
        ("throw", "catch"), ("commit", "rollback"), ("forward", "backward"),
        ("synchronous", "asynchronous"), ("sync", "async"),
        ("stdin", "stdout"), ("stdout", "stderr"), ("first", "last"),
        ("before", "after"), ("true", "false"), ("on", "off"),
        ("create", "drop"), ("truncate", "drop"),
    ]
)


def _tokens(s: str) -> list[str]:
    return _WORD.findall(s.lower())


def _has_negation(s: str) -> bool:
    return _NEG.search(s.lower()) is not None


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
            if frozenset((x, y)) in _ANTONYMS:
                return False

    da, db = _direction_operands(ta), _direction_operands(tb)
    if da and db and da == (db[1], db[0]) and da[0] != da[1]:
        return False

    return True


if __name__ == "__main__":
    # ponytail: smallest runnable self-check — the failure modes must be rejected,
    # real paraphrases must pass.
    assert not is_safe_match("how to encode base64 in python", "how to decode base64 in python")
    assert not is_safe_match("convert a list to a tuple in python", "convert a tuple to a list in python")
    assert not is_safe_match("convert a string to int in python", "convert an int to string in python")
    assert not is_safe_match("how to enable ssl in nginx", "how to disable ssl in nginx")
    assert not is_safe_match("delete a file in python", "do not delete a file in python")
    assert is_safe_match("how do I read a file in python", "what's the best way to read a file in python")
    assert is_safe_match("what is a python decorator", "explain python decorators")
    print("semantic_guard self-check OK")
