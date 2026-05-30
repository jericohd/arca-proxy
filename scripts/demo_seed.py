"""scripts/demo_seed.py — pre-seed 10 prompt pairs into Arca cache for demo (DEMO-01).

Run: `python scripts/demo_seed.py` (requires DATABRICKS_* + ANTHROPIC_API_KEY env vars).
Writes 10 canonical prompts to `demo_jedi.arca.cache_store` (Delta) AND upserts them
to the Direct Access VS index `demo_jedi.arca.prompt_index`, then syncs the index
so the demo's paraphrase lookup returns a hit at threshold 0.95.

Produces the guaranteed demo hits:
  - L1 exact: replay SEED_PROMPTS[0] verbatim during demo -> SHA256 match, <5ms
  - L2 paraphrase: "Write a Python function to flatten nested lists" vs seeded
    "Write a Python function to flatten a nested list" -> >=0.95 cosine hit.
"""
from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import sys
import time
import uuid
from typing import Any

SEED_PROMPTS: list[str] = [
    "What does `git rebase -i HEAD~3` do?",
    "Write a Python function to flatten a nested list",
    "Explain the difference between `async def` and `def` in Python",
    "What is a Python context manager and how do I write one?",
    "How do I configure a Databricks Vector Search Direct Access index?",
    "Write a SQL query to find the top 5 most expensive cache hits",
    "What is the difference between `is` and `==` in Python?",
    "How does the Anthropic SSE streaming protocol work?",
    "What is cosine similarity and why is it used for semantic search?",
    "Explain circuit breaker pattern in distributed systems",
]

CACHE_TABLE = "demo_jedi.arca.cache_store"
ENDPOINT = "arca-vs-endpoint"
INDEX = "demo_jedi.arca.prompt_index"


def _check_torch_guard() -> None:
    """Exit 1 with UI-SPEC message if torch is not installed."""
    if importlib.util.find_spec("torch") is None:
        print("torch not installed. Run the CPU wheel install first.")
        raise SystemExit(1)


async def _call_anthropic(prompt: str) -> dict[str, Any]:
    """Call real Anthropic API; return parsed JSON (non-streaming /v1/messages).

    Falls back to a deterministic stub response when ANTHROPIC_API_KEY is missing
    so the seed script is testable offline. Live demo MUST have the key set.
    """
    import httpx
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        return {
            "id": f"stub_{uuid.uuid4().hex[:12]}",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-4",
            "content": [{"type": "text", "text": f"(stub response for: {prompt[:40]})"}],
            "usage": {"input_tokens": 10, "output_tokens": 20},
        }
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 512,
                "messages": [{"role": "user", "content": prompt}],
            },
        )
        resp.raise_for_status()
        return resp.json()


def _insert_delta(row_id: str, prompt_hash_val: str, canonical: str,
                  embedding: list[float], response_json: dict, model: str) -> None:
    """Insert one row into demo_jedi.arca.cache_store via databricks-sql-connector.

    Uses parameterized INSERT. Caller decides when to flush (we commit per row here
    because N=10 is tiny).
    """
    try:
        from databricks.sql import connect as databricks_sql_connect
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("databricks-sql-connector not installed") from exc
    host = os.environ.get("DATABRICKS_HOST", "").replace("https://", "").replace("http://", "")
    token = os.environ.get("DATABRICKS_TOKEN", "")
    http_path = os.environ.get("DATABRICKS_HTTP_PATH", "")
    if not (host and token and http_path):
        # Tests patch this function; production must have env vars.
        raise RuntimeError("DATABRICKS_HOST / DATABRICKS_TOKEN / DATABRICKS_HTTP_PATH required")
    with databricks_sql_connect(server_hostname=host, http_path=http_path,
                                access_token=token) as conn:
        cur = conn.cursor()
        cur.execute(
            f"""
            INSERT INTO {CACHE_TABLE}
                (id, prompt_hash, prompt_text, embedding, response_json, model, created_at)
            VALUES (?, ?, ?, ?, ?, ?, current_timestamp())
            """,
            (row_id, prompt_hash_val, canonical, embedding, json.dumps(response_json), model),
        )


async def seed() -> None:
    """Main seeding routine: 10 prompts -> Delta + VS index + sync."""
    print("Seeding Arca cache with 10 prompt pairs for demo...")

    from arca.embeddings import embed
    from arca.normalizer import canonicalize, prompt_hash
    from databricks.vector_search.client import VectorSearchClient

    vsc = VectorSearchClient()
    index = vsc.get_index(endpoint_name=ENDPOINT, index_name=INDEX)

    for i, prompt in enumerate(SEED_PROMPTS, start=1):
        short = prompt[:50]
        print(f'  [{i}/10] Seeding: "{short}"')

        canonical_obj = {
            "model": "claude-sonnet-4",
            "messages": [{"role": "user", "content": prompt}],
        }
        canonical = canonicalize(json.dumps(canonical_obj))
        phash = prompt_hash(canonical)
        vec = await embed(canonical)
        row_id = str(uuid.uuid4())

        response_json = await _call_anthropic(prompt)

        # 1. Delta row (synchronous; wrap in to_thread to avoid blocking loop)
        await asyncio.to_thread(
            _insert_delta, row_id, phash, canonical,
            vec.tolist(), response_json, "claude-sonnet-4",
        )

        # 2. VS index upsert
        await asyncio.to_thread(
            index.upsert,
            [
                {
                    "id": row_id,
                    "prompt_hash": phash,
                    "prompt_text": canonical,
                    "embedding": vec.tolist(),
                    "response_json": json.dumps(response_json),
                    "model": "claude-sonnet-4",
                }
            ],
        )

    print("Waiting for Vector Search index to confirm upserts (5s)...")
    time.sleep(5)
    index.sync()
    print("Seed complete. 10 pairs written to Delta + VS index. Demo ready.")


def main() -> int:
    _check_torch_guard()
    try:
        asyncio.run(seed())
        return 0
    except Exception as exc:  # pragma: no cover
        print(f"[FAIL] demo seed failed: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
