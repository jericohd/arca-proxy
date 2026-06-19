"""Arca 90-second demo: miss -> exact hit -> semantic hit -> savings.

    python demo/demo.py

No credentials needed. The Anthropic upstream is simulated in-process with a
realistic 1.2 s response delay (clearly labeled below); everything else --
proxy, normalization, embedding, two-tier cache, cost accounting -- is the
real Arca code path, identical to production. Run `arca start` and point
ANTHROPIC_BASE_URL at it to do the same against the real API.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx
from httpx import ASGITransport, AsyncClient

SIMULATED_UPSTREAM_DELAY_S = 1.2
INPUT_TOKENS, OUTPUT_TOKENS = 1500, 350
MODEL = "claude-sonnet-4-6"

ANSWER = "git rebase -i HEAD~3 opens an interactive editor for the last three commits..."


def _sse(text: str) -> bytes:
    events = [
        ("message_start", {"type": "message_start", "message": {
            "id": "msg_demo", "type": "message", "role": "assistant", "model": MODEL,
            "content": [], "usage": {"input_tokens": INPUT_TOKENS, "output_tokens": 1}}}),
        ("content_block_start", {"type": "content_block_start", "index": 0,
                                 "content_block": {"type": "text", "text": ""}}),
        ("content_block_delta", {"type": "content_block_delta", "index": 0,
                                 "delta": {"type": "text_delta", "text": text}}),
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        ("message_delta", {"type": "message_delta", "delta": {"stop_reason": "end_turn"},
                           "usage": {"output_tokens": OUTPUT_TOKENS}}),
        ("message_stop", {"type": "message_stop"}),
    ]
    return b"".join(f"event: {n}\ndata: {json.dumps(p)}\n\n".encode() for n, p in events)


class SimulatedAnthropic(httpx.AsyncBaseTransport):
    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(SIMULATED_UPSTREAM_DELAY_S)
        return httpx.Response(200, content=_sse(ANSWER),
                              headers={"content-type": "text/event-stream"})


def _body(prompt: str) -> bytes:
    return json.dumps({"model": MODEL, "max_tokens": 1024, "stream": True,
                       "messages": [{"role": "user", "content": prompt}]}).encode()


async def main() -> None:
    os.environ.setdefault("ARCA_HOME", tempfile.mkdtemp(prefix="arca-demo-"))

    print("Loading embedding model (one-time, ~3 s)...")
    import arca.cache  # noqa: F401 — registers the cache hooks on the proxy
    from arca.embeddings import warm_up
    from arca.fallback import SQLiteFallback
    from arca.observability import calculate_cost
    from arca.proxy import app

    await asyncio.to_thread(warm_up)

    app.state.client = httpx.AsyncClient(transport=SimulatedAnthropic())
    store = SQLiteFallback()
    await store.start()
    app.state.sqlite_fallback = store

    per_call = calculate_cost(MODEL, INPUT_TOKENS, OUTPUT_TOKENS)
    spent = saved = 0.0
    print(f"\nUpstream is SIMULATED with a {SIMULATED_UPSTREAM_DELAY_S:.1f} s delay; "
          f"cost math uses real {MODEL} rates\n({INPUT_TOKENS} input / {OUTPUT_TOKENS} "
          f"output tokens per call = ${per_call:.4f}).\n")

    async def ask(label: str, prompt: str) -> None:
        nonlocal spent, saved
        t0 = time.perf_counter()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://arca",
                               timeout=30.0) as client:
            r = await client.post("/v1/messages", content=_body(prompt),
                                  headers={"x-api-key": "demo", "content-type": "application/json"})
        ms = (time.perf_counter() - t0) * 1000
        hit = r.headers.get("x-arca-cache") == "hit"
        if hit:
            saved += per_call
        else:
            spent += per_call
        status = "HIT " if hit else "MISS"
        cost = "$0.0000" if hit else f"${per_call:.4f}"
        print(f"  [{status}] {ms:8.1f} ms  {cost}   {label}")
        print(f"          prompt: {prompt!r}")
        await asyncio.sleep(0.3)  # let the async write-back land before the next call

    print("3 requests through the proxy:\n")
    await ask("first time seen -> forwarded upstream", "What does git rebase -i HEAD~3 do?")
    await ask("exact repeat -> L1 memory cache", "What does git rebase -i HEAD~3 do?")
    await ask("reworded -> semantic match (cosine >= 0.95)", "What does git rebase -i HEAD~3 do, exactly?")

    total = spent + saved
    print(f"\nResult: 2 of 3 calls served from cache.")
    print(f"  API spend without Arca: ${total:.4f}")
    print(f"  API spend with Arca:    ${spent:.4f}  (saved ${saved:.4f}, "
          f"{100 * saved / total:.0f}%)")
    print("\nSame mechanics against the real API: `arca start` then "
          "export ANTHROPIC_BASE_URL=http://localhost:8082")

    await store.stop()
    await app.state.client.aclose()


if __name__ == "__main__":
    asyncio.run(main())
