<!-- GSD:project-start source:PROJECT.md -->
## Project

**Arca**

Arca is a local proxy for Claude Code that intercepts outgoing Anthropic API calls and caches them semantically using Databricks as the data backend. Developers point Claude Code at `localhost:8082` with a single config change; Arca embeds prompts, searches a Databricks Vector Search index for similar past requests (threshold 0.95 cosine similarity), and returns cached responses instantly on hits or forwards to Anthropic on misses. All calls are tracked in Delta Lake and MLflow for cost analytics.

**Core Value:** A developer using Claude Code pays $0 and waits <50ms for any question they (or a teammate) have already asked — without changing how they work.

### Constraints

- **Timeline**: Demo-ready by 22 April 2026 (4 days) — interview window opens then
- **Tech stack**: Python 3.11 + FastAPI + sentence-transformers + Databricks SDK (locked)
- **Databricks**: Must reuse existing `demo_jedi` catalog and `demo-secrets` scope
- **Local operation**: Proxy runs on developer machine (not cloud-hosted for v1)
- **No breaking Claude Code**: Proxy must be fully transparent — Claude Code works identically
<!-- GSD:project-end -->

<!-- GSD:stack-start source:research/STACK.md -->
## Technology Stack

## Recommended Stack
| Component | Library | Version | Rationale |
|---|---|---|---|
| Runtime | Python | 3.11.x | Locked by PROJECT.md. 3.11 is the sweet spot — faster than 3.10, full wheel coverage for `sentence-transformers` / `databricks-sdk` / `pyarrow`. Avoid 3.12+ because ML-adjacent wheel availability still trails. |
| Web framework | `fastapi` | 0.115.x | Required for async + streaming. Use `lifespan` context manager (not deprecated `@app.on_event`) for `httpx.AsyncClient` startup. |
| ASGI server | `uvicorn[standard]` | 0.32.x | Ship with `uvloop` + `httptools` extras. Run with `--workers 1` for v1 (single developer, shared in-process cache warm state). |
| ASGI internals | `starlette` | ≥0.40 | Pulled transitively by FastAPI; required for `StreamingResponse` improvements (up to 30% throughput gains). Do not pin independently. |
| HTTP client (to Anthropic) | `httpx` | 0.27.x | Native async, built-in SSE support via `client.stream("POST", ...)`. Initialize **one** `AsyncClient` in `lifespan`, pass via `app.state` — do NOT create per-request. Set `timeout=httpx.Timeout(connect=5.0, read=600.0, write=10.0, pool=5.0)` — Anthropic streams can run minutes. |
| Embeddings | `sentence-transformers` | 3.3.x | Locked. Load `all-MiniLM-L6-v2` once at proxy startup (22 MB, 384 dims, ~60–80 ms/embedding on CPU). Call `model.encode(text, normalize_embeddings=True)` so downstream cosine = dot product. |
| Embeddings backend | `torch` (CPU) | 2.4.x | Install CPU-only wheel: `pip install torch --index-url https://download.pytorch.org/whl/cpu`. Saves ~2 GB install vs CUDA build. |
| Vector search | `databricks-vectorsearch` | ≥0.57 | Official SDK. 0.57+ enables reranker; use `VectorSearchClient().get_index(...).similarity_search(query_vector=..., num_results=1, columns=[...])`. TRIGGERED pipeline for v1 (manual sync on cache write) is fine; switch to CONTINUOUS later. |
| Databricks auth/SDK | `databricks-sdk` | 0.38.x | Single source of truth for workspace credentials. Reads `DATABRICKS_HOST` + `DATABRICKS_TOKEN` from env (`.databrickscfg` as fallback). VectorSearch + MLflow SDKs auto-pick up this config — don't pass tokens manually. |
| Delta writes | `databricks-sql-connector` | 3.7.x | Use `databricks.sql.connect(...)` with `INSERT INTO demo_jedi.arca.cache_store VALUES (...)` via parameterized queries. Pool connections: one long-lived connection in `lifespan`, guarded by `asyncio.Lock` (the connector is sync, so wrap writes in `asyncio.to_thread`). For v1's low QPS this is simpler than Spark Connect or `deltalake`/`pyarrow`. |
| MLflow | `mlflow-skinny[databricks]` | 3.1.x | `mlflow-skinny` avoids pulling scikit-learn, scipy, matplotlib (~400 MB saved). Set `mlflow.set_tracking_uri("databricks")` — picks up SDK auth. Wrap `start_run()` calls in `asyncio.to_thread` since MLflow client is sync. |
| CLI | `typer` | 0.15.x | Prescriptive choice over Click. Type-hint driven, zero decorators for params, ships Click underneath. Entry point: `[project.scripts] arca = "arca.cli:app"` in `pyproject.toml`. |
| Config | `pydantic-settings` | 2.6.x | Env-var driven config (`DATABRICKS_HOST`, `ANTHROPIC_API_KEY`, `ARCA_SIMILARITY_THRESHOLD`). Plays natively with FastAPI. |
| Packaging | `hatchling` | 1.25.x | Build backend. Simpler than Poetry for a CLI tool. `pyproject.toml` only, no `setup.py`. |
| Logging | `structlog` | 24.x | Structured JSON logs; easier to correlate with Delta `usage_log` rows than stdlib `logging`. |
| Testing | `pytest` + `pytest-asyncio` + `httpx` ASGITransport | latest | Use `httpx.AsyncClient(transport=ASGITransport(app=app))` — do NOT start a real uvicorn process in tests. |
## Key Findings
- **FastAPI is the right default, not a compromise.** For a transparent proxy that must pass through SSE streams from Anthropic, FastAPI's `StreamingResponse` + `httpx.AsyncClient.stream()` is the standard pattern. `aiohttp` is faster in raw benchmarks but has weaker SSE ergonomics and a smaller type story; not worth the switch for a 4-day deadline.
- **Reuse a single `httpx.AsyncClient` across requests.** Creating a client per request blows TLS handshake cost (~100–300 ms). Store it on `app.state.http` in the `lifespan` handler.
- **Streaming passthrough pattern:** on cache miss, open `client.stream("POST", upstream_url, ...)`, return a `StreamingResponse(gen(), media_type="text/event-stream")` where `gen()` yields chunks AND accumulates them into a buffer. When the stream ends, assemble the full `message` event and write to Delta + Vector Search *after* the client got their last byte. This keeps perceived latency equal to the upstream.
- **`normalize_embeddings=True` at encode time** means cosine similarity reduces to dot product — aligns with Databricks Vector Search's default cosine metric and avoids double-normalization bugs.
- **`all-MiniLM-L6-v2` is "good enough" for v1, not state-of-the-art in 2026.** Accuracy ~80% vs Nomic Embed ~81% on MTEB; 512-token context is the real limitation. Acceptable for Claude Code prompts (short, code-focused). Flag as a post-demo upgrade path (`BAAI/bge-small-en-v1.5` is the drop-in upgrade: same 384 dims, better MTEB scores, same speed). Keep `all-MiniLM-L6-v2` for the demo — swapping models invalidates the vector index.
- **Databricks auth is centralized.** Set `DATABRICKS_HOST` + `DATABRICKS_TOKEN` once; `databricks-sdk`, `databricks-vectorsearch`, `mlflow`, and `databricks-sql-connector` all read the unified config. Do not pass tokens into each client manually — creates 4 different bug surfaces.
- **MLflow writes are synchronous and latency-sensitive.** Wrap every `mlflow.log_metric` in `asyncio.to_thread` OR (better) buffer in memory and flush with a background task every 5 s. On-hit latency target is <100 ms — one blocking MLflow call alone is ~50–200 ms.
- **Vector Search TRIGGERED sync has a delay.** After inserting into `cache_store`, the index isn't queryable for ~1–5 min until the pipeline runs. For the demo, either (a) call `index.sync()` manually after write, or (b) accept that the first repeat of a prompt won't hit until the next sync. Do NOT architect around the assumption of instant availability.
- **Typer over Click**: identical functionality, half the code. `arca start`, `arca stats`, `arca init` become one-liners with type hints.
- **`mlflow-skinny` over `mlflow`**: no model serving, no autologging, no ML deps — exactly what a proxy needs. Saves ~400 MB.
## What NOT to Use
| Rejected | Reason |
|---|---|
| `aiohttp` instead of FastAPI/httpx | Tech stack is locked. Even if unlocked: weaker SSE ergonomics, no first-class Pydantic integration, marginal perf gain not worth the rewrite risk before a 4-day demo. |
| `requests` for upstream calls | Sync-only; blocks the event loop; will tank concurrency. |
| `flask` / `flask[async]` | Async support is bolted on, not native. Streaming proxy code becomes ugly. FastAPI is the native-async equivalent. |
| `fastembed` (Qdrant ONNX) | Theoretically 5–12x faster on CPU via ONNX Runtime, but real-world reports (GitHub issues #292, #535) show it's often **slower** than sentence-transformers on Apple Silicon and parity on x86. Not worth the operational risk — stay with sentence-transformers, add ONNX export later if profiling demands. |
| `nomic-embed-text-v2` or `BAAI/bge-large` for v1 | +1% accuracy, 3–5x size, 3x slower on CPU. Unity Catalog index is committed to 384 dims — switching models = rebuild index. Ship MiniLM, upgrade to `bge-small-en-v1.5` post-demo if accuracy flags as an issue. |
| `openai`-style client libraries for Anthropic | Arca must be transparent — forward raw bytes of `/v1/messages`. A typed client layer defeats the purpose. Use raw `httpx`. |
| `argparse` / `fire` / `click` | `argparse` = too much boilerplate. `fire` = magical, hard to document. `click` = fine, but Typer does the same with type hints → cleaner. |
| `poetry` as build backend | Works, but heavier than `hatchling` and slower install for end users. For a published CLI tool, `hatchling` + `pyproject.toml` is the 2026 default. |
| `mlflow` (full) | Pulls scikit-learn, scipy, matplotlib, and serving deps (~500 MB). `mlflow-skinny[databricks]` has exactly what's needed. |
| `pyspark` for Delta writes | Spins up a JVM locally, ~30 s cold start, 500 MB install. Overkill for inserting ~1 row per cache event. Use `databricks-sql-connector`. |
| `deltalake` (delta-rs) writing to Unity Catalog tables | UC + managed Delta tables work best via Databricks SQL or Spark. `delta-rs` to UC is possible but adds auth complexity (storage credentials). Not worth it for v1. |
| MLflow `autolog()` | Designed for ML training loops. For a proxy, explicit `log_metric` + `set_tag` is clearer and lower overhead. |
| In-request MLflow calls (sync) | Each `start_run`/`log_metric` hits the network. Batch in a background task or use `asyncio.to_thread` — never block the proxy response path on MLflow. |
| Per-request `httpx.AsyncClient()` | TLS handshake per request = +100–300 ms. Use one client on `app.state`. |
| `gunicorn` with multiple workers | Multiple workers = multiple embedding model copies in RAM (~90 MB each) + cache-state fragmentation across processes. v1 = single uvicorn worker. |
## Version Notes
- `fastapi` 0.115.x (0.115 is the current stable line, post-`lifespan` migration)
- `uvicorn` 0.32.x
- `httpx` 0.27.x
- `sentence-transformers` 3.3.x (3.x ships ONNX/OpenVINO backends if later needed)
- `torch` 2.4.x CPU wheel
- `databricks-vectorsearch` ≥0.57 (0.57 introduced reranker; pin `>=0.57,<1.0`)
- `databricks-sdk` 0.38.x
- `databricks-sql-connector` 3.7.x
- `mlflow-skinny[databricks]` ≥3.1 (MLflow 3.x is the modern line — tracing, GenAI metrics)
- `typer` 0.15.x
- `pydantic-settings` 2.6.x
- `pydantic` 2.9.x (transitive)
- `hatchling` 1.25.x (build backend)
- `structlog` 24.x
- `pytest` 8.x, `pytest-asyncio` 0.24.x
### Example `pyproject.toml` skeleton
### Confidence breakdown
| Area | Confidence | Notes |
|---|---|---|
| FastAPI + httpx + sentence-transformers | HIGH | Locked by PROJECT.md; official docs + Context7-grade patterns |
| Databricks Vector Search SDK 0.57+ | HIGH | Verified via official API docs |
| `databricks-sql-connector` for Delta writes | MEDIUM | Works well at v1 QPS; revisit if >10 writes/sec |
| MLflow 3.1 skinny remote tracking | HIGH | Documented pattern, matches existing POC |
| Typer over Click | MEDIUM | Preference, not technical blocker |
| `all-MiniLM-L6-v2` for v1 | HIGH for demo | MEDIUM longer-term — better models exist but require index rebuild |
## Sources
- [FastAPI async streaming guide (2026)](https://dasroot.net/posts/2026/03/async-streaming-responses-fastapi-comprehensive-guide/)
- [httpx async patterns with FastAPI](https://medium.com/@benshearlaw/how-to-use-httpx-request-client-with-fastapi-16255a9984a4)
- [StreamingResponse + httpx.AsyncClient discussion](https://github.com/fastapi/fastapi/discussions/6173)
- [Best open-source embedding models 2026 (BentoML)](https://www.bentoml.com/blog/a-guide-to-open-source-embedding-models)
- [MiniLM criticism / alternatives (HN)](https://news.ycombinator.com/item?id=46081800)
- [sentence-transformers/all-MiniLM-L6-v2 model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- [Databricks Vector Search Python API](https://api-docs.databricks.com/python/vector-search/databricks.vector_search.html)
- [Query a vector search index (AWS)](https://docs.databricks.com/aws/en/vector-search/query-vector-search)
- [Databricks SQL Connector for Python](https://docs.databricks.com/aws/en/dev-tools/python-sql-connector)
- [Delta Lake best practices](https://docs.databricks.com/aws/en/delta/best-practices)
- [MLflow tracking server configuration](https://docs.databricks.com/aws/en/mlflow/tracking-server-configuration)
- [Connect dev environment to MLflow](https://docs.databricks.com/aws/en/mlflow3/genai/getting-started/connect-environment)
- [Python CLI with Click and Typer (2026)](https://devtoolbox.dedyn.io/blog/python-click-typer-cli-guide)
- [pyproject.toml packaging guide (2026)](https://www.hrekov.com/blog/pyproject-toml-guide)
- [Typer packaging tutorial](https://typer.tiangolo.com/tutorial/package/)
- [Anthropic streaming messages (SSE)](https://platform.claude.com/docs/en/build-with-claude/streaming)
- [Claude Code proxy reference implementation](https://github.com/fuergaosi233/claude-code-proxy)
- [FastEmbed vs SentenceTransformers real-world speed issue](https://github.com/qdrant/fastembed/issues/292)
- [FastEmbed slower than SentenceTransformers on M2](https://github.com/qdrant/fastembed/issues/535)
<!-- GSD:stack-end -->

<!-- GSD:conventions-start source:CONVENTIONS.md -->
## Conventions

Conventions not yet established. Will populate as patterns emerge during development.
<!-- GSD:conventions-end -->

<!-- GSD:architecture-start source:ARCHITECTURE.md -->
## Architecture

Architecture not yet mapped. Follow existing patterns found in the codebase.
<!-- GSD:architecture-end -->

<!-- GSD:workflow-start source:GSD defaults -->
## GSD Workflow Enforcement

Before using Edit, Write, or other file-changing tools, start work through a GSD command so planning artifacts and execution context stay in sync.

Use these entry points:
- `/gsd:quick` for small fixes, doc updates, and ad-hoc tasks
- `/gsd:debug` for investigation and bug fixing
- `/gsd:execute-phase` for planned phase work

Do not make direct repo edits outside a GSD workflow unless the user explicitly asks to bypass it.
<!-- GSD:workflow-end -->



<!-- GSD:profile-start -->
## Developer Profile

> Profile not yet configured. Run `/gsd:profile-user` to generate your developer profile.
> This section is managed by `generate-claude-profile` -- do not edit manually.
<!-- GSD:profile-end -->
