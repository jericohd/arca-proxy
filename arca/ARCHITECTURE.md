# Arca — Arquitectura Técnica

## Flujo completo

```
┌─────────────────────────────────────────────────────────────┐
│                    Developer Machine                         │
│                                                              │
│   Claude Code ──► Arca Proxy :8082 (FastAPI)                │
│                        │                                     │
│              ┌─────────▼──────────┐                         │
│              │  1. Normalizar      │                         │
│              │     prompt         │                         │
│              │  2. Generar         │                         │
│              │     embedding      │                         │
│              └─────────┬──────────┘                         │
└────────────────────────┼────────────────────────────────────┘
                         │ HTTPS (Databricks SDK)
┌────────────────────────┼────────────────────────────────────┐
│              Databricks Cloud (demo_jedi workspace)          │
│                         │                                    │
│              ┌──────────▼──────────┐                        │
│              │  Vector Search      │                        │
│              │  demo_jedi.arca     │                        │
│              │  .prompt_index      │                        │
│              └──────────┬──────────┘                        │
│                    Hit? │ Miss?                              │
│              ┌──────────┴───────────┐                       │
│              │                      │                       │
│      ┌───────▼──────┐    ┌──────────▼──────────┐           │
│      │  Delta Lake   │    │ → Anthropic API      │           │
│      │  .cache_store │    │   (forward original) │           │
│      │  (leer resp)  │    └──────────┬──────────┘           │
│      └───────┬───────┘               │ guardar resultado    │
│              │               ┌───────▼──────┐               │
│              │               │  Delta Lake   │               │
│              │               │  .cache_store │               │
│              │               │  (escribir)   │               │
│              └───────┬───────┘               │               │
│                      │◄──────────────────────┘               │
│                      │                                       │
│              ┌───────▼──────────┐                           │
│              │     MLflow        │                          │
│              │  experiment: arca  │                         │
│              │  metrics: cost,   │                          │
│              │  latency, hit_rate│                          │
│              └───────┬──────────┘                           │
│                      │                                       │
│              ┌───────▼──────────┐                           │
│              │  Unity Catalog    │                          │
│              │  Lineage + Govern │                          │
│              └──────────────────┘                           │
└─────────────────────────────────────────────────────────────┘
                         │
                  Return response
                  to Claude Code
```

---

## Databricks Setup (reutiliza POC existente)

```sql
-- Reutiliza el catalog ya creado en el POC
USE CATALOG demo_jedi;

-- Nuevo schema para Arca
CREATE SCHEMA IF NOT EXISTS arca
COMMENT 'Arca: Claude Code optimizer — cache + analytics';

-- Tabla principal de cache
CREATE TABLE IF NOT EXISTS arca.cache_store (
  id              STRING DEFAULT uuid(),
  prompt_hash     STRING NOT NULL,    -- SHA256 del prompt normalizado
  prompt_text     STRING NOT NULL,
  embedding       ARRAY<FLOAT>,       -- 384 dims (MiniLM-L6-v2)
  response_json   STRING NOT NULL,    -- JSON completo de la respuesta Anthropic
  model           STRING,
  input_tokens    INT,
  output_tokens   INT,
  cost_usd        DOUBLE,
  hit_count       INT DEFAULT 0,
  created_at      TIMESTAMP DEFAULT current_timestamp(),
  last_hit_at     TIMESTAMP
) USING DELTA
TBLPROPERTIES ('delta.enableChangeDataFeed' = 'true');

-- Tabla de logs de uso (cada llamada, hit o miss)
CREATE TABLE IF NOT EXISTS arca.usage_log (
  id              STRING DEFAULT uuid(),
  session_id      STRING,
  cache_hit       BOOLEAN,
  model           STRING,
  input_tokens    INT,
  output_tokens   INT,
  cost_usd        DOUBLE,
  cost_saved_usd  DOUBLE,
  latency_ms      INT,
  created_at      TIMESTAMP DEFAULT current_timestamp()
) USING DELTA;
```

---

## Proxy Core (FastAPI)

```python
# arca/proxy.py — estructura central
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import httpx
from .cache import CacheEngine
from .tracker import MLflowTracker

app = FastAPI()
cache = CacheEngine()    # Databricks Vector Search + Delta Lake
tracker = MLflowTracker() # MLflow en Databricks

@app.post("/v1/messages")
async def proxy_messages(request: Request):
    body = await request.json()
    prompt = extract_prompt(body)
    
    # 1. Buscar en cache
    hit = await cache.lookup(prompt, threshold=0.95)
    
    if hit:
        tracker.log(cache_hit=True, cost_saved=hit.cost_usd)
        return JSONResponse(hit.response_json)
    
    # 2. Forward a Anthropic
    response = await forward_to_anthropic(body)
    
    # 3. Guardar en cache
    await cache.store(prompt, response)
    tracker.log(cache_hit=False, cost=calculate_cost(response))
    
    return JSONResponse(response)
```

---

## Embeddings

```python
# Modelo: sentence-transformers/all-MiniLM-L6-v2
# - 384 dimensiones
# - ~80ms por embedding en CPU
# - 22MB de tamaño — corre local sin GPU
# - Suficiente para similaridad semántica de prompts de código

from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
embedding = model.encode(prompt_text)  # → array de 384 floats
```

---

## Databricks Vector Search

```python
from databricks.vector_search.client import VectorSearchClient

vs_client = VectorSearchClient()

# Crear index (una vez)
vs_client.create_delta_sync_index(
    endpoint_name="arca-vs-endpoint",
    index_name="demo_jedi.arca.prompt_index",
    source_table_name="demo_jedi.arca.cache_store",
    pipeline_type="TRIGGERED",
    primary_key="id",
    embedding_dimension=384,
    embedding_vector_column="embedding"
)

# Buscar (en cada llamada)
results = vs_client.get_index(
    endpoint_name="arca-vs-endpoint",
    index_name="demo_jedi.arca.prompt_index"
).similarity_search(
    query_vector=embedding,
    columns=["id", "response_json", "cost_usd"],
    num_results=1
)
```

---

## MLflow Tracking

```python
import mlflow

EXPERIMENT = os.environ.get("ARCA_MLFLOW_EXPERIMENT") or f"/Users/{email}/arca"
# Derivado del usuario actual vía Databricks SDK; configurable con ARCA_MLFLOW_EXPERIMENT

mlflow.set_tracking_uri("databricks")
mlflow.set_experiment(EXPERIMENT)

with mlflow.start_run():
    mlflow.log_metric("cache_hit", 1 if hit else 0)
    mlflow.log_metric("cost_usd", cost)
    mlflow.log_metric("cost_saved_usd", saved)
    mlflow.log_metric("latency_ms", latency)
    mlflow.set_tag("model", model_name)
    mlflow.set_tag("session_id", session_id)
```

---

## Dashboard SQL (Databricks SQL)

```sql
-- Vista principal de savings
SELECT
  DATE(created_at)                                    AS fecha,
  COUNT(*)                                            AS total_llamadas,
  SUM(CASE WHEN cache_hit THEN 1 ELSE 0 END)          AS cache_hits,
  ROUND(AVG(CASE WHEN cache_hit THEN 100.0 ELSE 0 END), 1) AS hit_rate_pct,
  ROUND(SUM(cost_usd), 4)                             AS costo_real_usd,
  ROUND(SUM(cost_saved_usd), 4)                       AS ahorro_usd,
  ROUND(AVG(latency_ms))                              AS latencia_avg_ms
FROM demo_jedi.arca.usage_log
GROUP BY 1
ORDER BY 1 DESC;
```

---

## Estructura de archivos del proyecto

```
arca/
├── proxy.py          # FastAPI server principal
├── cache.py          # CacheEngine (Vector Search + Delta Lake)
├── tracker.py        # MLflow integration
├── embeddings.py     # SentenceTransformer wrapper
├── config.py         # Settings (Databricks workspace URL, etc.)
├── cli.py            # `arca start` / `arca stats`
databricks/
├── 00_setup.sql      # Schema + tables setup
├── 01_vector_index.py # Vector Search index creation
├── 02_dashboard.sql  # Databricks SQL dashboard queries
tests/
├── test_cache.py
├── test_proxy.py
pyproject.toml
README.md
```
