# Arca — Diagramas de Arquitectura

Para usar en la entrevista Databricks y para entender el funcionamiento del sistema.

---

## Diagrama 1 — Flujo de llamada (Sequence)

Muestra qué pasa exactamente cuando Claude Code hace una llamada a Anthropic.

```mermaid
sequenceDiagram
    participant CC as Claude Code
    participant AP as Arca Proxy :8082
    participant EM as Embeddings (local)
    participant VS as Databricks Vector Search
    participant DL as Delta Lake cache_store
    participant AN as Anthropic API
    participant ML as MLflow

    CC->>AP: POST /v1/messages {prompt}
    AP->>EM: encode(prompt) → 384-dim vector
    EM-->>AP: embedding
    AP->>VS: similarity_search(embedding, threshold=0.95)

    alt Cache HIT (cosine similarity >= 0.95)
        VS-->>AP: cached response + cost_usd
        AP->>DL: increment hit_count
        AP->>ML: log(cache_hit=true, cost_saved)
        AP-->>CC: response en ~43ms
    else Cache MISS
        VS-->>AP: sin match
        AP->>AN: forward original request
        AN-->>AP: response + token usage
        AP->>DL: INSERT (prompt, embedding, response, cost)
        AP->>VS: trigger index sync
        AP->>ML: log(cache_hit=false, cost_usd)
        AP-->>CC: response en ~1.2s
    end
```

---

## Diagrama 2 — Componentes Databricks

Cómo los 4 productos de Databricks trabajan juntos dentro de Arca.

```mermaid
flowchart TB
    subgraph DEV["Developer Machine"]
        CC["Claude Code"]
        PX["Arca Proxy\nFastAPI :8082"]
        CC -->|"todas las llamadas\nvia ANTHROPIC_BASE_URL"| PX
    end

    subgraph DBX["Databricks Workspace demo_jedi"]
        subgraph UC["Unity Catalog"]
            CAT["demo_jedi.arca (schema)\nGovernance + Lineage"]
        end

        subgraph DL["Delta Lake"]
            CS["cache_store\nACID + time travel + CDF"]
            UL["usage_log\nhistorial completo"]
        end

        subgraph VS["Mosaic AI Vector Search"]
            IDX["prompt_index\nDelta Sync automatico"]
        end

        subgraph MF["MLflow"]
            EX["Experiment: arca-proxy\nmetrics: cost, latency, hit_rate"]
        end

        subgraph SQL["Databricks SQL"]
            DASH["Dashboard ROI\nen tiempo real"]
        end

        CAT --> CS & UL
        CS <-->|"Delta Sync"| IDX
        UL & CS --> DASH
    end

    PX <-->|"similarity search"| VS
    PX -->|"write cache"| DL
    PX -->|"log metrics"| MF

    style IDX fill:#FF6B35,color:#fff
    style CS fill:#cd7f32,color:#fff
    style MF fill:#f59e0b,color:#000
    style CAT fill:#6366f1,color:#fff
    style DASH fill:#22c55e,color:#fff
```

---

## Diagrama 3 — Valor de negocio

El ROI que justifica Arca para una empresa.

```mermaid
flowchart LR
    subgraph SIN["Sin Arca"]
        C1["Llamada 1: $0.0034"]
        C2["Llamada similar: $0.0034"]
        C3["Llamada similar: $0.0034"]
        C1 & C2 & C3 --> T1["Total: $0.0102"]
    end

    subgraph CON["Con Arca (~45% hit rate)"]
        D1["Llamada 1 MISS: $0.0034"]
        D2["Llamada similar HIT: $0.00"]
        D3["Llamada similar HIT: $0.00"]
        D1 & D2 & D3 --> T2["Total: $0.0034\nAhorro: 67%"]
    end

    subgraph SCALE["A escala: 50 devs usando Claude Code"]
        M1["Sin Arca: ~$6,000/mes"]
        M2["Con Arca: ~$3,600-4,200/mes"]
        M1 -->|"Ahorro: $1,800-2,400/mes"| M2
        M2 --> ROI["ROI positivo desde el mes 1"]
    end

    style T2 fill:#22c55e,color:#fff
    style M2 fill:#22c55e,color:#fff
    style ROI fill:#22c55e,color:#fff
    style T1 fill:#ef4444,color:#fff
    style M1 fill:#ef4444,color:#fff
```

---

## Diagrama 4 — Por qué Databricks y no Redis

El argumento enterprise para usar Databricks como backend de cache.

```mermaid
flowchart TB
    subgraph REDIS["Opcion: Redis"]
        R1["✅ Velocidad (sub-ms)"]
        R2["❌ Sin governance"]
        R3["❌ Sin lineage de datos"]
        R4["❌ Sin analytics SQL"]
        R5["❌ Sin multitenancy enterprise"]
        R6["❌ No escala a TB de historico"]
        R7["❌ Infraestructura separada"]
    end

    subgraph DBX2["Opcion: Databricks (Arca)"]
        D1["✅ Velocidad suficiente (43ms hit)"]
        D2["✅ Unity Catalog: governance completo"]
        D3["✅ Delta Lake: lineage + time travel"]
        D4["✅ DBSQL: dashboard SQL nativo"]
        D5["✅ Multitenancy por schema/catalog"]
        D6["✅ Escala horizontal automatico"]
        D7["✅ La empresa ya tiene licencia"]
    end

    subgraph TARGET["Mercado objetivo"]
        T1["Empresa con 20+ devs\nusando Claude Code"]
        T2["Ya tienen licencia Databricks"]
        T3["Compliance: datos no salen\nde su infraestructura"]
    end

    DBX2 --> TARGET

    style D1 fill:#22c55e,color:#fff
    style D2 fill:#22c55e,color:#fff
    style D3 fill:#22c55e,color:#fff
    style D4 fill:#22c55e,color:#fff
    style D5 fill:#22c55e,color:#fff
    style D6 fill:#22c55e,color:#fff
    style D7 fill:#22c55e,color:#fff
    style R2 fill:#ef4444,color:#fff
    style R3 fill:#ef4444,color:#fff
    style R4 fill:#ef4444,color:#fff
    style R5 fill:#ef4444,color:#fff
    style R6 fill:#ef4444,color:#fff
    style R7 fill:#ef4444,color:#fff
```

---

## Cómo presentarlo en la entrevista

**Frase de apertura:**
> "Después de nuestra primera llamada construi algo para demostrar que entiendo Databricks desde adentro. Es un proxy para Claude Code que usa Vector Search para cache semántico y Delta Lake como storage. Corre en el mismo workspace que el POC de 5 agentes."

**Secuencia recomendada:**
1. Muestra Diagrama 2 (componentes Databricks) — ancla el producto en la plataforma
2. Muestra Diagrama 1 (flujo de llamada) — explica la mecánica
3. Muestra Diagrama 3 (ROI) — cierra con el business case
4. Si preguntan por Redis → Diagrama 4
