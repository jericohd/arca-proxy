# Arca — Roadmap

## Fase 0 — Proxy Esqueleto ✅

**Goal:** Claude Code apuntando a Arca sin que nada se rompa.

- [x] `pyproject.toml` + estructura de carpetas
- [x] FastAPI app que recibe `/v1/messages`
- [x] Passthrough transparente a Anthropic API
- [x] Verificación: Claude Code funciona con `ANTHROPIC_BASE_URL=http://localhost:8082`
- [x] Logging básico en terminal (cada llamada)

---

## Fase 1 — Delta Lake + Logging ✅

**Goal:** Cada llamada queda registrada en Databricks.

- [x] SQL setup: `demo_jedi.arca.cache_store` + `usage_log`
- [x] Conexión Databricks SDK desde el proxy
- [x] Escribir cada llamada a `usage_log`
- [x] MLflow experiment con métricas básicas por sesión
- [x] Verificación: Databricks SQL muestra las llamadas en tiempo real

---

## Fase 2 — Semantic Cache ✅

**Goal:** Primera cache hit.

- [x] SentenceTransformer embeddings (local, 384 dims)
- [x] Databricks Vector Search endpoint + index setup
- [x] Cache lookup en cada request (threshold: 0.95)
- [x] Cache write en cada miss (L1 LRU + L2 Delta + VS upsert)
- [x] `cost_saved_usd` calculado y loggeado
- [x] SQLite fallback offline

---

## Fase 3 — Dashboard + CLI ✅

**Goal:** Observable y usable desde terminal.

- [x] Databricks Lakeview dashboard (hit rate, cost saved, latency)
- [x] `arca stats` CLI command (resumen en terminal)
- [x] `arca doctor` — health check de todas las integraciones
- [x] `arca tail` — live stream de cache events
- [x] `scripts/demo_seed.py` — pre-seed de 10 pares
- [x] README con arquitectura y quick start

---

## Fase 4 — Open Source + Lanzamiento (Mayo 2026)

- [ ] GitHub repo público (arca-proxy)
- [ ] Landing page (arca.dev o similar)
- [ ] README con benchmark vs sin cache
- [ ] Post en LinkedIn / Twitter
- [ ] Product Hunt launch
- [ ] Primeros 100 usuarios free tier

---

## Fase 5 — Monetización (Junio 2026)

- [ ] Stripe integration
- [ ] Pro tier ($19/mes) — hosted Databricks backend
- [ ] Team tier ($49/mes) — shared cache
- [ ] Objetivo: 50 Pro usuarios = $950/mes

---

## Métricas de éxito

| Métrica | Target Fase 2 | Target Fase 5 |
|---|---|---|
| Cache hit rate | >30% en uso real | >50% |
| Cost reduction | >20% | >40% |
| Latency (hit) | <100ms | <50ms |
| Pro usuarios | - | 50 |
| MRR | $0 | $950+ |

---

## Build strategy con Claude Code

Cada fase se construye con Claude Code como co-developer:
- Fase 0-1: ~4h de trabajo real
- Fase 2: ~8h de trabajo real (la parte técnica más densa)
- Fase 3: ~4h de trabajo real

**Total estimado con Claude Code: 16h real = ~1 semana a 2-3h/día**
