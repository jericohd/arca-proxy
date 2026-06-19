# Arca — Claude Code Optimizer on Databricks

> *"Arca guarda lo que ya aprendiste para que no lo pagues dos veces."*

**Tipo:** Producto SaaS + Open Source core  
**Fundador:** Jedi Hernandez  
**Fecha de inicio:** 18 Abril 2026

---

## Qué es

Arca es un **proxy local para Claude Code** que intercepta las llamadas a la API de Anthropic y las cachea semánticamente usando Databricks como backend de datos.

En lugar de pagar cada llamada repetida o similar, Arca:
1. Embebe el prompt en un vector
2. Busca en Databricks Vector Search si ya respondiste algo similar (threshold: 0.95)
3. Si hay match → devuelve la respuesta cacheada en <50ms y $0
4. Si no → llama a Anthropic, guarda el resultado en Delta Lake
5. Registra todo en MLflow para analytics de costo

**Una línea de config. Sin cambiar nada más en tu workflow.**

```json
// ~/.claude/settings.json
{ "env": { "ANTHROPIC_BASE_URL": "http://localhost:8082" } }
```

---

## Por qué Databricks (y no Redis/Postgres)

| Componente | Alternativa naive | Por qué Databricks gana |
|---|---|---|
| Cache storage | Redis | Delta Lake = ACID + time travel + escala ilimitada |
| Vector search | pgvector | Databricks VS = serverless, sin infra que mantener |
| Analytics | custom dashboard | Databricks SQL + MLflow = listo en minutos |
| Governance | nada | Unity Catalog = quién usó qué, cuánto costó |
| Multitenancy | custom auth | Unity Catalog schemas por equipo |

Databricks no es solo el backend — **es el diferenciador del producto.**

---

## Modelo de negocio

### Tier Free (Open Source)
- Proxy local + cache contra cualquier backend compatible
- Analytics básicos (terminal)
- Sin límite de uso

### Tier Pro — $19/mes
- Backend Databricks gestionado por Arca (sin setup)
- Dashboard de costos en tiempo real
- Historial 90 días
- Soporte email

### Tier Team — $49/mes (hasta 5 devs)
- Shared knowledge base entre el equipo
- Delta Sharing para compartir cache entre workspaces
- Unity Catalog por equipo (atribución de costos por dev)
- Dashboards por proyecto

### Tier Enterprise — $299/mes
- Deploy en Databricks propio del cliente
- Unity Catalog en su cuenta
- SLA + soporte premium

---

## Stack tecnológico

```
Proxy:          Python 3.11 + FastAPI
Embeddings:     sentence-transformers/all-MiniLM-L6-v2 (local, rápido)
Cache storage:  Delta Lake (Databricks)
Vector search:  Databricks Vector Search
Tracking:       MLflow (Databricks managed)
Governance:     Unity Catalog (catalog: demo_jedi, schema: arca)
Secrets:        Databricks Secrets (scope: demo-secrets)
Packaging:      pip install arca-proxy
```

---

## Estado actual

- [x] Proxy skeleton (FastAPI passthrough transparente)
- [x] Delta Lake schema setup (`demo_jedi.arca`)
- [x] Embeddings + Vector Search index (`all-MiniLM-L6-v2`, 384 dims)
- [x] Cache L1 LRU + L2 Databricks Vector Search (threshold 0.95)
- [x] MLflow tracking por sesión
- [x] CLI completo: `start/stop/init/stats/doctor/tail`
- [x] SQLite fallback offline
- [x] `scripts/demo_seed.py` — pre-seed de 10 pares
- [x] README público con quick start
- [ ] Landing page
- [ ] PyPI publish (`arca-proxy`)

---

## Riesgos

| Riesgo | Probabilidad | Mitigación |
|---|---|---|
| Anthropic shipea esto nativo | Media | Diferenciador es Databricks backend (enterprise) |
| WozCode lleva ventaja | Alta | Nuestro moat = Databricks = enterprise/equipo |
| Databricks cambia Vector Search API | Baja | Abstracción interna |
| Tiempo de build vs entrevista | Media | MVP mínimo en 5 días es suficiente para demo |

---

## Conexión con Master Plan 2026

Arca sirve **tres propósitos simultáneos:**
1. **Demo técnico** para entrevista Databricks SA (22-29 Abril)
2. **Portafolio público** para posiciones en Anthropic, Nimble AI, roles US
3. **Revenue stream** a partir del mes 3 (Pro/Team tier)

No es un proyecto separado del Master Plan — es evidencia viva de las skills que estás vendiendo.
