-- Backup / documentation of DDL in 00_bootstrap.py. Re-running is safe (IF NOT EXISTS).
--
-- Arca Phase 0 — Unity Catalog schema + Delta table definitions
-- Catalog:  demo_jedi  (pre-existing — DO NOT recreate)
-- Schema:   demo_jedi.arca  (created here)
--
-- Run in Databricks SQL editor or as a SQL notebook.
-- All statements are idempotent (IF NOT EXISTS).
--
-- Requirements: DB-01
-- Note: similarity_score DOUBLE in usage_log is REQUIRED by OBS-01.
--       ARCHITECTURE.md omits this column — use THIS file as the source of truth.

-- ============================================================
-- 1. Set active catalog
-- ============================================================
USE CATALOG demo_jedi;

-- ============================================================
-- 2. Create schema
-- ============================================================
CREATE SCHEMA IF NOT EXISTS arca
  COMMENT 'Arca: Claude Code optimizer -- cache + analytics';

-- ============================================================
-- 3. Semantic cache store
--    - embedding ARRAY<FLOAT> (384 dims, all-MiniLM-L6-v2)
--    - Change Data Feed enabled for downstream streaming (Phase 3+)
-- ============================================================
CREATE TABLE IF NOT EXISTS demo_jedi.arca.cache_store (
  id              STRING DEFAULT uuid(),
  prompt_hash     STRING NOT NULL,
  prompt_text     STRING NOT NULL,
  embedding       ARRAY<FLOAT>,          -- 384 dims, all-MiniLM-L6-v2
  response_json   STRING NOT NULL,
  model           STRING,
  input_tokens    INT,
  output_tokens   INT,
  cost_usd        DOUBLE,
  hit_count       INT DEFAULT 0,
  created_at      TIMESTAMP DEFAULT current_timestamp(),
  last_hit_at     TIMESTAMP
) USING DELTA
  TBLPROPERTIES ('delta.enableChangeDataFeed' = 'true');

-- ============================================================
-- 4. Usage / analytics log
--    - similarity_score DOUBLE: required by OBS-01 (missing from ARCHITECTURE.md)
-- ============================================================
CREATE TABLE IF NOT EXISTS demo_jedi.arca.usage_log (
  id               STRING DEFAULT uuid(),
  session_id       STRING,
  cache_hit        BOOLEAN,
  model            STRING,
  input_tokens     INT,
  output_tokens    INT,
  cost_usd         DOUBLE,
  cost_saved_usd   DOUBLE,
  latency_ms       INT,
  similarity_score DOUBLE,               -- OBS-01 dependency — do NOT remove
  created_at       TIMESTAMP DEFAULT current_timestamp()
) USING DELTA;

-- ============================================================
-- 5. Verify
-- ============================================================
SHOW TABLES IN demo_jedi.arca;
