-- ============================================================
-- DATABRICKS POC: Unity Catalog Setup
-- Notebook: 00_catalog_setup.sql
-- Ejecutar como: SQL notebook en Databricks
-- ============================================================

-- 1. Crear catálogo (o usar 'main' si no tienes permisos de cuenta)
CREATE CATALOG IF NOT EXISTS demo_jedi
COMMENT 'POC: Multi-Agent Data Pipeline — Jedi Hernandez';

USE CATALOG demo_jedi;

-- 2. Crear schema
CREATE SCHEMA IF NOT EXISTS agent_pipeline
COMMENT 'Medallion architecture para el POC de 5 agentes';

USE SCHEMA agent_pipeline;

-- 3. Bronze — datos crudos (se pobla desde 01_sample_data.py)
CREATE TABLE IF NOT EXISTS bronze_raw (
  id          STRING,
  nombre      STRING,
  email       STRING,
  telefono    STRING,
  ciudad      STRING,
  monto       STRING,   -- intencionalmente STRING para mostrar problemas de calidad
  fecha_tx    STRING,   -- formato inconsistente a propósito
  categoria   STRING,
  _source_file STRING
)
USING DELTA
COMMENT 'Capa Bronze: datos crudos sin transformar';

-- 4. Silver — datos normalizados (escrito por Transformation Agent)
CREATE TABLE IF NOT EXISTS silver_normalized (
  id              STRING,
  nombre          STRING,
  email           STRING,
  telefono        STRING,
  ciudad          STRING,
  monto           DOUBLE,
  fecha_tx        DATE,
  categoria       STRING,
  _run_id         STRING,
  _ingestion_ts   TIMESTAMP,
  _quality_score  DOUBLE
)
USING DELTA
COMMENT 'Capa Silver: datos limpios y normalizados';

-- 5. Gold — insights ejecutivos (escrito por Reporting Agent)
CREATE TABLE IF NOT EXISTS gold_insights (
  run_id              STRING,
  report_timestamp    STRING,
  quality_score       DOUBLE,
  records_processed   BIGINT,
  records_normalized  BIGINT,
  executive_report    STRING
)
USING DELTA
COMMENT 'Capa Gold: reportes ejecutivos generados por LLM';

-- 6. Log de estado de agentes (escrito por cada agente)
CREATE TABLE IF NOT EXISTS agent_run_log (
  run_id        STRING,
  agent         STRING,
  status        STRING,
  record_count  BIGINT,
  error_count   BIGINT,
  logged_at     STRING
)
USING DELTA
COMMENT 'Audit trail de cada agente en cada ejecución';

-- 7. Verificar
SHOW TABLES IN demo_jedi.agent_pipeline;

-- 8. (Opcional) Habilitar Change Data Feed en Silver para downstream streaming
ALTER TABLE silver_normalized SET TBLPROPERTIES (delta.enableChangeDataFeed = true);

-- OUTPUT ESPERADO:
-- bronze_raw, silver_normalized, gold_insights, agent_run_log
