# ============================================================
# DATABRICKS POC: Multi-Agent Data Pipeline (5 Agentes)
# Notebook: 02_multi_agent_pipeline.py
# Autor: Irving "Jedi" Hernandez — Demo Databricks SA
#
# Arquitectura:
#   [Orchestrator] → [Ingestion] → [Validation] → [Transformation] → [Reporting]
#   Backend: Delta Lake + MLflow + Unity Catalog + LangGraph + Claude
# ============================================================

# COMMAND ----------
# MAGIC %pip install langgraph langchain-anthropic langchain-core --quiet

# COMMAND ----------
# MAGIC dbutils.library.restartPython()

# COMMAND ----------
# ── IMPORTS ──────────────────────────────────────────────────────────────────
import json
import uuid
import re
import operator
from datetime import datetime
from typing import TypedDict, List, Optional, Literal, Annotated

import mlflow
from pyspark.sql import functions as F
from pyspark.sql.types import StringType, DoubleType, DateType

from langgraph.graph import StateGraph, END
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

# COMMAND ----------
# ── CONFIGURACIÓN ─────────────────────────────────────────────────────────────
CATALOG    = "demo_jedi"
SCHEMA     = "agent_pipeline"
BRONZE     = f"{CATALOG}.{SCHEMA}.bronze_raw"
SILVER     = f"{CATALOG}.{SCHEMA}.silver_normalized"
GOLD       = f"{CATALOG}.{SCHEMA}.gold_insights"
LOG        = f"{CATALOG}.{SCHEMA}.agent_run_log"
import os
_user = os.getenv("DATABRICKS_USER", "your-email@example.com")
EXPERIMENT = f"/Users/{_user}/multi-agent-pipeline-poc"

# API Key — use Databricks Secrets in production; env var for local dev
try:
    ANTHROPIC_KEY = dbutils.secrets.get(scope="demo-secrets", key="anthropic-api-key")
except Exception:
    ANTHROPIC_KEY = os.getenv("ANTHROPIC_API_KEY", "")  # set ANTHROPIC_API_KEY in env

llm = ChatAnthropic(
    model="claude-3-5-haiku-20241022",  # rápido para POC; usar sonnet en prod
    max_tokens=1024,
    api_key=ANTHROPIC_KEY
)

mlflow.set_experiment(EXPERIMENT)

# COMMAND ----------
# ── ESTADO DEL PIPELINE ───────────────────────────────────────────────────────
class PipelineState(TypedDict):
    run_id:            str
    records:           List[dict]
    validation_report: Optional[dict]
    transformed_count: Optional[int]
    insights:          Optional[str]
    errors:            Annotated[List[str], operator.add]  # acumula errores de todos los agentes
    current_agent:     str
    status:            Literal["running", "validated", "transformed", "completed", "failed"]

# COMMAND ----------
# ── HELPER: Audit trail en Delta ──────────────────────────────────────────────
def log_agent_state(state: PipelineState) -> None:
    row = spark.createDataFrame([{
        "run_id":       state["run_id"],
        "agent":        state["current_agent"],
        "status":       state["status"],
        "record_count": len(state.get("records", [])),
        "error_count":  len(state.get("errors", [])),
        "logged_at":    datetime.now().isoformat()
    }])
    row.write.format("delta").mode("append").option("mergeSchema", "true").saveAsTable(LOG)

# COMMAND ----------
# ── AGENT 1: ORCHESTRATOR ─────────────────────────────────────────────────────
def orchestrator_agent(state: PipelineState) -> PipelineState:
    """
    Inicializa el run. Valida configuración. Registra parámetros globales en MLflow.
    En producción: aquí iría lógica LLM para interpretar instrucciones en lenguaje natural.
    """
    print(f"\n{'='*60}")
    print(f"🤖 [Agent 1: Orchestrator] — Run: {state['run_id'][:8]}")
    print(f"{'='*60}")

    mlflow.log_params({
        "run_id":       state["run_id"],
        "source_table": BRONZE,
        "agent_count":  5,
        "llm_model":    "claude-3-5-haiku-20241022",
        "framework":    "LangGraph + Databricks",
        "start_time":   datetime.now().isoformat()
    })

    state["current_agent"] = "orchestrator"
    state["status"]        = "running"
    log_agent_state(state)

    print(f"   ✅ MLflow run inicializado")
    print(f"   ✅ Audit trail activo en {LOG}")
    return state

# COMMAND ----------
# ── AGENT 2: INGESTION ────────────────────────────────────────────────────────
def ingestion_agent(state: PipelineState) -> PipelineState:
    """
    Lee datos crudos de la capa Bronze (Delta Lake).
    Perfila el dataset: conteo, columnas, sample.
    """
    print(f"\n📥 [Agent 2: Ingestion] — Leyendo Bronze layer...")

    try:
        df      = spark.table(BRONZE)
        records = [r.asDict() for r in df.collect()]

        mlflow.log_metrics({
            "records_ingested": len(records),
            "columns_count":    len(df.columns)
        })

        state["records"]       = records
        state["current_agent"] = "ingestion"
        print(f"   ✅ {len(records)} registros leídos | {len(df.columns)} columnas")

    except Exception as e:
        state["errors"]        = [f"IngestionAgent ERROR: {e}"]
        state["status"]        = "failed"
        print(f"   ❌ Error: {e}")

    log_agent_state(state)
    return state

# COMMAND ----------
# ── AGENT 3: VALIDATION (LLM-powered) ────────────────────────────────────────
def validation_agent(state: PipelineState) -> PipelineState:
    """
    Usa Claude para evaluar la calidad del dataset.
    Genera un reporte estructurado con score 0-100.
    Score < 70 → pipeline se detiene (no escribe Silver).
    """
    print(f"\n🔍 [Agent 3: Validation] — Evaluando calidad con LLM...")

    sample_str = json.dumps(state["records"][:5], indent=2, default=str)

    response = llm.invoke([
        SystemMessage(content=(
            "You are a senior data quality engineer. "
            "Analyze the data sample and return ONLY a valid JSON object — "
            "no markdown, no explanation, no code fences."
        )),
        HumanMessage(content=f"""Analyze this data sample and return this exact JSON:

{{
  "quality_score": <integer 0-100>,
  "completeness_pct": <integer 0-100>,
  "duplicate_risk": <"low"|"medium"|"high">,
  "issues": ["<specific issue 1>", "<specific issue 2>"],
  "transformations_needed": ["<action 1>", "<action 2>"],
  "is_valid": <true if quality_score >= 70, else false>
}}

Data sample (first 5 records):
{sample_str}
""")
    ])

    # Parse seguro — el LLM a veces envuelve en markdown
    try:
        report = json.loads(response.content)
    except json.JSONDecodeError:
        match = re.search(r'\{.*\}', response.content, re.DOTALL)
        if match:
            report = json.loads(match.group())
        else:
            report = {
                "quality_score": 60, "completeness_pct": 80,
                "duplicate_risk": "medium", "issues": ["parse error en respuesta LLM"],
                "transformations_needed": ["revisar manualmente"],
                "is_valid": False
            }

    mlflow.log_metrics({
        "quality_score":    report.get("quality_score", 0),
        "completeness_pct": report.get("completeness_pct", 0),
    })
    mlflow.log_text(json.dumps(report, indent=2), "validation_report.json")

    state["validation_report"] = report
    state["current_agent"]     = "validation"
    state["status"]            = "validated"

    score = report.get("quality_score", 0)
    valid = report.get("is_valid", False)
    print(f"   Quality Score:  {score}/100")
    print(f"   Completeness:   {report.get('completeness_pct')}%")
    print(f"   Duplicate Risk: {report.get('duplicate_risk')}")
    print(f"   Pipeline valid: {'✅ SÍ' if valid else '❌ NO — pipeline se detendrá'}")
    if report.get("issues"):
        print(f"   Issues: {', '.join(report['issues'][:3])}")

    log_agent_state(state)
    return state

# COMMAND ----------
# ── AGENT 4: TRANSFORMATION (PySpark nativo) ──────────────────────────────────
def transformation_agent(state: PipelineState) -> PipelineState:
    """
    Normaliza los datos usando PySpark.
    Aplica las transformaciones sugeridas por el Validation Agent.
    Escribe la capa Silver en Delta Lake.
    """
    print(f"\n⚙️  [Agent 4: Transformation] — Normalizando datos → Silver...")

    df = spark.table(BRONZE)

    # 1. Trim en todos los campos string
    string_cols = [f.name for f in df.schema.fields if isinstance(f.dataType, StringType)]
    for col in string_cols:
        df = df.withColumn(col, F.trim(F.col(col)))
        df = df.withColumn(col, F.when(F.col(col) == "", None).otherwise(F.col(col)))

    # 2. Normalizar ciudad → title case
    df = df.withColumn("ciudad", F.initcap(F.col("ciudad")))
    df = df.withColumn("ciudad", F.regexp_replace("ciudad", "^Cdmx$", "Ciudad de México"))

    # 3. Normalizar categoría → lower case
    df = df.withColumn("categoria", F.lower(F.col("categoria")))

    # 4. Monto: limpiar símbolos y convertir a DOUBLE
    df = df.withColumn("monto",
        F.when(F.col("monto") == "None", None)
         .otherwise(F.regexp_replace(F.col("monto"), r'[\$,\s]', "").cast(DoubleType()))
    )

    # 5. Fecha: normalizar múltiples formatos → DATE
    df = df.withColumn("fecha_tx",
        F.coalesce(
            F.to_date("fecha_tx", "yyyy-MM-dd"),
            F.to_date("fecha_tx", "dd/MM/yyyy"),
            F.to_date("fecha_tx", "dd-MM-yyyy"),
            F.to_date("fecha_tx", "yyyy/MM/dd"),
            F.to_date("fecha_tx", "MMM dd yyyy")
        )
    )

    # 6. Eliminar duplicados por id (keep first)
    df = df.dropDuplicates(["id"])

    # 7. Filtrar registros sin nombre (dato mínimo obligatorio)
    df = df.filter(F.col("nombre").isNotNull())

    # 8. Metadata de trazabilidad
    df = (df
        .withColumn("_run_id",        F.lit(state["run_id"]))
        .withColumn("_ingestion_ts",   F.current_timestamp())
        .withColumn("_quality_score",  F.lit(float(state["validation_report"].get("quality_score", 0))))
        .select("id", "nombre", "email", "telefono", "ciudad",
                "monto", "fecha_tx", "categoria",
                "_run_id", "_ingestion_ts", "_quality_score")
    )

    # 9. Escribir Silver (overwrite por run_id — merge en prod)
    df.write \
      .format("delta") \
      .mode("overwrite") \
      .option("mergeSchema", "true") \
      .saveAsTable(SILVER)

    count = df.count()
    mlflow.log_metric("records_written_silver", count)

    state["transformed_count"] = count
    state["current_agent"]     = "transformation"
    state["status"]            = "transformed"

    print(f"   ✅ {count} registros escritos en {SILVER}")
    print(f"   ✅ Normalizaciones: ciudad, categoría, monto, fecha, dedup, nulos")

    log_agent_state(state)
    return state

# COMMAND ----------
# ── AGENT 5: REPORTING (LLM-powered) ─────────────────────────────────────────
def reporting_agent(state: PipelineState) -> PipelineState:
    """
    Genera un reporte ejecutivo usando Claude.
    Escribe el reporte en la capa Gold (Delta Lake).
    """
    print(f"\n📊 [Agent 5: Reporting] — Generando insights ejecutivos con LLM...")

    vr = state["validation_report"]
    original   = len(state.get("records", []))
    normalized = state.get("transformed_count", 0)
    rejected   = original - normalized

    response = llm.invoke([
        SystemMessage(content=(
            "You are a senior data analytics consultant. "
            "Write concisely for a C-level audience. "
            "Respond in the same language the user writes in."
        )),
        HumanMessage(content=f"""Genera un reporte ejecutivo breve para este pipeline de datos.

Métricas del run:
- Registros ingresados:   {original}
- Registros normalizados: {normalized}
- Registros rechazados:   {rejected}
- Score de calidad:       {vr.get('quality_score')}/100
- Completitud:            {vr.get('completeness_pct')}%
- Riesgo de duplicados:   {vr.get('duplicate_risk')}
- Problemas detectados:   {', '.join(vr.get('issues', []))}
- Transformaciones aplicadas: {', '.join(vr.get('transformations_needed', []))}

Escribe 3 párrafos cortos:
1. Estado general del pipeline y calidad de datos
2. Hallazgos principales y anomalías detectadas
3. Recomendaciones accionables para el equipo de datos
""")
    ])

    insights = response.content

    # Escribir capa Gold
    gold_df = spark.createDataFrame([{
        "run_id":             state["run_id"],
        "report_timestamp":   datetime.now().isoformat(),
        "quality_score":      float(vr.get("quality_score", 0)),
        "records_processed":  original,
        "records_normalized": normalized,
        "executive_report":   insights
    }])
    gold_df.write.format("delta").mode("append").saveAsTable(GOLD)

    mlflow.log_text(insights, "executive_report.txt")
    mlflow.log_metrics({
        "records_rejected":  rejected,
        "pipeline_success":  1
    })

    state["insights"]      = insights
    state["current_agent"] = "reporting"
    state["status"]        = "completed"

    log_agent_state(state)
    print(f"   ✅ Reporte escrito en {GOLD}")
    return state

# COMMAND ----------
# ── ROUTING — lógica condicional post-validación ──────────────────────────────
def route_after_validation(state: PipelineState) -> str:
    if state["status"] == "failed":
        return END
    score = state.get("validation_report", {}).get("quality_score", 0)
    if score < 70:
        state["errors"] = [f"Quality score {score}/100 por debajo del umbral (70). Pipeline abortado."]
        state["status"] = "failed"
        mlflow.log_metric("pipeline_success", 0)
        log_agent_state(state)
        print(f"\n🚫 Pipeline detenido — calidad insuficiente ({score}/100)")
        return END
    return "transformation"

# COMMAND ----------
# ── CONSTRUIR EL GRAFO (LangGraph) ────────────────────────────────────────────
def build_pipeline() -> StateGraph:
    wf = StateGraph(PipelineState)

    # Nodos = agentes
    wf.add_node("orchestrator",   orchestrator_agent)
    wf.add_node("ingestion",      ingestion_agent)
    wf.add_node("validation",     validation_agent)
    wf.add_node("transformation", transformation_agent)
    wf.add_node("reporting",      reporting_agent)

    # Flujo
    wf.set_entry_point("orchestrator")
    wf.add_edge("orchestrator",   "ingestion")
    wf.add_edge("ingestion",      "validation")
    wf.add_conditional_edges(
        "validation",
        route_after_validation,
        {"transformation": "transformation", END: END}
    )
    wf.add_edge("transformation", "reporting")
    wf.add_edge("reporting",      END)

    return wf.compile()

# COMMAND ----------
# ── EJECUTAR ──────────────────────────────────────────────────────────────────
run_name = f"multi_agent_pipeline_{datetime.now().strftime('%Y%m%d_%H%M')}"

with mlflow.start_run(run_name=run_name):

    pipeline = build_pipeline()

    initial_state: PipelineState = {
        "run_id":            str(uuid.uuid4()),
        "records":           [],
        "validation_report": None,
        "transformed_count": None,
        "insights":          None,
        "errors":            [],
        "current_agent":     "start",
        "status":            "running"
    }

    final_state = pipeline.invoke(initial_state)

# COMMAND ----------
# ── RESULTADOS FINALES ────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  PIPELINE COMPLETADO — {final_state['status'].upper()}")
print(f"{'='*60}")
print(f"  Run ID:              {final_state['run_id']}")
print(f"  Registros Bronze:    {len(final_state.get('records', []))}")
print(f"  Registros Silver:    {final_state.get('transformed_count', 0)}")
vr = final_state.get('validation_report') or {}
print(f"  Quality Score:       {vr.get('quality_score', 'N/A')}/100")
print(f"  Completitud:         {vr.get('completeness_pct', 'N/A')}%")

if final_state.get("errors"):
    print(f"\n  ❌ Errores:")
    for e in final_state["errors"]:
        print(f"     {e}")

print(f"\n{'─'*60}")
print("  REPORTE EJECUTIVO")
print(f"{'─'*60}")
print(final_state.get("insights", "Sin reporte generado."))
print(f"{'='*60}")

# COMMAND ----------
# ── VALIDACIÓN EN DELTA (opcional — consultar resultados) ─────────────────────

# Silver layer
print("── Silver Layer ─────────────────────────────────────────")
display(spark.table(SILVER).orderBy("id"))

# COMMAND ----------
# Gold layer
print("── Gold Layer (Reportes Ejecutivos) ─────────────────────")
display(spark.table(GOLD).select("run_id", "quality_score", "records_processed",
                                  "records_normalized", "report_timestamp"))

# COMMAND ----------
# Audit trail de agentes
print("── Agent Run Log (Audit Trail) ──────────────────────────")
display(spark.table(LOG).filter(f"run_id = '{final_state['run_id']}'").orderBy("logged_at"))

# COMMAND ----------
# Time travel — cómo explicarlo en el whiteboard
# Ver el estado de Silver ANTES de esta ejecución:
# display(spark.table(SILVER).option("versionAsOf", 0))

# Ver el historial de la tabla:
# display(spark.sql(f"DESCRIBE HISTORY {SILVER}"))
