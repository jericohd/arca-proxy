# ============================================================
# DATABRICKS POC: Bronze Layer — Sample Data Loader
# Notebook: 01_sample_data.py
# Copiar celda por celda en un Python notebook de Databricks
# ============================================================

# COMMAND ----------
# Datos de muestra con problemas de calidad INTENCIONALES
# para que el Validation Agent y Transformation Agent
# tengan algo real que demostrar

from pyspark.sql import Row
from pyspark.sql.functions import lit
from datetime import date

# Registros con problemas: fechas inconsistentes, montos como string,
# nulos, emails malformados, duplicados — típico de datos reales
raw_data = [
    Row(id="001", nombre="  Carlos Mendoza ", email="carlos.mendoza@gmail.com",
        telefono="55-1234-5678", ciudad="Ciudad de México", monto="15000.50",
        fecha_tx="2026-03-15", categoria="electronica", _source_file="upload_001.csv"),

    Row(id="002", nombre="María López", email="maria.lopez@hotmail.com",
        telefono="(81)8765-4321", ciudad="monterrey",  monto="$2,300.00",
        fecha_tx="15/03/2026", categoria="ROPA", _source_file="upload_001.csv"),

    Row(id="003", nombre="Roberto García ", email="roberto_garcia@empresa",  # email malformado
        telefono="33-9876-5432", ciudad="Guadalajara", monto="8750",
        fecha_tx="2026-03-16", categoria="Electronica", _source_file="upload_002.csv"),

    Row(id="004", nombre="", email="ana.martinez@corp.mx",  # nombre vacío
        telefono="55-2345-6789", ciudad="CDMX", monto="45000.00",
        fecha_tx="Mar 16 2026", categoria="servicios", _source_file="upload_002.csv"),

    Row(id="005", nombre="Luis Hernández", email="luis.hdz@yahoo.com",
        telefono="442-345-6789", ciudad="Querétaro", monto="None",  # nulo como string
        fecha_tx="2026-03-17", categoria="electronica", _source_file="upload_003.csv"),

    Row(id="006", nombre="Patricia Sánchez", email="patricia.sanchez@gmail.com",
        telefono="55-3456-7890", ciudad="Ciudad de México", monto="12500.75",
        fecha_tx="2026-03-17", categoria="ropa", _source_file="upload_003.csv"),

    Row(id="007", nombre="  Jorge Ramírez", email="jorge.ramirez@outlook.com",
        telefono="81-4567-8901", ciudad="Monterrey", monto="3200.00",
        fecha_tx="17-03-2026", categoria="servicios", _source_file="upload_004.csv"),

    Row(id="008", nombre="Sofía Torres", email="sofia.torres@empresa.com.mx",
        telefono="33-5678-9012", ciudad="guadalajara", monto="9800.50",
        fecha_tx="2026-03-18", categoria="ELECTRONICA", _source_file="upload_004.csv"),

    Row(id="003", nombre="Roberto García", email="roberto_garcia@empresa",  # DUPLICADO de id 003
        telefono="33-9876-5432", ciudad="Guadalajara", monto="8750",
        fecha_tx="2026-03-16", categoria="electronica", _source_file="upload_005.csv"),

    Row(id="009", nombre="Carmen Flores", email="carmen.flores@hotmail.com",
        telefono="442-678-9012", ciudad="Querétaro", monto="21000.00",
        fecha_tx="2026/03/18", categoria="Servicios", _source_file="upload_005.csv"),
]

# COMMAND ----------
df = spark.createDataFrame(raw_data)

# Escribir a Bronze (append para simular múltiples cargas)
df.write \
  .format("delta") \
  .mode("overwrite") \
  .option("mergeSchema", "true") \
  .saveAsTable("demo_jedi.agent_pipeline.bronze_raw")

print(f"✅ Bronze layer poblado con {df.count()} registros")
print(f"   Incluye: duplicados, nulos, fechas inconsistentes, montos como string")
print(f"   Propósito: demostrar capacidad de Validation + Transformation agents")

# COMMAND ----------
# Verificar datos cargados
display(spark.table("demo_jedi.agent_pipeline.bronze_raw"))
