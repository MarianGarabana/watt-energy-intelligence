# Databricks notebook source
# MAGIC %md
# MAGIC # 🥉 Bronze Layer — Raw Data Ingestion
# MAGIC
# MAGIC **WATT Energy Intelligence Platform**
# MAGIC
# MAGIC This notebook ingests raw data from three sources and writes it to Delta Lake Bronze tables.
# MAGIC No cleaning, no transformations — raw data exactly as received from the APIs.
# MAGIC
# MAGIC | Source       | Table                         | Frequency  |
# MAGIC |--------------|-------------------------------|------------|
# MAGIC | EIA API      | `watt.bronze.eia_demand`      | Hourly     |
# MAGIC | EIA API      | `watt.bronze.eia_generation`  | Hourly     |
# MAGIC | Open-Meteo   | `watt.bronze.weather`         | Hourly     |
# MAGIC | ENTSO-E      | `watt.bronze.entso_load`      | Hourly     |
# MAGIC
# MAGIC **Run this notebook daily via a Databricks Job.**

# COMMAND ----------
# MAGIC %md ## 0. Setup & Configuration

# COMMAND ----------

import sys
import os
import logging
from datetime import datetime, timezone

# ── If running from Databricks Repos, add ingestion/ to path ──────────────────
# The repo root is auto-added to sys.path in Databricks; sub-packages need this
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) \
    if "__file__" in dir() else "/Workspace/Repos/<your-username>/watt-energy-intelligence"
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, StringType, DoubleType, TimestampType
)
from delta.tables import DeltaTable

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ── Spark session (already available in Databricks as `spark`) ─────────────────
# spark = SparkSession.builder.getOrCreate()  # uncomment for local testing

print(f"Spark version : {spark.version}")
print(f"Run timestamp : {datetime.now(timezone.utc).isoformat()}")

# COMMAND ----------
# MAGIC %md ## 1. Configuration — Edit These

# COMMAND ----------

# ── API Keys (use Databricks Secrets in production) ──────────────────────────
# In Databricks: dbutils.secrets.get(scope="watt", key="eia_api_key")
EIA_API_KEY   = dbutils.secrets.get(scope="watt", key="eia_api_key")   # or hardcode for testing
ENTSO_API_KEY = dbutils.secrets.get(scope="watt", key="entso_api_key") # or None to skip ENTSO

# ── Regions to ingest ─────────────────────────────────────────────────────────
EIA_REGIONS   = ["CAL", "TEX", "NY", "NE", "MIDW", "NW"]
ENTSO_COUNTRIES = ["DE", "FR", "ES"]   # set to [] to skip ENTSO on Day 1

# ── How much history to pull each run ─────────────────────────────────────────
DAYS_BACK = 30   # first run: 30 days. subsequent daily runs: 2 days

# ── Delta Lake catalog / schema ───────────────────────────────────────────────
CATALOG = "watt"
SCHEMA  = "bronze"

# COMMAND ----------
# MAGIC %md ## 2. Create Catalog & Schema

# COMMAND ----------

spark.sql(f"CREATE CATALOG IF NOT EXISTS {CATALOG}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")
spark.sql(f"USE {CATALOG}.{SCHEMA}")
print(f"✓ Using catalog: {CATALOG}.{SCHEMA}")

# COMMAND ----------
# MAGIC %md ## 3. Ingest EIA Demand Data

# COMMAND ----------

from ingestion.eia_client import EIAClient

eia_client = EIAClient(api_key=EIA_API_KEY)

demand_df, generation_df = eia_client.get_all_regions(
    regions=EIA_REGIONS,
    days_back=DAYS_BACK,
)

# ── Convert to Spark DataFrame ─────────────────────────────────────────────────
spark_demand = spark.createDataFrame(demand_df)

# ── Add ingestion metadata ─────────────────────────────────────────────────────
spark_demand = spark_demand.withColumn("ingested_at", F.current_timestamp()) \
                            .withColumn("source", F.lit("EIA_API"))

print(f"EIA Demand — rows: {spark_demand.count():,}")
spark_demand.show(5, truncate=False)

# COMMAND ----------
# MAGIC %md ### Write Demand to Bronze Delta Table (merge/upsert to avoid duplicates)

# COMMAND ----------

demand_table = f"{CATALOG}.{SCHEMA}.eia_demand"

# First run: create table. Subsequent runs: merge to avoid duplicates.
if not spark.catalog.tableExists(demand_table):
    spark_demand.write.format("delta").saveAsTable(demand_table)
    print(f"✓ Created table: {demand_table}")
else:
    delta_tbl = DeltaTable.forName(spark, demand_table)
    (
        delta_tbl.alias("existing")
        .merge(
            spark_demand.alias("new"),
            "existing.timestamp = new.timestamp AND existing.region = new.region"
        )
        .whenNotMatchedInsertAll()
        .execute()
    )
    print(f"✓ Merged into: {demand_table}")

spark.sql(f"SELECT COUNT(*) as total_rows FROM {demand_table}").show()

# COMMAND ----------
# MAGIC %md ## 4. Ingest EIA Generation Mix

# COMMAND ----------

spark_gen = spark.createDataFrame(generation_df)
spark_gen = spark_gen.withColumn("ingested_at", F.current_timestamp()) \
                     .withColumn("source", F.lit("EIA_API"))

print(f"EIA Generation — rows: {spark_gen.count():,}")

gen_table = f"{CATALOG}.{SCHEMA}.eia_generation"

if not spark.catalog.tableExists(gen_table):
    spark_gen.write.format("delta").saveAsTable(gen_table)
    print(f"✓ Created table: {gen_table}")
else:
    delta_tbl = DeltaTable.forName(spark, gen_table)
    (
        delta_tbl.alias("existing")
        .merge(
            spark_gen.alias("new"),
            "existing.timestamp = new.timestamp "
            "AND existing.region = new.region "
            "AND existing.fuel_type = new.fuel_type"
        )
        .whenNotMatchedInsertAll()
        .execute()
    )
    print(f"✓ Merged into: {gen_table}")

# Show generation mix summary
print("\n── Generation Mix by Fuel Type (avg MW) ──")
spark.sql(f"""
    SELECT fuel_type, ROUND(AVG(generation_mwh), 0) as avg_mwh
    FROM {gen_table}
    GROUP BY fuel_type
    ORDER BY avg_mwh DESC
""").show()

# COMMAND ----------
# MAGIC %md ## 5. Ingest Weather Data

# COMMAND ----------

from ingestion.weather_client import WeatherClient

weather_client = WeatherClient()
weather_df = weather_client.get_all_regions(
    regions=EIA_REGIONS,
    days_back=DAYS_BACK,
)

spark_weather = spark.createDataFrame(weather_df)
spark_weather = spark_weather.withColumn("ingested_at", F.current_timestamp()) \
                             .withColumn("source", F.lit("OPEN_METEO"))

print(f"Weather — rows: {spark_weather.count():,}")
spark_weather.show(3, truncate=False)

weather_table = f"{CATALOG}.{SCHEMA}.weather"

if not spark.catalog.tableExists(weather_table):
    spark_weather.write.format("delta").saveAsTable(weather_table)
    print(f"✓ Created table: {weather_table}")
else:
    delta_tbl = DeltaTable.forName(spark, weather_table)
    (
        delta_tbl.alias("existing")
        .merge(
            spark_weather.alias("new"),
            "existing.timestamp = new.timestamp AND existing.region = new.region"
        )
        .whenNotMatchedInsertAll()
        .execute()
    )
    print(f"✓ Merged into: {weather_table}")

# COMMAND ----------
# MAGIC %md ## 6. Ingest ENTSO-E European Load (optional)

# COMMAND ----------

if ENTSO_COUNTRIES and ENTSO_API_KEY:
    from ingestion.entso_client import ENTSOClient

    entso_client = ENTSOClient(api_key=ENTSO_API_KEY)
    entso_df = entso_client.get_all_countries(
        countries=ENTSO_COUNTRIES,
        days_back=DAYS_BACK,
    )

    if not entso_df.empty:
        spark_entso = spark.createDataFrame(entso_df)
        spark_entso = spark_entso.withColumn("ingested_at", F.current_timestamp()) \
                                 .withColumn("source", F.lit("ENTSO_E"))

        entso_table = f"{CATALOG}.{SCHEMA}.entso_load"
        if not spark.catalog.tableExists(entso_table):
            spark_entso.write.format("delta").saveAsTable(entso_table)
            print(f"✓ Created table: {entso_table}")
        else:
            delta_tbl = DeltaTable.forName(spark, entso_table)
            (
                delta_tbl.alias("existing")
                .merge(
                    spark_entso.alias("new"),
                    "existing.timestamp = new.timestamp AND existing.country = new.country"
                )
                .whenNotMatchedInsertAll()
                .execute()
            )
            print(f"✓ Merged into: {entso_table}")
else:
    print("⏭ Skipping ENTSO-E (no API key or no countries configured)")

# COMMAND ----------
# MAGIC %md ## 7. Bronze Layer Summary

# COMMAND ----------

print("\n" + "=" * 60)
print("🥉 BRONZE LAYER SUMMARY")
print("=" * 60)

tables = [
    f"{CATALOG}.{SCHEMA}.eia_demand",
    f"{CATALOG}.{SCHEMA}.eia_generation",
    f"{CATALOG}.{SCHEMA}.weather",
]
if ENTSO_COUNTRIES:
    tables.append(f"{CATALOG}.{SCHEMA}.entso_load")

for table in tables:
    if spark.catalog.tableExists(table):
        count = spark.sql(f"SELECT COUNT(*) as n FROM {table}").collect()[0]["n"]
        min_ts = spark.sql(f"SELECT MIN(timestamp) as t FROM {table}").collect()[0]["t"]
        max_ts = spark.sql(f"SELECT MAX(timestamp) as t FROM {table}").collect()[0]["t"]
        print(f"\n  📋 {table}")
        print(f"     Rows      : {count:,}")
        print(f"     Date range: {min_ts} → {max_ts}")

print("\n✅ Bronze ingestion complete.")
print(f"   Next step: Run notebook 02_silver_cleaning.py")
