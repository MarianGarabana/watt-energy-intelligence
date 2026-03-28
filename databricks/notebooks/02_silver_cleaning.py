# Databricks notebook source

# MAGIC %md
# MAGIC # 🥈 Silver Layer — Cleaning & Validation
# MAGIC
# MAGIC **WATT Energy Intelligence Platform**
# MAGIC
# MAGIC In this notebook we read the raw Bronze Delta tables, clean and validate
# MAGIC the data, and write structured Silver Delta tables ready for feature engineering.
# MAGIC
# MAGIC The Bronze → Silver step is about **trust**: fixing types, removing duplicates,
# MAGIC handling nulls, and flagging bad records — so everything downstream can rely on clean data.
# MAGIC
# MAGIC | Bronze Table               | Silver Table               | Key Cleaning Steps                        |
# MAGIC |----------------------------|----------------------------|-------------------------------------------|
# MAGIC | `watt.bronze.eia_demand`   | `watt.silver.eia_demand`   | types, dedup, null filter, range check    |
# MAGIC | `watt.bronze.eia_generation` | `watt.silver.eia_generation` | types, dedup, null filter, negatives  |
# MAGIC | `watt.bronze.weather`      | `watt.silver.weather`      | types, dedup, null fill, range check      |

# COMMAND ----------

# MAGIC %md
# MAGIC ## Section 1. Setup

# COMMAND ----------

# Same imports you used throughout the MDA course
import pyspark.sql.functions as F
from pyspark.sql.functions import col, when, to_timestamp, hour, dayofweek, month
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, TimestampType

# Catalog and schema — same pattern as the OLIST group project
CATALOG = "watt"
BRONZE   = f"{CATALOG}.bronze"
SILVER   = f"{CATALOG}.silver"

# Create the Silver schema if it doesn't exist yet
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {SILVER}")
spark.sql(f"USE {CATALOG}.{SILVER.split('.')[1]}")

print(f"Reading from  : {BRONZE}")
print(f"Writing to    : {SILVER}")
print(f"Spark version : {spark.version}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Section 2. Helper — Data Quality Report
# MAGIC
# MAGIC This is the same null-counting pattern from the MDA lectures.
# MAGIC We run it on every table before and after cleaning so we can see the improvement.

# COMMAND ----------

def quality_report(df, table_name):
    """
    Prints a data quality summary for a DataFrame.
    Shows row count, null counts per column, and duplicate count.
    Same pattern as the null detection used in the MDA exercises.
    """
    print(f"\n── Quality Report: {table_name} ──")
    print(f"   Total rows : {df.count():,}")

    # Count nulls per column — exact pattern from MDA notebooks
    null_counts = df.select([
        F.count(F.when(F.col(c).isNull(), c)).alias(c)
        for c in df.columns
    ])
    print("   Null counts per column:")
    null_counts.show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Section 3. Clean EIA Demand Data
# MAGIC
# MAGIC **What we are fixing:**
# MAGIC - `timestamp` arrives as a string — we cast it to TimestampType
# MAGIC - `demand_mwh` arrives as string or int — we cast to DoubleType
# MAGIC - Remove duplicate (timestamp, region) pairs using `dropDuplicates()`
# MAGIC - Filter out rows where demand is null or physically impossible (≤ 0 MWh)

# COMMAND ----------

# Read Bronze table — same read pattern as the OLIST notebooks
demand_raw = spark.read.table(f"{BRONZE}.eia_demand")

print("Bronze schema (before cleaning):")
demand_raw.printSchema()

quality_report(demand_raw, "bronze.eia_demand")

# COMMAND ----------

# Clean the demand table
# Each .withColumn() step is one transformation — same chaining style as in MDA
demand_clean = (
    demand_raw

    # Cast timestamp string → proper TimestampType
    # Same as: .withColumn("CRASH_TIMESTAMP", to_timestamp("CRASH_DATE",...)) from the flights notebook
    .withColumn("timestamp", to_timestamp(col("timestamp")))

    # Cast demand to DoubleType so Spark ML and arithmetic work correctly
    .withColumn("demand_mwh", col("demand_mwh").cast(DoubleType()))

    # Remove rows where demand arrived as null — same as orders_clean in OLIST
    .filter(col("demand_mwh").isNotNull())
    .filter(col("timestamp").isNotNull())
    .filter(col("region").isNotNull())

    # Physically impossible: a whole grid region cannot have zero or negative demand
    .filter(col("demand_mwh") > 0)

    # Remove duplicate (timestamp, region) pairs — same as dropDuplicates(["order_id"]) in OLIST
    .dropDuplicates(["timestamp", "region"])
)

# Add a derived column for hour-of-day — same pattern as CRASH_HOUR in the flights notebook
# This will be useful in the Gold feature engineering step
demand_clean = (
    demand_clean
    .withColumn("hour_of_day",  hour(col("timestamp")))
    .withColumn("day_of_week",  dayofweek(col("timestamp")))   # 1=Sunday ... 7=Saturday
    .withColumn("month_of_year", month(col("timestamp")))
)

print("\nSilver schema (after cleaning):")
demand_clean.printSchema()

quality_report(demand_clean, "silver.eia_demand")

# COMMAND ----------

# Write to Silver Delta table
# Same write pattern used throughout the OLIST project:
# df.write.mode("overwrite").parquet(SILVER_PATH)
# Here we use Delta format instead of parquet — same idea, better for Databricks
demand_clean.write \
            .format("delta") \
            .mode("overwrite") \
            .saveAsTable(f"{SILVER}.eia_demand")

print(f"✓ Silver demand written: {demand_clean.count():,} rows → {SILVER}.eia_demand")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Section 4. Clean EIA Generation Mix Data
# MAGIC
# MAGIC **What we are fixing:**
# MAGIC - Same type casting as demand
# MAGIC - Deduplicate on (timestamp, region, fuel_type) — three columns this time
# MAGIC - Filter out negative generation values (instruments sometimes report -1 on error)
# MAGIC - Add a `is_renewable` flag using `F.when().otherwise()` — same pattern as the
# MAGIC   `categorical_rating` column from the ratings notebook

# COMMAND ----------

# Read Bronze generation table
generation_raw = spark.read.table(f"{BRONZE}.eia_generation")

print("Bronze schema:")
generation_raw.printSchema()

quality_report(generation_raw, "bronze.eia_generation")

# COMMAND ----------

# Renewable fuel types — we use this to create the is_renewable flag below
RENEWABLE_FUELS = ["SUN", "WND", "WAT", "GEO"]   # EIA codes: solar, wind, hydro, geothermal

generation_clean = (
    generation_raw

    # Type casting — same approach as demand
    .withColumn("timestamp",      to_timestamp(col("timestamp")))
    .withColumn("generation_mwh", col("generation_mwh").cast(DoubleType()))

    # Drop nulls in key columns
    .filter(col("timestamp").isNotNull())
    .filter(col("region").isNotNull())
    .filter(col("fuel_type").isNotNull())

    # Generation can be 0 (plant offline) but not meaningfully negative
    # Filter out extreme negatives (< -100) — small negatives can be measurement noise
    .filter(col("generation_mwh") >= -100)

    # Deduplicate on three columns — same dropDuplicates() pattern as in OLIST
    .dropDuplicates(["timestamp", "region", "fuel_type"])

    # Add is_renewable flag using F.when().otherwise()
    # Same pattern as the categorical_rating column in the ratings notebook
    .withColumn("is_renewable",
                F.when(F.col("fuel_type").isin(RENEWABLE_FUELS), 1)
                 .otherwise(0))

    # Standardise fuel_type to uppercase — same as WEATHER_CONDITION cleaning in flights notebook
    .withColumn("fuel_type", F.upper(F.trim(col("fuel_type"))))

    # Add time features
    .withColumn("hour_of_day",   hour(col("timestamp")))
    .withColumn("day_of_week",   dayofweek(col("timestamp")))
    .withColumn("month_of_year", month(col("timestamp")))
)

print("\nSilver schema:")
generation_clean.printSchema()

quality_report(generation_clean, "silver.eia_generation")

# COMMAND ----------

# Quick sense check — show average generation by fuel type using groupBy + agg
# Same groupBy pattern as in the MDA ratings and OLIST notebooks
print("Average generation by fuel type (sanity check):")
generation_clean \
    .groupBy("fuel_type", "is_renewable") \
    .agg(
        F.round(F.avg("generation_mwh"), 1).alias("avg_mwh"),
        F.count("*").alias("row_count")
    ) \
    .orderBy(F.desc("avg_mwh")) \
    .show(truncate=False)

# COMMAND ----------

generation_clean.write \
                .format("delta") \
                .mode("overwrite") \
                .saveAsTable(f"{SILVER}.eia_generation")

print(f"✓ Silver generation written: {generation_clean.count():,} rows → {SILVER}.eia_generation")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Section 5. Clean Weather Data
# MAGIC
# MAGIC **What we are fixing:**
# MAGIC - Cast timestamp to TimestampType
# MAGIC - Cast all numeric columns to DoubleType
# MAGIC - Fill missing weather values with the column median (weather sensors sometimes drop out)
# MAGIC - Validate temperature is within a physically reasonable range (−50°C to 60°C)
# MAGIC - Add a derived `wind_category` column using `F.when().otherwise()` — same Beaufort-like
# MAGIC   scale approach as the delay severity column in the flights notebook

# COMMAND ----------

weather_raw = spark.read.table(f"{BRONZE}.weather")

print("Bronze schema:")
weather_raw.printSchema()

quality_report(weather_raw, "bronze.weather")

# COMMAND ----------

# All numeric weather columns — we cast them all in one loop
WEATHER_NUMERIC_COLS = [
    "temperature_2m",
    "apparent_temperature",
    "shortwave_radiation",
    "wind_speed_10m",
    "wind_direction_10m",
    "cloud_cover",
    "precipitation",
    "relative_humidity_2m",
]

# Start with type casting and basic filters
weather_clean = weather_raw \
    .withColumn("timestamp", to_timestamp(col("timestamp"))) \
    .filter(col("timestamp").isNotNull()) \
    .filter(col("region").isNotNull()) \
    .dropDuplicates(["timestamp", "region"])

# Cast every numeric column to DoubleType in a loop
# Same idea as casting multiple columns in the MDA exercises
for c in WEATHER_NUMERIC_COLS:
    weather_clean = weather_clean.withColumn(c, col(c).cast(DoubleType()))

# COMMAND ----------

# Compute medians for filling missing values
# We use approxQuantile — same as the statistical functions shown in MDA
medians = {}
for c in WEATHER_NUMERIC_COLS:
    result = weather_clean.approxQuantile(c, [0.5], 0.01)
    medians[c] = round(result[0], 2) if result else 0.0

print("Median values used for null filling:")
for k, v in medians.items():
    print(f"   {k}: {v}")

# Fill nulls with the median — same as df.na.fill() from MDA notebooks
weather_clean = weather_clean.na.fill(medians)

# COMMAND ----------

# Validate temperature range — physically impossible values get filtered out
# Same filter chaining as used in the MDA crash and flights notebooks
weather_clean = (
    weather_clean
    .filter(col("temperature_2m").between(-50, 60))       # °C — impossible outside this range
    .filter(col("relative_humidity_2m").between(0, 100))  # humidity is always 0–100%
    .filter(col("shortwave_radiation") >= 0)              # radiation is never negative
    .filter(col("wind_speed_10m") >= 0)                   # wind speed is never negative
)

# COMMAND ----------

# Add derived columns — same F.when().otherwise() pattern as the flights notebook
# wind_category follows a simplified Beaufort scale
weather_clean = (
    weather_clean

    # Classify wind speed into categories
    .withColumn("wind_category",
                F.when(col("wind_speed_10m") < 5,  "calm")
                 .when(col("wind_speed_10m") < 15, "light")
                 .when(col("wind_speed_10m") < 30, "moderate")
                 .otherwise("strong"))

    # is_sunny: shortwave radiation above 200 W/m² means meaningful solar generation
    .withColumn("is_sunny",
                F.when(col("shortwave_radiation") > 200, 1).otherwise(0))

    # Time features — same as demand and generation above
    .withColumn("hour_of_day",   hour(col("timestamp")))
    .withColumn("day_of_week",   dayofweek(col("timestamp")))
    .withColumn("month_of_year", month(col("timestamp")))
)

print("\nSilver schema:")
weather_clean.printSchema()

quality_report(weather_clean, "silver.weather")

# COMMAND ----------

weather_clean.write \
             .format("delta") \
             .mode("overwrite") \
             .saveAsTable(f"{SILVER}.weather")

print(f"✓ Silver weather written: {weather_clean.count():,} rows → {SILVER}.weather")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Section 6. Silver Layer Summary
# MAGIC
# MAGIC Use `spark.sql()` to query the Silver tables and confirm everything looks right.
# MAGIC Same pattern as `createOrReplaceTempView` + `spark.sql()` from the MDA lectures.

# COMMAND ----------

# Register Silver tables as temp views so we can query them with SQL
# Same pattern used throughout the MDA course
spark.read.table(f"{SILVER}.eia_demand").createOrReplaceTempView("silver_demand")
spark.read.table(f"{SILVER}.eia_generation").createOrReplaceTempView("silver_generation")
spark.read.table(f"{SILVER}.weather").createOrReplaceTempView("silver_weather")

# COMMAND ----------

# How many rows per region in the demand table?
spark.sql("""
    SELECT   region,
             COUNT(*)                          AS total_rows,
             MIN(timestamp)                    AS earliest,
             MAX(timestamp)                    AS latest,
             ROUND(AVG(demand_mwh), 0)         AS avg_demand_mwh,
             ROUND(MAX(demand_mwh), 0)         AS peak_demand_mwh
    FROM     silver_demand
    GROUP BY region
    ORDER BY avg_demand_mwh DESC
""").show(truncate=False)

# COMMAND ----------

# What is the renewable share by region?
spark.sql("""
    SELECT   region,
             ROUND(SUM(CASE WHEN is_renewable = 1 THEN generation_mwh ELSE 0 END) /
                   SUM(generation_mwh) * 100, 1) AS renewable_pct
    FROM     silver_generation
    GROUP BY region
    ORDER BY renewable_pct DESC
""").show(truncate=False)

# COMMAND ----------

# Weather overview
spark.sql("""
    SELECT   region,
             COUNT(*)                               AS rows,
             ROUND(AVG(temperature_2m), 1)          AS avg_temp_c,
             ROUND(AVG(wind_speed_10m), 1)          AS avg_wind_ms,
             ROUND(AVG(shortwave_radiation), 0)     AS avg_radiation,
             SUM(is_sunny)                          AS sunny_hours
    FROM     silver_weather
    GROUP BY region
    ORDER BY avg_temp_c DESC
""").show(truncate=False)

# COMMAND ----------

print("\n" + "=" * 55)
print("🥈 SILVER LAYER COMPLETE")
print("=" * 55)
print(f"\n  {SILVER}.eia_demand    ✓")
print(f"  {SILVER}.eia_generation ✓")
print(f"  {SILVER}.weather        ✓")
print(f"\n  Next step → Run 03_gold_features.py")
