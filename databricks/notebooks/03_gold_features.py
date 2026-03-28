# Databricks notebook source

# MAGIC %md
# MAGIC # 🥇 Gold Layer — Feature Engineering
# MAGIC
# MAGIC **WATT Energy Intelligence Platform**
# MAGIC
# MAGIC In this notebook we read the clean Silver Delta tables, engineer 30+ ML-ready
# MAGIC features, and write a single Gold Delta table used by all four ML models.
# MAGIC
# MAGIC The Silver → Gold step is about **richness**: joining datasets, creating lag
# MAGIC features, rolling statistics, and calendar dummies so the models have every
# MAGIC signal they need in one flat table.
# MAGIC
# MAGIC | Input (Silver)                | Output (Gold)               | What we add                              |
# MAGIC |-------------------------------|-----------------------------|------------------------------------------|
# MAGIC | `watt.silver.eia_demand`      | `watt.gold.ml_features`     | lag features, rolling stats              |
# MAGIC | `watt.silver.eia_generation`  | `watt.gold.ml_features`     | renewable share, fuel mix                |
# MAGIC | `watt.silver.weather`         | `watt.gold.ml_features`     | weather signals, interactions            |

# COMMAND ----------

# MAGIC %md
# MAGIC ## Section 1. Setup

# COMMAND ----------

# Same imports used throughout the MDA course
import pyspark.sql.functions as F
from pyspark.sql.functions import (
    col, lag, avg, stddev, max as spark_max, min as spark_min,
    to_timestamp, hour, dayofweek, month, when, sum as spark_sum,
    round as spark_round, log1p, lit
)
# Window is the key import for lag and rolling features — covered in MDA Session 5
from pyspark.sql.window import Window

CATALOG = "watt"
SILVER  = f"{CATALOG}.silver"
GOLD    = f"{CATALOG}.gold"

# Create the Gold schema if it doesn't already exist
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {GOLD}")

print(f"Reading from  : {SILVER}")
print(f"Writing to    : {GOLD}")
print(f"Spark version : {spark.version}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Section 2. Load Silver Tables
# MAGIC
# MAGIC Same `.read.table()` pattern used in the Silver notebook and OLIST project.
# MAGIC We register each table as a temp view straight away so we can also
# MAGIC query them with `spark.sql()` later — same as `createOrReplaceTempView()`
# MAGIC from the MDA lectures.

# COMMAND ----------

# Read demand — same read pattern as the Silver notebook
demand   = spark.read.table(f"{SILVER}.eia_demand")
gen      = spark.read.table(f"{SILVER}.eia_generation")
weather  = spark.read.table(f"{SILVER}.weather")

# Register as temp views so we can use spark.sql() on them
# Exact same pattern as in the MDA flights and OLIST notebooks
demand.createOrReplaceTempView("sv_demand")
gen.createOrReplaceTempView("sv_generation")
weather.createOrReplaceTempView("sv_weather")

print("Silver row counts:")
print(f"  demand      : {demand.count():,}")
print(f"  generation  : {gen.count():,}")
print(f"  weather     : {weather.count():,}")

# Quick look at the schemas — same as printSchema() in every MDA notebook
print("\nDemand schema:")
demand.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Section 3. Build Renewable Share per (timestamp, region)
# MAGIC
# MAGIC Before joining, we aggregate the generation table so each (timestamp, region)
# MAGIC pair has one row summarising the full fuel mix.
# MAGIC
# MAGIC We use `groupBy().agg()` — the same pattern as the OLIST revenue aggregation
# MAGIC and the MDA ratings notebook.

# COMMAND ----------

# Aggregate generation to one row per (timestamp, region)
# groupBy + agg — same as groupBy("customer_city").agg(F.sum(...)) in OLIST
gen_agg = (
    gen
    .groupBy("timestamp", "region")
    .agg(
        # Total generation across all fuel types
        spark_round(spark_sum("generation_mwh"), 2).alias("total_gen_mwh"),

        # Renewable generation only — CASE WHEN equivalent using F.when()
        # Same as the is_renewable flag we built in the Silver notebook
        spark_round(
            spark_sum(F.when(col("is_renewable") == 1, col("generation_mwh")).otherwise(0)), 2
        ).alias("renewable_gen_mwh"),

        # Solar share — EIA code SUN
        spark_round(
            spark_sum(F.when(col("fuel_type") == "SUN", col("generation_mwh")).otherwise(0)), 2
        ).alias("solar_gen_mwh"),

        # Wind share — EIA code WND
        spark_round(
            spark_sum(F.when(col("fuel_type") == "WND", col("generation_mwh")).otherwise(0)), 2
        ).alias("wind_gen_mwh"),

        # Natural gas — NG
        spark_round(
            spark_sum(F.when(col("fuel_type") == "NG",  col("generation_mwh")).otherwise(0)), 2
        ).alias("gas_gen_mwh"),

        # Coal — COL
        spark_round(
            spark_sum(F.when(col("fuel_type") == "COL", col("generation_mwh")).otherwise(0)), 2
        ).alias("coal_gen_mwh"),
    )
)

# Derived ratio: renewable % of total
# F.when() guard prevents division by zero — same pattern used for rate calculations in MDA
gen_agg = gen_agg.withColumn(
    "renewable_pct",
    F.when(col("total_gen_mwh") > 0,
           spark_round(col("renewable_gen_mwh") / col("total_gen_mwh") * 100, 2))
    .otherwise(lit(0.0))
)

print("Generation aggregated schema:")
gen_agg.printSchema()
print(f"Rows: {gen_agg.count():,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Section 4. Join Demand + Generation + Weather
# MAGIC
# MAGIC A three-way join on (timestamp, region).
# MAGIC
# MAGIC We use `left join` so that demand rows are never dropped if a generation or
# MAGIC weather row is missing — same defensive join strategy discussed in MDA.
# MAGIC Same `.join()` syntax used in the flights and OLIST merge notebooks.

# COMMAND ----------

# Step 1: Join demand ← generation aggregates
# .join(other, on=[...], how="left") — same as df.merge(..., how="left") in pandas MDA exercises
joined = demand.join(
    gen_agg,
    on=["timestamp", "region"],
    how="left"
)

# Step 2: Join ← weather
# We only need the weather sensor columns (not the duplicate time columns)
weather_cols = [
    "timestamp", "region",
    "temperature_2m", "apparent_temperature",
    "shortwave_radiation", "wind_speed_10m",
    "wind_direction_10m", "cloud_cover",
    "precipitation", "relative_humidity_2m",
    "wind_category", "is_sunny"
]

joined = joined.join(
    weather.select(weather_cols),
    on=["timestamp", "region"],
    how="left"
)

print("Joined schema:")
joined.printSchema()
print(f"Joined rows: {joined.count():,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Section 5. Lag Features with Window Functions
# MAGIC
# MAGIC Lag features tell the model what demand looked like 1 hour ago, 24 hours ago,
# MAGIC and 168 hours ago (one week ago).  These are the strongest predictors in
# MAGIC time-series energy forecasting.
# MAGIC
# MAGIC We use `Window.partitionBy().orderBy()` and `F.lag()` — this is the
# MAGIC **Window function** pattern from MDA Session 5.
# MAGIC
# MAGIC `partitionBy("region")` means the lag is computed independently for each
# MAGIC grid region, exactly like `partitionBy("customer_state")` in the OLIST window
# MAGIC notebook.

# COMMAND ----------

# Define a Window spec partitioned by region, ordered by time
# Same as Window.partitionBy("customer_state").orderBy("order_purchase_timestamp")
# from the OLIST window functions exercise
demand_window = Window.partitionBy("region").orderBy("timestamp")

# Add lag features — F.lag(col, offset) looks back `offset` rows within the window
# Lag 1  = what demand was 1 hour ago
# Lag 24 = same hour yesterday
# Lag 168 = same hour last week (7 days × 24 hours)
joined = (
    joined
    .withColumn("demand_lag_1h",   lag(col("demand_mwh"), 1  ).over(demand_window))
    .withColumn("demand_lag_24h",  lag(col("demand_mwh"), 24 ).over(demand_window))
    .withColumn("demand_lag_168h", lag(col("demand_mwh"), 168).over(demand_window))
)

# Same for renewable_pct — helps the renewable forecaster model
joined = (
    joined
    .withColumn("renewable_lag_24h",  lag(col("renewable_pct"), 24 ).over(demand_window))
    .withColumn("renewable_lag_168h", lag(col("renewable_pct"), 168).over(demand_window))
)

print("Lag features added.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Section 6. Rolling Window Statistics
# MAGIC
# MAGIC Rolling averages and standard deviations smooth out noise and give the model
# MAGIC a short-term trend signal.
# MAGIC
# MAGIC We use `.rowsBetween(start, 0)` to define a sliding window of N past rows
# MAGIC — this is the rolling window pattern from MDA Session 5.
# MAGIC
# MAGIC `rowsBetween(-N, 0)` means "from N rows before the current row up to (and
# MAGIC including) the current row" — equivalent to pandas `df.rolling(N).mean()`.

# COMMAND ----------

# Rolling windows of different sizes — all partitioned by region
# -5  means 6-hour rolling  (current + 5 previous hours)
# -23 means 24-hour rolling (current + 23 previous hours)
# -167 means 168-hour rolling (current + 167 previous hours)
window_6h   = Window.partitionBy("region").orderBy("timestamp").rowsBetween(-5,   0)
window_24h  = Window.partitionBy("region").orderBy("timestamp").rowsBetween(-23,  0)
window_168h = Window.partitionBy("region").orderBy("timestamp").rowsBetween(-167, 0)

joined = (
    joined

    # ── Demand rolling stats ─────────────────────────────────────────────────
    # Rolling mean demand — smoothed trend signal
    .withColumn("demand_roll_mean_6h",   spark_round(avg(col("demand_mwh")).over(window_6h),   1))
    .withColumn("demand_roll_mean_24h",  spark_round(avg(col("demand_mwh")).over(window_24h),  1))
    .withColumn("demand_roll_mean_168h", spark_round(avg(col("demand_mwh")).over(window_168h), 1))

    # Rolling std — measures volatility; high std = unusual demand period
    .withColumn("demand_roll_std_6h",   spark_round(stddev(col("demand_mwh")).over(window_6h),  1))
    .withColumn("demand_roll_std_24h",  spark_round(stddev(col("demand_mwh")).over(window_24h), 1))

    # Rolling max — peak demand in window (important for grid stress detection)
    .withColumn("demand_roll_max_24h",  spark_round(spark_max(col("demand_mwh")).over(window_24h), 1))

    # ── Temperature rolling stats ────────────────────────────────────────────
    .withColumn("temp_roll_mean_24h", spark_round(avg(col("temperature_2m")).over(window_24h), 2))
    .withColumn("temp_roll_mean_168h",spark_round(avg(col("temperature_2m")).over(window_168h),2))
)

print("Rolling window features added.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Section 7. Calendar & Cyclical Features
# MAGIC
# MAGIC Time features let the model learn daily, weekly, and seasonal patterns.
# MAGIC
# MAGIC - `is_weekend` uses the same `F.when().otherwise()` pattern as the Silver notebook
# MAGIC - Sine/cosine encoding of hour and month converts circular time into continuous
# MAGIC   signals — this avoids the model thinking hour 23 and hour 0 are "far apart"

# COMMAND ----------

# is_weekend flag — same F.when().otherwise() pattern as the categorical_rating column
# in the MDA ratings notebook
# day_of_week: 1 = Sunday, 7 = Saturday in Spark
joined = joined.withColumn(
    "is_weekend",
    F.when(col("day_of_week").isin([1, 7]), 1).otherwise(0)
)

# is_peak_hour — morning and evening peaks in energy demand (7–10am, 5–9pm)
joined = joined.withColumn(
    "is_peak_hour",
    F.when(col("hour_of_day").between(7, 10) | col("hour_of_day").between(17, 21), 1)
    .otherwise(0)
)

# Cyclical encoding of hour_of_day using sine and cosine
# This is the standard trick for encoding periodic features in ML
# Hour 0 and hour 23 should be "close" to each other — raw integers don't capture this
import math
joined = (
    joined
    .withColumn("hour_sin",  spark_round(F.sin(col("hour_of_day")  * (2 * math.pi / 24)),  4))
    .withColumn("hour_cos",  spark_round(F.cos(col("hour_of_day")  * (2 * math.pi / 24)),  4))
    .withColumn("month_sin", spark_round(F.sin(col("month_of_year") * (2 * math.pi / 12)), 4))
    .withColumn("month_cos", spark_round(F.cos(col("month_of_year") * (2 * math.pi / 12)), 4))
)

print("Calendar and cyclical features added.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Section 8. Interaction & Derived Features
# MAGIC
# MAGIC Domain knowledge about energy: demand spikes when it is very hot (air conditioning)
# MAGIC or very cold (heating).  We capture this with:
# MAGIC - `temp_x_hour` — temperature × hour of day (heat stress during peak hours)
# MAGIC - `heating_degrees` and `cooling_degrees` — classic energy industry features
# MAGIC   measuring how far temperature deviates from the comfort zone (18°C)
# MAGIC - `demand_vs_roll_mean` — ratio of current demand to 24h rolling mean;
# MAGIC   a ratio > 1 flags unusually high demand (anomaly signal)

# COMMAND ----------

COMFORT_TEMP = 18.0   # Degrees Celsius — standard comfort baseline in energy modelling

joined = (
    joined

    # Temperature × hour interaction — high temp during peak hours = strong AC signal
    .withColumn("temp_x_hour",
                spark_round(col("temperature_2m") * col("hour_of_day"), 2))

    # Heating Degree: how much colder than comfort? (0 if above comfort)
    # Same F.when() guard pattern as the renewable_pct division above
    .withColumn("heating_degrees",
                F.when(col("temperature_2m") < COMFORT_TEMP,
                       spark_round(lit(COMFORT_TEMP) - col("temperature_2m"), 2))
                 .otherwise(lit(0.0)))

    # Cooling Degree: how much hotter than comfort? (0 if below comfort)
    .withColumn("cooling_degrees",
                F.when(col("temperature_2m") > COMFORT_TEMP,
                       spark_round(col("temperature_2m") - lit(COMFORT_TEMP), 2))
                 .otherwise(lit(0.0)))

    # Demand vs 24h rolling mean — ratio > 1 means anomalously high demand
    # F.when() prevents division by zero
    .withColumn("demand_vs_roll_mean",
                F.when(col("demand_roll_mean_24h") > 0,
                       spark_round(col("demand_mwh") / col("demand_roll_mean_24h"), 4))
                 .otherwise(lit(1.0)))

    # Log-transformed demand — reduces skew, often improves regression models
    # log1p(x) = log(1 + x) so it handles zeros safely — same as the MDA log transform cell
    .withColumn("demand_log",
                spark_round(log1p(col("demand_mwh")), 4))

    # Wind power potential: proportional to wind_speed³ (physics of wind turbines)
    .withColumn("wind_power_potential",
                spark_round(F.pow(col("wind_speed_10m"), 3), 2))

    # Cloud cover × shortwave radiation interaction
    .withColumn("effective_solar",
                spark_round((lit(100) - col("cloud_cover")) / 100 * col("shortwave_radiation"), 2))
)

print("Interaction and derived features added.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Section 9. Region One-Hot Encoding
# MAGIC
# MAGIC Tree-based models (XGBoost, LightGBM) handle categorical strings natively,
# MAGIC but we also create binary region dummies for models that prefer numeric input.
# MAGIC
# MAGIC Same `F.when().otherwise()` pattern used in the Silver notebook for `is_renewable`.

# COMMAND ----------

# One-hot encode the 6 US regions
# Same approach as converting categorical columns to binary flags in the MDA exercises
REGIONS = ["CAL", "TEX", "NY", "NE", "MIDW", "NW"]

for region_code in REGIONS:
    joined = joined.withColumn(
        f"region_{region_code}",
        F.when(col("region") == region_code, 1).otherwise(0)
    )

print(f"Region dummies added for: {REGIONS}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Section 10. Select Final Feature Set
# MAGIC
# MAGIC We select only the columns we want in the Gold table, giving each a clear
# MAGIC name.  Same explicit `.select()` pattern from the MDA data preparation notebooks.

# COMMAND ----------

# Final column selection — explicit list so we know exactly what goes into the models
GOLD_COLS = [
    # ── Identity ────────────────────────────────────────────────────────────
    "timestamp",
    "region",

    # ── Target variable ─────────────────────────────────────────────────────
    "demand_mwh",         # what we are forecasting
    "demand_log",         # log-transformed target (optional)

    # ── Lag features ────────────────────────────────────────────────────────
    "demand_lag_1h",
    "demand_lag_24h",
    "demand_lag_168h",
    "renewable_lag_24h",
    "renewable_lag_168h",

    # ── Rolling stats ────────────────────────────────────────────────────────
    "demand_roll_mean_6h",
    "demand_roll_mean_24h",
    "demand_roll_mean_168h",
    "demand_roll_std_6h",
    "demand_roll_std_24h",
    "demand_roll_max_24h",
    "temp_roll_mean_24h",
    "temp_roll_mean_168h",

    # ── Calendar features ────────────────────────────────────────────────────
    "hour_of_day",
    "day_of_week",
    "month_of_year",
    "is_weekend",
    "is_peak_hour",
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",

    # ── Weather features ─────────────────────────────────────────────────────
    "temperature_2m",
    "apparent_temperature",
    "shortwave_radiation",
    "wind_speed_10m",
    "wind_direction_10m",
    "cloud_cover",
    "precipitation",
    "relative_humidity_2m",
    "wind_category",
    "is_sunny",

    # ── Interaction features ──────────────────────────────────────────────────
    "temp_x_hour",
    "heating_degrees",
    "cooling_degrees",
    "demand_vs_roll_mean",
    "wind_power_potential",
    "effective_solar",

    # ── Generation / renewables ───────────────────────────────────────────────
    "total_gen_mwh",
    "renewable_gen_mwh",
    "renewable_pct",
    "solar_gen_mwh",
    "wind_gen_mwh",
    "gas_gen_mwh",
    "coal_gen_mwh",

    # ── Region dummies ────────────────────────────────────────────────────────
    "region_CAL", "region_TEX", "region_NY",
    "region_NE",  "region_MIDW", "region_NW",
]

gold_df = joined.select(GOLD_COLS)

print(f"Gold feature set: {len(GOLD_COLS)} columns")
print(f"Gold rows before dropping rows with null lag cols: {gold_df.count():,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Section 11. Drop Rows That Lack Lag Features
# MAGIC
# MAGIC The first few rows for each region will have null lag values because there
# MAGIC are not enough prior rows to look back 168 hours.  We drop those rows so
# MAGIC the models train on complete records only.
# MAGIC
# MAGIC Same `.filter(col(...).isNotNull())` pattern from the Silver cleaning notebook.

# COMMAND ----------

# Drop rows where any of the key lag features are null
# These are the first 168 rows per region — unavoidable with lag features
LAG_COLS = ["demand_lag_1h", "demand_lag_24h", "demand_lag_168h"]

for lag_col in LAG_COLS:
    gold_df = gold_df.filter(col(lag_col).isNotNull())

print(f"Gold rows after dropping incomplete lag rows: {gold_df.count():,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Section 12. Quality Check
# MAGIC
# MAGIC Same null-counting pattern used in the Silver `quality_report()` function.
# MAGIC Here we focus on the lag and rolling columns since those are the ones most
# MAGIC likely to have residual nulls.

# COMMAND ----------

# Check null counts for the key feature groups
# Same pattern as quality_report() in the Silver notebook
CHECK_COLS = LAG_COLS + [
    "demand_roll_mean_24h",
    "temperature_2m",
    "renewable_pct",
    "demand_vs_roll_mean",
]

null_check = gold_df.select([
    F.count(F.when(F.col(c).isNull(), c)).alias(c)
    for c in CHECK_COLS
])

print("Null counts in key feature columns (should all be 0):")
null_check.show(truncate=False)

# COMMAND ----------

# Quick summary statistics — same as describe() in pandas, .describe() in Spark
print("Summary statistics for numeric features:")
gold_df.select(
    "demand_mwh", "demand_lag_24h", "temperature_2m",
    "renewable_pct", "demand_vs_roll_mean"
).describe().show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Section 13. Write Gold Delta Table
# MAGIC
# MAGIC Same `.write.format("delta").mode("overwrite").saveAsTable()` pattern
# MAGIC used in the Silver notebook.

# COMMAND ----------

# Write to Gold Delta table
# Same write pattern used in Silver — Delta format gives us ACID transactions
# and allows downstream ML jobs to read consistent snapshots
gold_df.write \
       .format("delta") \
       .mode("overwrite") \
       .saveAsTable(f"{GOLD}.ml_features")

print(f"✓ Gold table written: {gold_df.count():,} rows → {GOLD}.ml_features")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Section 14. Gold Layer Summary
# MAGIC
# MAGIC Register the Gold table as a temp view and run SQL sanity checks.
# MAGIC Same `createOrReplaceTempView()` + `spark.sql()` pattern from MDA.

# COMMAND ----------

# Register Gold table as a temp view for SQL queries
# Same pattern as in the Silver notebook Section 6
spark.read.table(f"{GOLD}.ml_features").createOrReplaceTempView("gold_features")

# COMMAND ----------

# How many feature rows do we have per region?
spark.sql("""
    SELECT   region,
             COUNT(*)                            AS total_rows,
             MIN(timestamp)                      AS earliest,
             MAX(timestamp)                      AS latest,
             ROUND(AVG(demand_mwh), 0)           AS avg_demand_mwh,
             ROUND(AVG(temperature_2m), 1)       AS avg_temp_c,
             ROUND(AVG(renewable_pct), 1)        AS avg_renewable_pct,
             ROUND(AVG(demand_vs_roll_mean), 3)  AS avg_anomaly_ratio
    FROM     gold_features
    GROUP BY region
    ORDER BY avg_demand_mwh DESC
""").show(truncate=False)

# COMMAND ----------

# Are there any demand spikes (anomaly candidates)?
# demand_vs_roll_mean > 1.3 means demand is 30% above its 24h rolling average
spark.sql("""
    SELECT   region,
             timestamp,
             ROUND(demand_mwh, 0)          AS demand_mwh,
             ROUND(demand_roll_mean_24h, 0) AS roll_mean_24h,
             ROUND(demand_vs_roll_mean, 3)  AS anomaly_ratio
    FROM     gold_features
    WHERE    demand_vs_roll_mean > 1.30
    ORDER BY anomaly_ratio DESC
    LIMIT    20
""").show(truncate=False)

# COMMAND ----------

# Renewable share by region and hour — shows the solar + wind daily profile
spark.sql("""
    SELECT   region,
             hour_of_day,
             ROUND(AVG(renewable_pct), 1)       AS avg_renewable_pct,
             ROUND(AVG(solar_gen_mwh), 0)       AS avg_solar_mwh,
             ROUND(AVG(wind_gen_mwh), 0)        AS avg_wind_mwh,
             ROUND(AVG(effective_solar), 1)     AS avg_effective_solar
    FROM     gold_features
    GROUP BY region, hour_of_day
    ORDER BY region, hour_of_day
""").show(50, truncate=False)

# COMMAND ----------

# Weekend vs weekday demand comparison — using the is_weekend flag we built
spark.sql("""
    SELECT   region,
             CASE WHEN is_weekend = 1 THEN 'Weekend' ELSE 'Weekday' END AS day_type,
             ROUND(AVG(demand_mwh), 0)           AS avg_demand_mwh,
             ROUND(AVG(renewable_pct), 1)        AS avg_renewable_pct,
             COUNT(*)                            AS rows
    FROM     gold_features
    GROUP BY region, is_weekend
    ORDER BY region, is_weekend
""").show(truncate=False)

# COMMAND ----------

# Feature column count sanity check
feature_count = len(spark.read.table(f"{GOLD}.ml_features").columns)
print(f"\nTotal columns in Gold table: {feature_count}")

print("\n" + "=" * 55)
print("🥇 GOLD LAYER COMPLETE")
print("=" * 55)
print(f"\n  {GOLD}.ml_features  ✓")
print(f"  {feature_count} features ready for ML models")
print(f"\n  Next step → Run 04_eda.py  (or  05_demand_forecaster.py)")
