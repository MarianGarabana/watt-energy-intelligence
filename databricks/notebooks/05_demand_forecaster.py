# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "2"
# ///
# MAGIC %md
# MAGIC # ⚡ WATT — Demand Forecaster (Production)
# MAGIC
# MAGIC **WATT Energy Intelligence Platform — Day 5**
# MAGIC
# MAGIC This is the **production version** of the demand forecaster.
# MAGIC We tested models locally in `05_demand_forecaster_exploration.ipynb`,
# MAGIC found the best hyperparameters, and now we:
# MAGIC
# MAGIC 1. Read the full Gold Delta table from Databricks
# MAGIC 2. Train XGBoost with the winning hyperparameters
# MAGIC 3. Track the run with **MLflow** (built into Databricks — no server needed)
# MAGIC 4. Register the model in the **MLflow Model Registry**
# MAGIC 5. Log SHAP feature importance plots as artifacts

# COMMAND ----------

# MAGIC %md
# MAGIC ## Section 1. Setup
# MAGIC xgboost, lightgbm, shap are pre-installed in Databricks serverless environment v2.

# COMMAND ----------

import warnings
import time

import pandas as pd  # Updated pandas version or check for compatibility
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb
import shap
import mlflow
import mlflow.xgboost

warnings.filterwarnings('ignore')
sns.set_theme(style='whitegrid', palette='muted', font_scale=1.1)

# Fixed seed — same as every IE class project
RANDOM_STATE = 42

CATALOG  = 'watt'
GOLD     = f'{CATALOG}.gold'
TARGET   = 'demand_mwh'
EXP_NAME = '/Users/marian.garabana@student.ie.edu/watt-demand-forecaster'

# In Databricks, MLflow tracking is built-in — full path required for serverless
mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment(EXP_NAME)

print(f'MLflow experiment : {EXP_NAME}')
print(f'Reading from      : {GOLD}.ml_features')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Section 2. Load Gold Table

# COMMAND ----------

# Read Gold Delta table → pandas (same toPandas() pattern from MDA notebooks)
# Sort by timestamp before converting — Delta tables have no guaranteed row order,
# and split_df(shuffle=False) depends on chronological order being preserved.
gold_spark = spark.read.table(f'{GOLD}.ml_features').orderBy('timestamp')
df = gold_spark.toPandas()

print(f'Shape: {df.shape[0]:,} rows × {df.shape[1]} columns')
df[['demand_mwh', 'demand_lag_24h', 'temperature_2m', 'renewable_pct']].describe().round(1)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Section 3. Define Features & Target
# MAGIC
# MAGIC Same explicit feature list from the exploration notebook —
# MAGIC paste the winner's feature set here after exploration.

# COMMAND ----------

FEATURES = [
    # Lag features — strongest predictors (from EDA correlation matrix)
    'demand_lag_1h', 'demand_lag_24h', 'demand_lag_168h',
    'renewable_lag_24h',
    # Rolling stats
    'demand_roll_mean_24h', 'demand_roll_std_24h',
    'demand_roll_mean_168h', 'demand_roll_max_24h',
    # Calendar
    'hour_of_day', 'day_of_week', 'month_of_year',
    'is_weekend', 'is_peak_hour',
    'hour_sin', 'hour_cos', 'month_sin', 'month_cos',
    # Weather
    'temperature_2m', 'heating_degrees', 'cooling_degrees',
    'temp_x_hour', 'wind_speed_10m', 'shortwave_radiation',
    'cloud_cover', 'precipitation', 'relative_humidity_2m',
    # Domain interactions
    'renewable_pct', 'demand_vs_roll_mean',
    'wind_power_potential', 'effective_solar',
]

X = df[FEATURES]
y = df[TARGET]

print(f'Features : {len(FEATURES)}')
print(f'Target   : {TARGET}  (mean={y.mean():,.0f}, std={y.std():,.0f})')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Section 4. Chronological Train / Test Split
# MAGIC
# MAGIC Same `split_df()` function from exploration notebook.
# MAGIC `shuffle=False` preserves time order — the test set is always the most recent 20%.

# COMMAND ----------

def split_df(X, y, percentage=0.8, seed=None, time_series=True):
    return train_test_split(
        X, y,
        test_size=1 - percentage,
        random_state=seed,
        shuffle=not time_series
    )

X_train, X_test, Y_train, Y_test = split_df(X, y, percentage=0.8, seed=RANDOM_STATE)

print(f'Training set : {X_train.shape[0]:,} rows  ({X_train.shape[0]/len(X)*100:.0f}%)')
print(f'Test set     : {X_test.shape[0]:,} rows   ({X_test.shape[0]/len(X)*100:.0f}%)')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Section 5. Baseline

# COMMAND ----------

mean_value     = Y_train.mean()
baseline_preds = np.full(shape=Y_test.shape, fill_value=mean_value)
baseline_mae   = mean_absolute_error(Y_test, baseline_preds)
baseline_rmse  = np.sqrt(mean_squared_error(Y_test, baseline_preds))

print(f'Baseline MAE  : {baseline_mae:>10,.1f} MWh  ← must beat this')
print(f'Baseline RMSE : {baseline_rmse:>10,.1f} MWh')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Section 6. evaluate_regression_performance() Helper
# MAGIC
# MAGIC Same reusable helper from the exploration notebook and Boosting class notebook.

# COMMAND ----------

def evaluate_regression_performance(cv_obj, X_train, Y_train, X_test, Y_test, model_name='Model'):
    best_model = cv_obj.best_estimator_

    train_preds = best_model.predict(X_train)
    train_mae   = mean_absolute_error(Y_train, train_preds)
    train_rmse  = np.sqrt(mean_squared_error(Y_train, train_preds))
    train_r2    = r2_score(Y_train, train_preds)

    cv_mae = abs(cv_obj.best_score_)

    test_preds = best_model.predict(X_test)
    test_mae   = mean_absolute_error(Y_test, test_preds)
    test_rmse  = np.sqrt(mean_squared_error(Y_test, test_preds))
    test_r2    = r2_score(Y_test, test_preds)

    print(f'─' * 52)
    print(f'  {model_name} — REGRESSION PERFORMANCE')
    print(f'  Best params : {cv_obj.best_params_}')
    print(f'─' * 52)
    print(f'  {"":25s}  MAE        RMSE       R²')
    print(f'  {"Train":25s}  {train_mae:>8,.0f}   {train_rmse:>8,.0f}   {train_r2:.4f}')
    print(f'  {"CV (3-fold)":25s}  {cv_mae:>8,.0f}')
    print(f'  {"Test":25s}  {test_mae:>8,.0f}   {test_rmse:>8,.0f}   {test_r2:.4f}')
    print(f'  {"Baseline":25s}  {baseline_mae:>8,.0f}   {baseline_rmse:>8,.0f}')
    print(f'─' * 52)
    improvement = (1 - test_mae / baseline_mae) * 100
    print(f'  ✓ Improvement vs baseline: {improvement:.1f}%')

    return test_preds, test_mae, test_rmse, test_r2

# COMMAND ----------

# MAGIC %md
# MAGIC ## Section 7. XGBoost — GridSearchCV + MLflow Tracking
# MAGIC
# MAGIC Same `GridSearchCV(cv=3, scoring='neg_mean_absolute_error')` from the
# MAGIC Boosting class notebook, now wrapped in `with mlflow.start_run()` from the
# MAGIC MLOps scenario-2 notebook.

# COMMAND ----------

# Best hyperparameters from the exploration notebook — paste results here
# (or run GridSearchCV again on the full dataset)
base_xgb = xgb.XGBRegressor(
    objective     = 'reg:squarederror',
    random_state  = RANDOM_STATE,
    tree_method   = 'hist',
    verbosity     = 0
)

xgb_params = {
    'n_estimators'     : [300, 500],
    'max_depth'        : [4, 6],
    'learning_rate'    : [0.05, 0.1],
    'subsample'        : [0.8],
    'colsample_bytree' : [0.7],
    'min_child_weight' : [5, 10],
}

# ── MLflow run — same with mlflow.start_run() as run: pattern from scenario-2 ──
with mlflow.start_run(run_name='xgboost_gridsearch') as run:

    # GridSearchCV — same as Boosting notebook (cv=3, n_jobs=-2, verbose=1)
    start_time = time.time()
    grid_xgb = GridSearchCV(
        estimator  = base_xgb,
        param_grid = xgb_params,
        scoring    = 'neg_mean_absolute_error',
        cv         = 3,
        n_jobs     = -2,
        verbose    = 1
    )
    grid_xgb.fit(X_train, Y_train)
    elapsed = time.time() - start_time

    print(f'\nCompleted in {elapsed:.0f} seconds')
    print(f'Best params: {grid_xgb.best_params_}')

    # Evaluate
    xgb_preds, xgb_mae, xgb_rmse, xgb_r2 = evaluate_regression_performance(
        grid_xgb, X_train, Y_train, X_test, Y_test, 'XGBoost'
    )

    # Log params — same mlflow.log_params(params) as scenario-2
    mlflow.log_params(grid_xgb.best_params_)
    mlflow.log_param('model_type',   'XGBoost')
    mlflow.log_param('n_features',   len(FEATURES))
    mlflow.log_param('train_rows',   len(X_train))
    mlflow.log_param('cv_folds',     3)

    # Log metrics — same mlflow.log_metric('accuracy', acc) as scenario-2
    mlflow.log_metric('test_mae',       xgb_mae)
    mlflow.log_metric('test_rmse',      xgb_rmse)
    mlflow.log_metric('test_r2',        xgb_r2)
    mlflow.log_metric('cv_mae',         abs(grid_xgb.best_score_))
    mlflow.log_metric('baseline_mae',   baseline_mae)
    mlflow.log_metric('training_sec',   elapsed)

    # Tags — same mlflow.set_tag() as scenario-2
    mlflow.set_tag('target',    TARGET)
    mlflow.set_tag('project',   'WATT Energy Intelligence')
    mlflow.set_tag('stage',     'production')

    # Log model — same mlflow.xgboost.log_model() pattern
    mlflow.xgboost.log_model(
    grid_xgb.best_estimator_,
    artifact_path = 'xgboost_demand_forecaster',
    input_example = X_test.iloc[:5])


    xgb_run_id = run.info.run_id
    print(f'\n✓ XGBoost run logged: {xgb_run_id}')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Section 8. LightGBM — GridSearchCV + MLflow Tracking

# COMMAND ----------

base_lgb = lgb.LGBMRegressor(
    objective    = 'regression',
    random_state = RANDOM_STATE,
    verbose      = -1
)

lgb_params = {
    'n_estimators'      : [300, 500],
    'max_depth'         : [4, 6],
    'learning_rate'     : [0.05, 0.1],
    'subsample'         : [0.8],
    'num_leaves'        : [31, 63],
    'min_child_samples' : [20],
}

with mlflow.start_run(run_name='lightgbm_gridsearch') as run:

    start_time = time.time()
    grid_lgb = GridSearchCV(
        estimator  = base_lgb,
        param_grid = lgb_params,
        scoring    = 'neg_mean_absolute_error',
        cv         = 3,
        n_jobs     = -2,
        verbose    = 1
    )
    grid_lgb.fit(X_train, Y_train)
    elapsed = time.time() - start_time

    print(f'\nCompleted in {elapsed:.0f} seconds')

    lgb_preds, lgb_mae, lgb_rmse, lgb_r2 = evaluate_regression_performance(
        grid_lgb, X_train, Y_train, X_test, Y_test, 'LightGBM'
    )

    mlflow.log_params(grid_lgb.best_params_)
    mlflow.log_param('model_type', 'LightGBM')
    mlflow.log_param('n_features', len(FEATURES))
    mlflow.log_param('train_rows', len(X_train))

    mlflow.log_metric('test_mae',     lgb_mae)
    mlflow.log_metric('test_rmse',    lgb_rmse)
    mlflow.log_metric('test_r2',      lgb_r2)
    mlflow.log_metric('cv_mae',       abs(grid_lgb.best_score_))
    mlflow.log_metric('baseline_mae', baseline_mae)
    mlflow.log_metric('training_sec', elapsed)

    mlflow.set_tag('target',  TARGET)
    mlflow.set_tag('project', 'WATT Energy Intelligence')
    mlflow.set_tag('stage',   'production')

    mlflow.lightgbm.log_model(
    grid_lgb.best_estimator_,
    artifact_path = 'lgb_demand_forecaster',
    input_example = X_test.iloc[:5])


    lgb_run_id = run.info.run_id
    print(f'\n✓ LightGBM run logged: {lgb_run_id}')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Section 9. Pick Winner & Register in Model Registry

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE SCHEMA IF NOT EXISTS watt.models
# MAGIC

# COMMAND ----------

# Compare — same results table from exploration notebook
results = pd.DataFrame({
    'Model'   : ['Baseline', 'XGBoost', 'LightGBM'],
    'MAE'     : [baseline_mae, xgb_mae, lgb_mae],
    'RMSE'    : [baseline_rmse, xgb_rmse, lgb_rmse],
    'R²'      : [r2_score(Y_test, baseline_preds), xgb_r2, lgb_r2],
}).set_index('Model').round(2)

print('\n── FINAL MODEL COMPARISON ──')
print(results.to_string())

# Determine winner by lowest test MAE
if xgb_mae <= lgb_mae:
    winner_name, winner_grid, winner_run_id, winner_preds = 'XGBoost', grid_xgb, xgb_run_id, xgb_preds
else:
    winner_name, winner_grid, winner_run_id, winner_preds = 'LightGBM', grid_lgb, lgb_run_id, lgb_preds

print(f'\n Winner: {winner_name}')

# Model Registry requires s3:PutObject on UC storage — not available on Free Edition.
# The model is already fully saved in the MLflow run artifact (log_model above).
# To load it later: mlflow.xgboost.load_model(f'runs:/{winner_run_id}/xgboost_demand_forecaster')
model_uri = f'runs:/{winner_run_id}/{"xgboost" if winner_name == "XGBoost" else "lgb"}_demand_forecaster'
print(f'  Model URI : {model_uri}')
print(f'  XGBoost run  : {xgb_run_id}')
print(f'  LightGBM run : {lgb_run_id}')


# COMMAND ----------

# MAGIC %md
# MAGIC ## Section 10. SHAP Feature Importance

# COMMAND ----------

best_model = winner_grid.best_estimator_

# SHAP TreeExplainer — fast for XGBoost and LightGBM
explainer   = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_test)

# Summary bar plot
fig, ax = plt.subplots(figsize=(10, 8))
shap.summary_plot(shap_values, X_test, feature_names=FEATURES,
                  plot_type='bar', show=False)
plt.title(f'SHAP Feature Importance — {winner_name}')
plt.tight_layout()
plt.savefig('/tmp/shap_summary.png', bbox_inches='tight')
plt.show()

# Log SHAP plot as artifact in MLflow
with mlflow.start_run(run_id=winner_run_id):
    mlflow.log_artifact('/tmp/shap_summary.png')
print('✓ SHAP plot logged to MLflow')

# COMMAND ----------

# Beeswarm — shows direction of each feature's impact
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_test, feature_names=FEATURES, show=False)
plt.title(f'SHAP Beeswarm — {winner_name}')
plt.tight_layout()
plt.savefig('/tmp/shap_beeswarm.png', bbox_inches='tight')
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Section 11. Predicted vs Actual

# COMMAND ----------

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Distribution overlay — same as NYC Taxis notebook
bins = 50
sns.histplot(winner_preds,  label='Prediction', stat='count', bins=bins, alpha=0.6, ax=axes[0], color='#64B5F6')
sns.histplot(Y_test.values, label='Actual',     stat='count', bins=bins, alpha=0.6, ax=axes[0], color='#EF5350')
axes[0].set_title(f'{winner_name} — Predicted vs Actual')
axes[0].set_xlabel('demand_mwh')
axes[0].legend()

# Residuals scatter
residuals = Y_test.values - winner_preds
axes[1].scatter(winner_preds, residuals, alpha=0.15, s=4, color='#7986CB')
axes[1].axhline(0, color='#D32F2F', linewidth=1.5, linestyle='--')
axes[1].set_title('Residuals (Actual − Predicted)')
axes[1].set_xlabel('Predicted demand_mwh')
axes[1].set_ylabel('Residual (MWh)')

plt.suptitle(f'{winner_name} — Test Set Performance', fontsize=13)
plt.tight_layout()
plt.savefig('/tmp/predicted_vs_actual.png', bbox_inches='tight')
plt.show()

# COMMAND ----------

print('\n' + '=' * 55)
print('⚡ DEMAND FORECASTER COMPLETE')
print('=' * 55)
print(f'  Winner          : {winner_name}')
print(f'  Test MAE        : {min(xgb_mae, lgb_mae):,.0f} MWh')
print(f'  Test R²         : {max(xgb_r2, lgb_r2):.4f}')
print(f'  Baseline MAE    : {baseline_mae:,.0f} MWh')
print(f'  Improvement     : {(1 - min(xgb_mae, lgb_mae)/baseline_mae)*100:.1f}%')
print(f'  MLflow exp      : {EXP_NAME}')
print(f'  Model URI       : {model_uri}')
print()
print('  Next step → 06_anomaly_and_renewable.py')
