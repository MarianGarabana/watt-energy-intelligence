# ⚡ WATT — Smart Energy Intelligence Platform

> End-to-end data platform for real-time electricity demand forecasting,
> grid anomaly detection, renewable generation intelligence, and carbon intensity scoring.

[![CI/CD](https://github.com/<your-username>/watt-energy-intelligence/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/<your-username>/watt-energy-intelligence/actions)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://python.org)
[![Databricks](https://img.shields.io/badge/Databricks-Delta%20Lake-orange.svg)](https://databricks.com)

**Live Dashboard:** [watt-dashboard.streamlit.app](https://watt-dashboard.streamlit.app) *(coming Day 12)*
**Live API:** [watt-api.onrender.com/docs](https://watt-api.onrender.com/docs) *(coming Day 12)*

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        WATT Platform                            │
│                                                                 │
│  ┌──────────┐    ┌──────────────────────────────────────────┐  │
│  │  Sources  │    │         Databricks Workspace             │  │
│  │          │    │                                          │  │
│  │ EIA API  │───▶│ 🥉 Bronze (raw Delta tables)             │  │
│  │ ENTSO-E  │    │      ↓ PySpark cleaning                  │  │
│  │ Weather  │    │ 🥈 Silver (validated Delta tables)        │  │
│  └──────────┘    │      ↓ PySpark feature engineering        │  │
│                  │ 🥇 Gold (ML-ready feature table)          │  │
│                  │      ↓ MLflow training                    │  │
│                  │ 🤖 4 ML Models in Model Registry          │  │
│                  └──────────────────────────────────────────┘  │
│                              ↓                                  │
│  ┌──────────────┐    ┌──────────────────┐                      │
│  │ FastAPI      │    │ Streamlit         │                      │
│  │ REST API     │    │ Dashboard (6 pgs) │                      │
│  │ /docs        │    │ + Plotly charts   │                      │
│  └──────────────┘    └──────────────────┘                      │
└─────────────────────────────────────────────────────────────────┘
```

## Tech Stack

| Layer | Technologies |
|-------|-------------|
| Data Architecture | Databricks, Delta Lake, Apache Spark (PySpark), Medallion Architecture |
| ML Models | XGBoost, LightGBM, scikit-learn, Isolation Forest, SHAP |
| MLOps | MLflow (Databricks), Evidently AI, Optuna, GitHub Actions |
| API | FastAPI, Pydantic v2, PostgreSQL, SQLAlchemy |
| Dashboard | Streamlit, Plotly, streamlit-extras |
| Infrastructure | Docker, GitHub Actions CI/CD, Render, Streamlit Cloud |

## ML Models

| Model | Problem | Algorithm |
|-------|---------|-----------|
| Demand Forecaster | 24h ahead electricity demand | XGBoost + LightGBM ensemble |
| Anomaly Detector | Grid fault / unusual consumption | Isolation Forest |
| Renewable Forecaster | Solar + wind generation | Gradient Boosting |
| Carbon Intensity Scorer | gCO₂/kWh prediction + green windows | Ridge Regression |

## Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/<your-username>/watt-energy-intelligence.git
cd watt-energy-intelligence

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set up environment variables
cp .env.example .env
# → Edit .env and add your EIA_API_KEY (free at https://www.eia.gov/opendata/)

# 4. Test the API clients locally
python ingestion/eia_client.py
python ingestion/weather_client.py

# 5. Run tests
pytest tests/ -v
```

## Databricks Setup

1. Connect your Databricks workspace to this GitHub repo via **Databricks Repos**
2. Add your API keys to **Databricks Secrets** (`watt` scope):
   ```bash
   databricks secrets create-scope watt
   databricks secrets put --scope watt --key eia_api_key
   ```
3. Run `databricks/notebooks/01_bronze_ingestion.py` to load the first 30 days of data
4. Schedule it as a **Databricks Job** for daily runs

## Project Structure

```
watt-energy-intelligence/
├── databricks/notebooks/   # PySpark notebooks (Bronze → Silver → Gold → ML)
├── ingestion/              # API clients (EIA, ENTSO-E, Open-Meteo)
├── processing/             # Feature engineering utilities
├── ml/                     # Model training + monitoring scripts
├── api/                    # FastAPI application
├── dashboard/              # Streamlit multi-page app
└── tests/                  # pytest test suite
```

## Build Log

| Day | Deliverable | Status |
|-----|-------------|--------|
| 1 | Project setup + Bronze ingestion + API clients | ✅ Done |
| 2 | Silver layer (PySpark cleaning) | 🔄 Next |
| 3 | Gold layer (feature engineering) | ⏳ |
| ... | ... | ... |

---

*Built by [Marian Garabana](https://linkedin.com/in/marian-garabana) · IE University MBDS 2026*
