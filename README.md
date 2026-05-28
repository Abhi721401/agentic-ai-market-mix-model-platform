# 📊 Agentic AI-Powered Market Mix Modeling (MMM) Platform

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30-red.svg)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> A production-quality, **100% free-to-build** Marketing Mix Modeling platform combining econometrics, Bayesian inference, forecasting, budget optimization, explainable AI, and LLM agents — all in one Streamlit dashboard.

---

## 🏗️ Architecture

```
Data Generation → Feature Engineering → MMM Models → Forecasting
      ↓                 ↓                   ↓              ↓
  Synthetic         Adstock +          Ridge/Lasso/    Prophet /
  3-year data       Saturation         XGBoost/Bayes   ARIMA / XGB
                                           ↓
                                   Budget Optimizer (scipy)
                                           ↓
                                   SHAP Explainability
                                           ↓
                                   Groq AI Agents
                                           ↓
                                   Streamlit Dashboard
```

## ✨ Features

| Module | What it does |
|--------|-------------|
| `data_generation.py` | 3 years synthetic marketing data with adstock + saturation baked in |
| `adstock.py` | Geometric + Weibull adstock transformations |
| `saturation.py` | Hill (power) + logistic saturation curves |
| `preprocessing.py` | EDA, VIF, seasonal decomposition, feature engineering |
| `mmm_model.py` | Linear / Ridge / Lasso / XGBoost with ROI + elasticity |
| `bayesian_mmm.py` | PyMC Bayesian MMM with credible intervals |
| `forecasting.py` | Prophet + ARIMA + XGBoost + scenario simulator |
| `optimization.py` | scipy-based budget allocator (SLSQP) |
| `explainability.py` | SHAP feature importance, summary + waterfall plots |
| `agents.py` | 5 Groq-powered AI agents (Analyst/Forecast/Opt/Report/Chat) |
| `groq_utils.py` | Token-efficient Groq API wrapper with rule-based fallback |
| `dashboard/app.py` | Full Streamlit dashboard (6 pages) |

## 🆓 Free to Build

All tools used are **100% free / open source**:
- Python, Pandas, NumPy, Scikit-learn, Statsmodels — free
- PyMC (Bayesian) — free
- Prophet (Meta) — free
- XGBoost — free
- SHAP — free
- Plotly, Streamlit — free
- Groq API — **free tier** (llama3-8b-8192, 14,400 req/day free)
  - Get key: https://console.groq.com → no credit card required

## 🚀 Installation

```bash
# 1. Clone or create project
git clone <your-repo> mmm-platform
cd mmm-platform

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate.bat     # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up environment
cp .env.example .env
# Edit .env and add your free Groq API key
# GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxx

# 5. Run full pipeline
python main.py

# 6. Launch dashboard
streamlit run dashboard/app.py
```

## 📖 Usage

### Run just data generation
```python
from src.data_generation import generate_marketing_data
df = generate_marketing_data("data/marketing_mmm_data.csv")
```

### Run just optimization
```python
from src.optimization import optimize_budget, allocation_summary
result = optimize_budget(total_budget=500_000)
print(allocation_summary(result))
```

### Use AI agents
```python
from src.agents import chat_agent
response = chat_agent("Which channel has the highest ROI?")
print(response)
```

### Run Bayesian MMM
```python
from src.bayesian_mmm import build_and_fit_bayesian_mmm
from src.preprocessing import load_data, engineer_features
df = engineer_features(load_data())
result = build_and_fit_bayesian_mmm(df, n_samples=500, n_chains=2)
```

## 📊 Dashboard Pages

1. **Executive Overview** — KPIs, revenue trend, spend mix pie, scatter analysis
2. **Marketing Analytics** — ROI bars, attribution donut, correlation heatmap, saturation curves
3. **Forecasting** — Prophet/ARIMA/XGBoost comparison + 90-day forecast + scenario simulator
4. **Optimization** — Budget allocation optimizer with response curves and allocation table
5. **Explainability** — SHAP feature importance, summary plot, waterfall, sales drivers
6. **AI Assistant** — 5 AI agent buttons + free-form chat powered by Groq llama3-8b-8192

## 📁 Output Files

After `python main.py`:

```
data/
  marketing_mmm_data.csv       ← 1,096 rows synthetic data

models/
  ridge_regression.pkl
  lasso_regression.pkl
  linear_regression.pkl
  xgboost.pkl
  prophet_model.pkl
  arima_model.pkl
  xgboost_forecast.pkl
  scaler.pkl
  feat_cols.pkl
  all_results.pkl
  opt_result.pkl
  shap_result.pkl

reports/
  summary_statistics.csv
  vif_analysis.csv
  model_metrics.csv
  channel_contributions.csv
  channel_roi.csv
  forecast_metrics.csv
  budget_optimization.csv
  bayesian_mmm_summary.csv     ← if PyMC installed
```

## ☁️ Deployment

### Streamlit Cloud (Free)
```bash
# 1. Push to GitHub
git push origin main

# 2. Go to https://share.streamlit.io
# 3. Connect repo → set main file: dashboard/app.py
# 4. Add GROQ_API_KEY in Secrets
```

### Hugging Face Spaces (Free)
```bash
# Create space → Streamlit SDK
# Push code, add GROQ_API_KEY to Space secrets
# Set startup command: streamlit run dashboard/app.py
```

### Local Docker
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
RUN python main.py
CMD ["streamlit", "run", "dashboard/app.py", "--server.port=8501"]
```

## 🔬 Business Impact

- **ROI measurement**: Identify which channels generate the most revenue per dollar
- **Budget efficiency**: Typical uplift of 10–20% with optimized allocation
- **Forecasting**: 90-day sales projection with confidence intervals
- **Scenario planning**: Instant "what-if" analysis for budget decisions
- **Explainability**: Board-ready SHAP visualizations of sales drivers
- **AI-powered**: Natural language Q&A for non-technical stakeholders

## 🔮 Future Improvements

- [ ] Real data ingestion (Google Ads API, Meta Ads API, Shopify)
- [ ] Time-varying adstock (Nevergrad / NUTS)
- [ ] Bayesian hyperparameter optimization (Optuna)
- [ ] Multi-objective optimization (budget vs reach vs frequency)
- [ ] Automated anomaly detection with email alerts
- [ ] A/B test lift measurement module
- [ ] CrewAI multi-agent pipeline for automated reporting
- [ ] Geo-level MMM (hierarchical model per region)
- [ ] Real-time streaming dashboard (Streamlit + WebSocket)

## 📦 Tech Stack

```
Statistics / Econometrics: statsmodels, scipy, numpy
ML / Forecasting:          scikit-learn, xgboost, prophet
Bayesian Inference:        pymc, arviz
Explainability:            shap
Optimization:              scipy.optimize (SLSQP)
LLM / Agents:              groq (llama3-8b-8192, free), langchain-groq
Visualization:             plotly, streamlit
Data:                      pandas, numpy
```

---

*Built for AI/ML portfolios, graduate research, startup prototypes, and OpenAI Residency applications.*
