"""
main.py
-------
Orchestrates the full MMM pipeline:
  1. Data generation
  2. Feature engineering
  3. Model training (all frequentist models)
  4. Forecasting
  5. Budget optimization
  6. SHAP explainability
  7. Save artifacts for dashboard

Run: python main.py
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
from pathlib import Path

# ─── Setup ────────────────────────────────────────────────────────────────────
for d in ["data", "models", "reports"]:
    Path(d).mkdir(exist_ok=True)


def run_pipeline():
    print("=" * 60)
    print("  Agentic MMM Platform — Pipeline")
    print("=" * 60)

    # ── 1. Data Generation ────────────────────────────────────────────────────
    print("\n[1/6] Generating synthetic marketing data...")
    from src.data_generation import generate_marketing_data
    df_raw = generate_marketing_data("data/marketing_mmm_data.csv")
    print(f"      Generated {len(df_raw)} rows x {df_raw.shape[1]} columns")

    # ── 2. Feature Engineering ────────────────────────────────────────────────
    print("\n[2/6] Engineering features (adstock + saturation + time)...")
    from src.preprocessing import (
        engineer_features, train_test_split_temporal,
        get_feature_sets, summary_stats, compute_vif
    )
    df = engineer_features(df_raw)
    print(f"      Feature matrix: {df.shape}")

    stats = summary_stats(df_raw)
    stats.to_csv("reports/summary_statistics.csv")

    channels = [
        "google_ads_spend", "facebook_ads_spend", "youtube_ads_spend",
        "tv_ads_spend", "influencer_marketing_spend", "email_marketing_spend",
    ]
    sat_cols = [f"{ch}_saturated" for ch in channels if f"{ch}_saturated" in df.columns]
    if len(sat_cols) >= 2:
        vif_df = compute_vif(df, sat_cols)
        vif_df.to_csv("reports/vif_analysis.csv", index=False)

    # ── 3. Model Training ─────────────────────────────────────────────────────
    print("\n[3/6] Training MMM models (Linear, Ridge, Lasso, XGBoost)...")
    from src.mmm_model import train_all_models, compute_channel_contributions, compute_roi

    train, test = train_test_split_temporal(df)
    feature_sets = get_feature_sets(df)
    feat_cols = feature_sets["with_saturation"]
    feat_cols = [c for c in feat_cols if c in df.columns]

    results, feat_cols = train_all_models(df, feat_cols, train, test)

    metrics_rows = []
    for name, res in results.items():
        row = {"model": name}
        row.update({f"train_{k}": v for k, v in res["train_metrics"].items()})
        row.update({f"test_{k}": v for k, v in res["test_metrics"].items()})
        metrics_rows.append(row)
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv("reports/model_metrics.csv", index=False)

    best_name = max(results.keys(), key=lambda k: results[k]["test_metrics"]["R2"])
    best_result = results[best_name]
    best_model = best_result["model"]
    best_scaler = best_result["scaler"]
    print(f"      Best model: {best_name} (Test R2={best_result['test_metrics']['R2']:.4f})")

    contributions = compute_channel_contributions(best_model, df, feat_cols, best_scaler)
    roi_df = compute_roi(df, contributions)
    contributions.to_csv("reports/channel_contributions.csv", index=False)
    roi_df.to_csv("reports/channel_roi.csv", index=False)

    joblib.dump(feat_cols, "models/feat_cols.pkl")
    joblib.dump(results, "models/all_results.pkl")

    # ── 4. Forecasting ────────────────────────────────────────────────────────
    print("\n[4/6] Training forecast models (Prophet, ARIMA, XGBoost)...")
    from src.forecasting import fit_prophet, fit_arima, fit_xgboost_forecast

    prophet_result = fit_prophet(df, horizon_days=90)
    arima_result   = fit_arima(df, horizon_days=90)
    xgb_result     = fit_xgboost_forecast(df, horizon_days=90)

    forecast_metrics = pd.DataFrame({
        "Prophet": prophet_result["metrics"],
        "ARIMA":   arima_result["metrics"],
        "XGBoost": xgb_result["metrics"],
    }).T
    forecast_metrics.to_csv("reports/forecast_metrics.csv")
    print(f"      Forecast models complete.")

    joblib.dump(prophet_result, "models/prophet_result.pkl")
    joblib.dump(arima_result,   "models/arima_result.pkl")
    joblib.dump(xgb_result,     "models/xgb_forecast_result.pkl")

    # ── 5. Budget Optimization ────────────────────────────────────────────────
    print("\n[5/6] Running budget optimization...")
    from src.optimization import optimize_budget, allocation_summary

    total_daily_spend = float(df[channels].mean().sum() * 90)
    opt_result = optimize_budget(total_budget=total_daily_spend)
    alloc_summary = allocation_summary(opt_result)
    alloc_summary.to_csv("reports/budget_optimization.csv", index=False)
    print(f"      Revenue uplift: {opt_result['revenue_uplift']:.2f}%")
    joblib.dump(opt_result, "models/opt_result.pkl")

    # ── 6. SHAP Explainability ────────────────────────────────────────────────
    print("\n[6/6] Computing SHAP values...")
    from src.explainability import compute_shap_values

    ridge_result = results.get("Ridge Regression", results[best_name])
    ridge_model  = ridge_result["model"]
    ridge_scaler = ridge_result["scaler"]

    X_shap = df[feat_cols].fillna(0)
    if ridge_scaler is not None:
        X_shap_scaled = pd.DataFrame(
            ridge_scaler.transform(X_shap.values),
            columns=feat_cols,
        )
    else:
        X_shap_scaled = X_shap.copy()

    shap_result = compute_shap_values(ridge_model, X_shap_scaled, model_type="linear")
    joblib.dump(shap_result, "models/shap_result.pkl")
    print(f"      SHAP computed for {len(feat_cols)} features")

    # ── Done ──────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  Data rows: {len(df)}")
    print(f"  Best model: {best_name} | Test R2: {best_result['test_metrics']['R2']:.4f}")
    print(f"  Budget uplift: {opt_result['revenue_uplift']:.2f}%")
    print(f"\n  Run: streamlit run dashboard/app.py")
    print("=" * 60)

    return {
        "df": df, "df_raw": df_raw, "results": results,
        "feat_cols": feat_cols,
        "prophet_result": prophet_result, "xgb_result": xgb_result,
        "opt_result": opt_result, "shap_result": shap_result,
    }


if __name__ == "__main__":
    run_pipeline()
