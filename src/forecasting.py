"""
forecasting.py
--------------
Multi-model sales forecasting:
  - Prophet (trend + seasonality)
  - ARIMA (statsmodels)
  - XGBoost (feature-based)

Also includes:
  - Scenario simulator: "what if Google Ads +20%?"
  - Future ROI + channel performance projection
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    print("[Forecasting] Prophet not available.")

try:
    from statsmodels.tsa.arima.model import ARIMA
    ARIMA_AVAILABLE = True
except ImportError:
    ARIMA_AVAILABLE = False

TARGET = "sales_revenue"
CHANNELS = [
    "google_ads_spend", "facebook_ads_spend", "youtube_ads_spend",
    "tv_ads_spend", "influencer_marketing_spend", "email_marketing_spend",
]
MODEL_DIR = Path("models")


def mape(y_true, y_pred):
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def eval_metrics(y_true, y_pred) -> dict:
    return {
        "RMSE": round(np.sqrt(mean_squared_error(y_true, y_pred)), 2),
        "MAE":  round(mean_absolute_error(y_true, y_pred), 2),
        "MAPE": round(mape(np.array(y_true), np.array(y_pred)), 3),
        "R2":   round(r2_score(y_true, y_pred), 4),
    }


# ─── Prophet ──────────────────────────────────────────────────────────────────

def fit_prophet(df: pd.DataFrame, horizon_days: int = 90) -> dict:
    """Fit Prophet model and forecast horizon_days ahead."""
    if not PROPHET_AVAILABLE:
        return _dummy_forecast(df, horizon_days, "Prophet")

    # Prophet requires ds + y columns; add regressors
    regressors = ["discount_percentage", "holiday_flag", "competitor_promotion", "total_spend"]
    regressors = [r for r in regressors if r in df.columns]

    prophet_df = df[["date", TARGET] + regressors].rename(
        columns={"date": "ds", TARGET: "y"}
    ).copy()

    # Make total_spend if missing
    if "total_spend" not in prophet_df.columns:
        prophet_df["total_spend"] = df[CHANNELS].sum(axis=1).values

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode="multiplicative",
        changepoint_prior_scale=0.1,
        seasonality_prior_scale=10,
        interval_width=0.95,
    )

    for reg in regressors:
        model.add_regressor(reg)

    model.fit(prophet_df)

    # Future dataframe
    future = model.make_future_dataframe(periods=horizon_days)
    # Fill regressors for future (use last-30-day mean as proxy)
    for reg in regressors:
        last_val = prophet_df[reg].tail(30).mean()
        future[reg] = future[reg].fillna(last_val) if reg in future.columns else last_val
        future.loc[future[reg].isna(), reg] = last_val

    forecast = model.predict(future)

    # Eval on historical
    hist_pred = forecast.loc[forecast["ds"].isin(prophet_df["ds"]), "yhat"].values
    y_true = prophet_df["y"].values
    min_len = min(len(y_true), len(hist_pred))
    metrics = eval_metrics(y_true[:min_len], hist_pred[:min_len])

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_DIR / "prophet_model.pkl")

    return {
        "model": model,
        "forecast": forecast,
        "future_forecast": forecast.tail(horizon_days),
        "metrics": metrics,
        "name": "Prophet",
    }


# ─── ARIMA ────────────────────────────────────────────────────────────────────

def fit_arima(df: pd.DataFrame, horizon_days: int = 90, order: tuple = (2, 1, 2)) -> dict:
    """Fit ARIMA model on aggregated weekly data (faster, more stable)."""
    if not ARIMA_AVAILABLE:
        return _dummy_forecast(df, horizon_days, "ARIMA")

    # Use weekly aggregation for speed
    weekly = df.set_index("date")[TARGET].resample("W").sum()
    n_test = max(4, len(weekly) // 5)
    train_s = weekly.iloc[:-n_test]
    test_s  = weekly.iloc[-n_test:]

    try:
        model = ARIMA(train_s, order=order)
        fitted = model.fit()
        test_pred = fitted.forecast(steps=n_test)
        metrics = eval_metrics(test_s.values, test_pred.values)

        # Forecast future (weeks → approximate daily)
        n_future_weeks = max(1, horizon_days // 7)
        future_weekly = fitted.forecast(steps=n_test + n_future_weeks)
        future_pred = np.repeat(future_weekly[-n_future_weeks:] / 7, 7)[:horizon_days]

        last_date = df["date"].max()
        future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon_days)
        future_df = pd.DataFrame({"ds": future_dates, "yhat": future_pred})

        MODEL_DIR.mkdir(exist_ok=True)
        joblib.dump(fitted, MODEL_DIR / "arima_model.pkl")
    except Exception as e:
        print(f"[ARIMA] Fitting error: {e}. Using naive forecast.")
        return _dummy_forecast(df, horizon_days, "ARIMA")

    return {
        "model": fitted,
        "future_forecast": future_df,
        "metrics": metrics,
        "name": "ARIMA",
    }


# ─── XGBoost Forecasting ──────────────────────────────────────────────────────

def fit_xgboost_forecast(df: pd.DataFrame, horizon_days: int = 90) -> dict:
    """Feature-based XGBoost forecasting with recursive prediction."""
    df = df.copy()

    feat_cols = [
        "t", "dow_sin", "dow_cos", "month_sin", "month_cos", "quarter",
        "holiday_flag", "discount_percentage", "weather_index",
        "sales_lag_1", "sales_lag_7", "sales_roll_mean_7", "sales_roll_mean_30",
        "total_spend",
    ]
    feat_cols = [c for c in feat_cols if c in df.columns]

    if "total_spend" not in df.columns:
        df["total_spend"] = df[CHANNELS].sum(axis=1)

    n_test = int(len(df) * 0.2)
    train = df.iloc[:-n_test].dropna(subset=feat_cols + [TARGET])
    test  = df.iloc[-n_test:].dropna(subset=feat_cols + [TARGET])

    X_train, y_train = train[feat_cols].values, train[TARGET].values
    X_test,  y_test  = test[feat_cols].values,  test[TARGET].values

    model = xgb.XGBRegressor(
        n_estimators=400, max_depth=5, learning_rate=0.04,
        subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0,
    )
    model.fit(X_train, y_train)
    test_pred = model.predict(X_test)
    metrics = eval_metrics(y_test, test_pred)

    # Simple future forecast using last-known values as proxy
    last_row = df[feat_cols].iloc[-1].copy()
    last_t = df["t"].iloc[-1]
    last_date = df["date"].max()
    future_preds = []
    future_dates = []

    for i in range(1, horizon_days + 1):
        row = last_row.copy()
        row["t"] = last_t + i
        future_dates.append(last_date + pd.Timedelta(days=i))
        pred = model.predict(row.values.reshape(1, -1))[0]
        future_preds.append(pred)
        # Update lag features
        if "sales_lag_1" in feat_cols:
            row["sales_lag_1"] = pred
        if "sales_lag_7" in feat_cols:
            row["sales_lag_7"] = future_preds[-7] if len(future_preds) >= 7 else pred

    future_df = pd.DataFrame({"ds": future_dates, "yhat": future_preds})

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_DIR / "xgboost_forecast.pkl")

    return {
        "model": model,
        "feat_cols": feat_cols,
        "future_forecast": future_df,
        "metrics": metrics,
        "name": "XGBoost",
        "test_pred": test_pred,
        "y_test": y_test,
    }


# ─── Scenario Simulator ───────────────────────────────────────────────────────

def scenario_simulator(
    df: pd.DataFrame,
    channel_changes: dict,
    model,
    feature_cols: list,
    scaler=None,
) -> dict:
    """
    Simulate: "What if Google Ads spend increases by 20%?"
    channel_changes: e.g. {"google_ads_spend": 1.2, "facebook_ads_spend": 0.8}
    Returns baseline vs scenario sales.
    """
    from src.adstock import apply_adstock_to_df
    from src.saturation import apply_saturation_to_df

    df_scenario = df.copy()

    # Apply multipliers to raw spend
    for ch, mult in channel_changes.items():
        if ch in df_scenario.columns:
            df_scenario[ch] = df_scenario[ch] * mult

    # Recompute adstock + saturation
    df_scenario = apply_adstock_to_df(df_scenario)
    df_scenario = apply_saturation_to_df(df_scenario)

    feat_cols_available = [c for c in feature_cols if c in df_scenario.columns]

    X_base = df[feat_cols_available].fillna(0).values
    X_scen = df_scenario[feat_cols_available].fillna(0).values

    if scaler:
        X_base = scaler.transform(X_base)
        X_scen = scaler.transform(X_scen)

    base_pred = model.predict(X_base)
    scen_pred = model.predict(X_scen)

    uplift = (scen_pred.mean() - base_pred.mean()) / (base_pred.mean() + 1e-9) * 100

    return {
        "baseline_avg": round(base_pred.mean(), 2),
        "scenario_avg": round(scen_pred.mean(), 2),
        "uplift_pct": round(uplift, 3),
        "baseline_total": round(base_pred.sum(), 2),
        "scenario_total": round(scen_pred.sum(), 2),
        "revenue_lift": round(scen_pred.sum() - base_pred.sum(), 2),
        "channel_changes": channel_changes,
    }


def _dummy_forecast(df: pd.DataFrame, horizon_days: int, name: str) -> dict:
    """Naive last-value carry-forward forecast fallback."""
    last_date = df["date"].max()
    last_val = df[TARGET].tail(30).mean()
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon_days)
    future_df = pd.DataFrame({
        "ds": future_dates,
        "yhat": np.random.normal(last_val, last_val * 0.05, horizon_days),
    })
    return {
        "model": None,
        "future_forecast": future_df,
        "metrics": {"RMSE": 0, "MAE": 0, "MAPE": 0, "R2": 0},
        "name": name,
    }


if __name__ == "__main__":
    from src.preprocessing import load_data, engineer_features
    df = engineer_features(load_data())
    result = fit_prophet(df, horizon_days=90)
    print(f"Prophet metrics: {result['metrics']}")
    result_xgb = fit_xgboost_forecast(df, horizon_days=90)
    print(f"XGBoost metrics: {result_xgb['metrics']}")
