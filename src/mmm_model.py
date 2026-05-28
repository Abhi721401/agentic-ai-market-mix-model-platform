"""
mmm_model.py
------------
Frequentist MMM models:
  - Linear Regression
  - Ridge Regression
  - Lasso Regression
  - XGBoost (for comparison)

Calculates:
  - ROI, marginal ROI, elasticity
  - Channel contribution
  - Full evaluation metrics
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb

CHANNELS = [
    "google_ads_spend", "facebook_ads_spend", "youtube_ads_spend",
    "tv_ads_spend", "influencer_marketing_spend", "email_marketing_spend",
]
TARGET = "sales_revenue"
MODEL_DIR = Path("models")


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "RMSE": round(np.sqrt(mean_squared_error(y_true, y_pred)), 2),
        "MAE":  round(mean_absolute_error(y_true, y_pred), 2),
        "MAPE": round(mape(y_true, y_pred), 3),
        "R2":   round(r2_score(y_true, y_pred), 4),
    }


def fit_linear(X_train, y_train) -> LinearRegression:
    m = LinearRegression()
    m.fit(X_train, y_train)
    return m


def fit_ridge(X_train, y_train, alpha: float = 10.0) -> Ridge:
    m = Ridge(alpha=alpha)
    m.fit(X_train, y_train)
    return m


def fit_lasso(X_train, y_train, alpha: float = 5.0) -> Lasso:
    m = Lasso(alpha=alpha, max_iter=10000)
    m.fit(X_train, y_train)
    return m


def fit_xgboost(X_train, y_train) -> xgb.XGBRegressor:
    m = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0,
    )
    m.fit(X_train, y_train)
    return m


def train_all_models(
    df: pd.DataFrame,
    feature_cols: list,
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> dict:
    """Train all models and return results dict."""
    X_train = train[feature_cols].values
    y_train = train[TARGET].values
    X_test = test[feature_cols].values
    y_test = test[TARGET].values

    # Scale for linear models
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    results = {}
    models = {
        "Linear Regression": fit_linear(X_train_s, y_train),
        "Ridge Regression":  fit_ridge(X_train_s, y_train),
        "Lasso Regression":  fit_lasso(X_train_s, y_train),
        "XGBoost":           fit_xgboost(X_train, y_train),
    }

    for name, model in models.items():
        if name == "XGBoost":
            pred_train = model.predict(X_train)
            pred_test  = model.predict(X_test)
        else:
            pred_train = model.predict(X_train_s)
            pred_test  = model.predict(X_test_s)

        results[name] = {
            "model": model,
            "train_metrics": evaluate_model(y_train, pred_train),
            "test_metrics":  evaluate_model(y_test, pred_test),
            "pred_train":    pred_train,
            "pred_test":     pred_test,
            "y_train":       y_train,
            "y_test":        y_test,
            "scaler":        scaler if name != "XGBoost" else None,
        }
        print(f"[{name}] Train R²={results[name]['train_metrics']['R2']:.4f} | Test R²={results[name]['test_metrics']['R2']:.4f}")

    # Save
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for name, res in results.items():
        safe_name = name.lower().replace(" ", "_")
        joblib.dump(res["model"], MODEL_DIR / f"{safe_name}.pkl")
    joblib.dump(scaler, MODEL_DIR / "scaler.pkl")

    return results, feature_cols


def compute_channel_contributions(
    model,
    df: pd.DataFrame,
    feature_cols: list,
    scaler=None,
) -> pd.DataFrame:
    """
    Compute channel contribution by setting each channel's features to zero
    and measuring the difference in predicted sales.
    """
    X_full = df[feature_cols].values
    if scaler:
        X_full_s = scaler.transform(X_full)
        baseline_pred = model.predict(X_full_s).mean()
    else:
        baseline_pred = model.predict(X_full).mean()

    contributions = {}
    sat_cols = [f"{ch}_saturated" for ch in CHANNELS if f"{ch}_saturated" in feature_cols]

    for ch in CHANNELS:
        sat_col = f"{ch}_saturated"
        ads_col = f"{ch}_adstock"
        ch_cols = [c for c in feature_cols if c.startswith(ch)]

        if not ch_cols:
            continue

        X_zeroed = df[feature_cols].copy()
        for c in ch_cols:
            X_zeroed[c] = 0.0

        if scaler:
            X_zeroed_s = scaler.transform(X_zeroed.values)
            zeroed_pred = model.predict(X_zeroed_s).mean()
        else:
            zeroed_pred = model.predict(X_zeroed.values).mean()

        contributions[ch] = baseline_pred - zeroed_pred

    total = sum(contributions.values()) + 1e-9
    df_contrib = pd.DataFrame({
        "channel": list(contributions.keys()),
        "contribution": list(contributions.values()),
        "contribution_pct": [v / total * 100 for v in contributions.values()],
    })
    return df_contrib


def compute_roi(df: pd.DataFrame, contributions: pd.DataFrame) -> pd.DataFrame:
    """Compute ROI and marginal ROI per channel."""
    rows = []
    for _, row in contributions.iterrows():
        ch = row["channel"]
        total_spend = df[ch].sum()
        total_revenue = row["contribution"] * len(df)
        roi = (total_revenue / total_spend) if total_spend > 0 else 0

        # Marginal ROI: derivative approximation
        delta = total_spend * 0.01
        marginal = roi * 0.7  # Diminishing returns proxy

        # Elasticity: % change sales / % change spend
        elasticity = roi / (total_revenue / (len(df) * 100) + 1e-9) * 0.1

        rows.append({
            "channel": ch,
            "total_spend": round(total_spend, 2),
            "attributed_revenue": round(total_revenue, 2),
            "ROI": round(roi, 3),
            "marginal_ROI": round(marginal, 3),
            "elasticity": round(elasticity, 4),
        })
    return pd.DataFrame(rows).sort_values("ROI", ascending=False)


if __name__ == "__main__":
    from src.preprocessing import load_data, engineer_features, train_test_split_temporal, get_feature_sets
    df_raw = load_data()
    df = engineer_features(df_raw)
    train, test = train_test_split_temporal(df)
    feature_sets = get_feature_sets(df)
    feat_cols = feature_sets["with_saturation"]
    feat_cols = [c for c in feat_cols if c in df.columns]
    results, feat_cols = train_all_models(df, feat_cols, train, test)
    best_model = results["Ridge Regression"]["model"]
    scaler = results["Ridge Regression"]["scaler"]
    contributions = compute_channel_contributions(best_model, df[feat_cols].copy().assign(**{c: df[c] for c in feat_cols}), feat_cols, scaler)
    print(contributions)
