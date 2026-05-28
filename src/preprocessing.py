"""
preprocessing.py
----------------
Full EDA + feature engineering pipeline:
  - Summary statistics
  - Missing value check
  - Correlation analysis
  - VIF analysis
  - Seasonal decomposition
  - Adstock + saturation feature creation
  - Train/test split
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler

from src.adstock import apply_adstock_to_df
from src.saturation import apply_saturation_to_df

CHANNELS = [
    "google_ads_spend", "facebook_ads_spend", "youtube_ads_spend",
    "tv_ads_spend", "influencer_marketing_spend", "email_marketing_spend",
]
EXTERNAL = [
    "discount_percentage", "holiday_flag", "competitor_promotion",
    "inflation_rate", "weather_index", "website_traffic",
]
TARGET = "sales_revenue"


def load_data(path: str = "data/marketing_mmm_data.csv") -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    stats = df.describe().T
    stats["missing"] = df.isnull().sum()
    stats["missing_pct"] = (df.isnull().sum() / len(df) * 100).round(2)
    stats["skew"] = df.skew(numeric_only=True)
    stats["kurt"] = df.kurt(numeric_only=True)
    return stats


def compute_vif(df: pd.DataFrame, features: list) -> pd.DataFrame:
    """Variance Inflation Factor for multicollinearity detection."""
    sub = df[features].dropna()
    vif_data = []
    for i, col in enumerate(features):
        try:
            v = variance_inflation_factor(sub.values, i)
        except Exception:
            v = np.nan
        vif_data.append({"feature": col, "VIF": round(v, 2)})
    return pd.DataFrame(vif_data).sort_values("VIF", ascending=False)


def seasonal_decomposition_result(df: pd.DataFrame, period: int = 7):
    """Decompose sales into trend, seasonal, residual."""
    sales = df.set_index("date")[TARGET].fillna(method="ffill")
    result = seasonal_decompose(sales, model="multiplicative", period=period, extrapolate_trend="freq")
    return result


def acf_pacf_values(df: pd.DataFrame, nlags: int = 40):
    sales = df[TARGET].values
    acf_vals = acf(sales, nlags=nlags)
    pacf_vals = pacf(sales, nlags=nlags)
    return acf_vals, pacf_vals


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Full feature engineering pipeline."""
    df = df.copy()

    # Adstock
    df = apply_adstock_to_df(df)

    # Saturation on adstocked values
    df = apply_saturation_to_df(df, use_adstock=True)

    # Time features
    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    df["quarter"] = df["date"].dt.quarter
    df["t"] = (df["date"] - df["date"].min()).dt.days

    # Cyclical encoding
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # Lag features for target (1, 7, 14, 30 days)
    for lag in [1, 7, 14, 30]:
        df[f"sales_lag_{lag}"] = df[TARGET].shift(lag)

    # Rolling means
    for win in [7, 14, 30]:
        df[f"sales_roll_mean_{win}"] = df[TARGET].shift(1).rolling(win).mean()

    # Log transform spend
    for ch in CHANNELS:
        df[f"{ch}_log"] = np.log1p(df[ch])

    # Interaction: holiday × total spend
    spend_cols = CHANNELS
    df["total_spend"] = df[spend_cols].sum(axis=1)
    df["holiday_spend_interact"] = df["holiday_flag"] * df["total_spend"]

    df = df.dropna().reset_index(drop=True)
    return df


def get_feature_sets(df: pd.DataFrame):
    """Return different feature sets for modeling."""
    adstock_cols = [f"{ch}_adstock" for ch in CHANNELS if f"{ch}_adstock" in df.columns]
    saturated_cols = [f"{ch}_saturated" for ch in CHANNELS if f"{ch}_saturated" in df.columns]
    time_cols = ["t", "dow_sin", "dow_cos", "month_sin", "month_cos", "quarter"]
    external_cols = [c for c in EXTERNAL if c in df.columns]

    feature_sets = {
        "baseline": time_cols + external_cols,
        "with_adstock": adstock_cols + time_cols + external_cols,
        "with_saturation": saturated_cols + time_cols + external_cols,
        "full": adstock_cols + saturated_cols + time_cols + external_cols,
    }
    return feature_sets


def train_test_split_temporal(df: pd.DataFrame, test_ratio: float = 0.2):
    """Temporal split — no data leakage."""
    n = len(df)
    split = int(n * (1 - test_ratio))
    train = df.iloc[:split].copy()
    test = df.iloc[split:].copy()
    return train, test


if __name__ == "__main__":
    df = load_data()
    stats = summary_stats(df)
    print(stats)
    df_feat = engineer_features(df)
    print(f"Engineered features: {df_feat.shape}")
    train, test = train_test_split_temporal(df_feat)
    print(f"Train: {len(train)}, Test: {len(test)}")
