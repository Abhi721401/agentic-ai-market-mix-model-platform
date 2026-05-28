"""
adstock.py
----------
Adstock transformation utilities.
Supports geometric, Weibull CDF, and delayed adstock.
"""

import numpy as np
import pandas as pd

CHANNELS = [
    "google_ads_spend", "facebook_ads_spend", "youtube_ads_spend",
    "tv_ads_spend", "influencer_marketing_spend", "email_marketing_spend",
]

# Default decay rates calibrated per channel
DEFAULT_DECAYS = {
    "google_ads_spend": 0.4,
    "facebook_ads_spend": 0.3,
    "youtube_ads_spend": 0.5,
    "tv_ads_spend": 0.6,
    "influencer_marketing_spend": 0.25,
    "email_marketing_spend": 0.15,
}


def geometric_adstock(x: np.ndarray, decay: float) -> np.ndarray:
    """Standard geometric adstock: x_t + decay * x_{t-1}^adstock"""
    result = np.zeros_like(x, dtype=float)
    result[0] = float(x[0])
    for t in range(1, len(x)):
        result[t] = float(x[t]) + decay * result[t - 1]
    return result


def weibull_adstock(x: np.ndarray, shape: float = 1.5, scale: float = 5.0, maxlag: int = 21) -> np.ndarray:
    """
    Weibull CDF-based adstock.
    More flexible lag distribution: accounts for delayed peak effects (e.g. TV).
    """
    from scipy.stats import weibull_min
    lags = np.arange(1, maxlag + 1)
    weights = weibull_min.pdf(lags, c=shape, scale=scale)
    weights = weights / (weights.sum() + 1e-9)

    result = np.zeros_like(x, dtype=float)
    for t in range(len(x)):
        for lag_idx, lag in enumerate(lags):
            if t - lag >= 0:
                result[t] += weights[lag_idx] * float(x[t - lag])
        result[t] += float(x[t])  # same-day effect
    return result


def apply_adstock_to_df(
    df: pd.DataFrame,
    channels: list = None,
    decays: dict = None,
    suffix: str = "_adstock",
) -> pd.DataFrame:
    """Apply geometric adstock to all marketing channels. Returns df with new columns."""
    channels = channels or CHANNELS
    decays = decays or DEFAULT_DECAYS
    df = df.copy()
    for ch in channels:
        if ch in df.columns:
            decay = decays.get(ch, 0.3)
            df[f"{ch}{suffix}"] = geometric_adstock(df[ch].values, decay)
    return df


if __name__ == "__main__":
    # Quick test
    x = np.array([100, 200, 50, 0, 0, 300, 100])
    print("Input:          ", x)
    print("Geo adstock 0.4:", np.round(geometric_adstock(x, 0.4), 1))
    print("Weibull adstock:", np.round(weibull_adstock(x), 1))
