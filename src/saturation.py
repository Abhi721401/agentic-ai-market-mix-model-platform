"""
saturation.py
-------------
Saturation transformations:
  - Hill (power) saturation
  - Logistic saturation
  - Negative exponential saturation

Used after adstock to model diminishing returns.
"""

import numpy as np
import pandas as pd

CHANNELS = [
    "google_ads_spend", "facebook_ads_spend", "youtube_ads_spend",
    "tv_ads_spend", "influencer_marketing_spend", "email_marketing_spend",
]

# (alpha, gamma) per channel — calibrated for Hill saturation
DEFAULT_SAT_PARAMS = {
    "google_ads_spend":           {"alpha": 0.6,  "gamma": 2000},
    "facebook_ads_spend":         {"alpha": 0.55, "gamma": 1800},
    "youtube_ads_spend":          {"alpha": 0.65, "gamma": 1500},
    "tv_ads_spend":               {"alpha": 0.5,  "gamma": 3000},
    "influencer_marketing_spend": {"alpha": 0.7,  "gamma": 1000},
    "email_marketing_spend":      {"alpha": 0.8,  "gamma": 500},
}


def hill_saturation(x: np.ndarray, alpha: float, gamma: float) -> np.ndarray:
    """
    Hill (power) saturation.
    f(x) = x^alpha / (x^alpha + gamma^alpha)
    Output ∈ (0, 1).
    """
    x = np.clip(x, 0, None).astype(float)
    numer = np.power(x, alpha)
    denom = numer + np.power(gamma, alpha) + 1e-9
    return numer / denom


def logistic_saturation(x: np.ndarray, lam: float = 0.001) -> np.ndarray:
    """
    Logistic saturation.
    f(x) = 1 - exp(-lam * x)
    Output ∈ (0, 1).
    """
    x = np.clip(x, 0, None).astype(float)
    return 1.0 - np.exp(-lam * x)


def neg_exp_saturation(x: np.ndarray, rate: float = 0.0005) -> np.ndarray:
    """Negative exponential: f(x) = 1 - exp(-rate * x)"""
    return logistic_saturation(x, lam=rate)


def apply_saturation_to_df(
    df: pd.DataFrame,
    channels: list = None,
    params: dict = None,
    use_adstock: bool = True,
    suffix: str = "_saturated",
) -> pd.DataFrame:
    """
    Apply Hill saturation to channels (optionally to adstock versions).
    Returns df with new columns.
    """
    channels = channels or CHANNELS
    params = params or DEFAULT_SAT_PARAMS
    df = df.copy()
    for ch in channels:
        col = f"{ch}_adstock" if use_adstock and f"{ch}_adstock" in df.columns else ch
        if col in df.columns:
            p = params.get(ch, {"alpha": 0.6, "gamma": 1500})
            df[f"{ch}{suffix}"] = hill_saturation(df[col].values, p["alpha"], p["gamma"])
    return df


if __name__ == "__main__":
    x = np.array([0, 100, 500, 1000, 2000, 5000, 10000], dtype=float)
    print("Input:      ", x)
    print("Hill(0.6,1500):", np.round(hill_saturation(x, 0.6, 1500), 4))
    print("Logistic:   ", np.round(logistic_saturation(x, lam=0.001), 4))
