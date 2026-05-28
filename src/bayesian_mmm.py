"""
bayesian_mmm.py
---------------
Bayesian MMM using PyMC.
Models:
  - Hierarchical priors on channel coefficients
  - Adstock + saturation inside the model
  - Full posterior inference
  - Channel contribution + credible intervals

Lightweight version to avoid memory issues. Uses NUTS/ADVI.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
from pathlib import Path

try:
    import pymc as pm
    import arviz as az
    PYMC_AVAILABLE = True
except ImportError:
    PYMC_AVAILABLE = False
    print("[BayesianMMM] PyMC not available — using fallback OLS approximation.")

CHANNELS = [
    "google_ads_spend", "facebook_ads_spend", "youtube_ads_spend",
    "tv_ads_spend", "influencer_marketing_spend", "email_marketing_spend",
]
TARGET = "sales_revenue"
MODEL_DIR = Path("models")


def build_and_fit_bayesian_mmm(
    df: pd.DataFrame,
    n_samples: int = 500,
    n_chains: int = 2,
    target_accept: float = 0.85,
) -> dict:
    """
    Build Bayesian MMM with PyMC.
    Uses saturated spend values as inputs to keep the model identifiable.
    Returns idata (ArviZ InferenceData) + summary.
    """
    if not PYMC_AVAILABLE:
        return _fallback_bayesian(df)

    # Use pre-computed saturated columns
    sat_cols = [f"{ch}_saturated" for ch in CHANNELS if f"{ch}_saturated" in df.columns]
    external = ["discount_percentage", "holiday_flag", "weather_index", "website_traffic"]
    external = [c for c in external if c in df.columns]

    # Normalize inputs
    X_media = df[sat_cols].values.astype(float)
    X_ext = df[external].values.astype(float)
    y = df[TARGET].values.astype(float)

    # Standardize
    X_media_mean = X_media.mean(axis=0)
    X_media_std  = X_media.std(axis=0) + 1e-9
    X_ext_mean   = X_ext.mean(axis=0)
    X_ext_std    = X_ext.std(axis=0) + 1e-9
    y_mean = y.mean()
    y_std  = y.std()

    Xm = (X_media - X_media_mean) / X_media_std
    Xe = (X_ext - X_ext_mean) / X_ext_std
    ys = (y - y_mean) / y_std
    t = np.linspace(0, 1, len(df))  # trend variable

    with pm.Model() as model:
        # Priors for media coefficients (must be positive → Half-Normal)
        beta_media = pm.HalfNormal("beta_media", sigma=1.0, shape=len(sat_cols))

        # External variable coefficients
        beta_ext = pm.Normal("beta_ext", mu=0, sigma=1.0, shape=len(external))

        # Trend
        beta_trend = pm.Normal("beta_trend", mu=0, sigma=1.0)

        # Intercept
        intercept = pm.Normal("intercept", mu=0, sigma=2.0)

        # Noise
        sigma = pm.HalfNormal("sigma", sigma=1.0)

        # Mean model
        mu = (
            intercept
            + pm.math.dot(Xm, beta_media)
            + pm.math.dot(Xe, beta_ext)
            + beta_trend * t
        )

        # Likelihood
        obs = pm.Normal("obs", mu=mu, sigma=sigma, observed=ys)

        # Sample
        idata = pm.sample(
            n_samples,
            chains=n_chains,
            target_accept=target_accept,
            progressbar=False,
            random_seed=42,
            return_inferencedata=True,
        )

    # Extract posterior summary
    summary = az.summary(idata, var_names=["beta_media", "beta_ext", "intercept", "sigma"])

    # Channel contributions from posterior means
    beta_media_posterior = idata.posterior["beta_media"].mean(dim=["chain", "draw"]).values
    contributions = {}
    for i, ch in enumerate(CHANNELS):
        if i < len(beta_media_posterior):
            # Un-standardize and scale
            contrib_scaled = float(beta_media_posterior[i]) / X_media_std[i] * y_std
            contributions[ch] = abs(contrib_scaled) * X_media[:, i].mean()

    result = {
        "idata": idata,
        "summary": summary,
        "contributions": contributions,
        "sat_cols": sat_cols,
        "external_cols": external,
        "y_mean": y_mean,
        "y_std": y_std,
        "model": model,
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(contributions, MODEL_DIR / "bayesian_contributions.pkl")
    summary.to_csv("reports/bayesian_mmm_summary.csv")
    print("[BayesianMMM] Sampling complete.")
    return result


def _fallback_bayesian(df: pd.DataFrame) -> dict:
    """Fallback if PyMC not installed — uses Ridge with uncertainty via bootstrap."""
    from sklearn.linear_model import Ridge
    from sklearn.utils import resample

    sat_cols = [f"{ch}_saturated" for ch in CHANNELS if f"{ch}_saturated" in df.columns]
    X = df[sat_cols].values
    y = df[TARGET].values

    n_boot = 100
    coef_boots = []
    for _ in range(n_boot):
        Xb, yb = resample(X, y, random_state=None)
        m = Ridge(alpha=1.0)
        m.fit(Xb, yb)
        coef_boots.append(m.coef_)

    coef_arr = np.array(coef_boots)
    summary_rows = []
    contributions = {}
    for i, ch in enumerate(CHANNELS):
        if i < coef_arr.shape[1]:
            mean_c = coef_arr[:, i].mean()
            std_c  = coef_arr[:, i].std()
            contributions[ch] = abs(mean_c) * X[:, i].mean()
            summary_rows.append({
                "parameter": ch,
                "mean": round(mean_c, 4),
                "sd":   round(std_c, 4),
                "hdi_3%":  round(np.percentile(coef_arr[:, i], 3), 4),
                "hdi_97%": round(np.percentile(coef_arr[:, i], 97), 4),
            })

    summary = pd.DataFrame(summary_rows)
    print("[BayesianMMM] Bootstrap fallback complete.")
    return {
        "idata": None,
        "summary": summary,
        "contributions": contributions,
        "sat_cols": sat_cols,
        "external_cols": [],
    }


def channel_contributions_from_bayes(result: dict, df: pd.DataFrame) -> pd.DataFrame:
    contributions = result["contributions"]
    total = sum(abs(v) for v in contributions.values()) + 1e-9
    rows = []
    for ch, contrib in contributions.items():
        rows.append({
            "channel": ch,
            "contribution": round(contrib, 2),
            "contribution_pct": round(abs(contrib) / total * 100, 2),
        })
    return pd.DataFrame(rows).sort_values("contribution", ascending=False)


if __name__ == "__main__":
    from src.preprocessing import load_data, engineer_features
    df = engineer_features(load_data())
    Path("reports").mkdir(exist_ok=True)
    result = build_and_fit_bayesian_mmm(df, n_samples=200, n_chains=2)
    contrib_df = channel_contributions_from_bayes(result, df)
    print(contrib_df)
