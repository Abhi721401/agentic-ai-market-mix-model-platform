"""
optimization.py
---------------
Budget optimization engine using scipy.optimize.
Objective: maximize expected sales under total budget constraint.

Features:
  - Min/max spend constraints per channel
  - Revenue lift estimation
  - Current vs optimized allocation
  - Response curve visualization
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.optimize import minimize, differential_evolution
from typing import Dict, Optional, Tuple

from src.saturation import hill_saturation
from src.adstock import DEFAULT_DECAYS

CHANNELS = [
    "google_ads_spend", "facebook_ads_spend", "youtube_ads_spend",
    "tv_ads_spend", "influencer_marketing_spend", "email_marketing_spend",
]

# ROI and saturation params (calibrated to data generation assumptions)
CHANNEL_ROI = {
    "google_ads_spend":           3.5,
    "facebook_ads_spend":         2.8,
    "youtube_ads_spend":          3.0,
    "tv_ads_spend":               2.2,
    "influencer_marketing_spend": 4.0,
    "email_marketing_spend":      5.0,
}

SAT_PARAMS = {
    "google_ads_spend":           {"alpha": 0.6,  "gamma": 2000},
    "facebook_ads_spend":         {"alpha": 0.55, "gamma": 1800},
    "youtube_ads_spend":          {"alpha": 0.65, "gamma": 1500},
    "tv_ads_spend":               {"alpha": 0.5,  "gamma": 3000},
    "influencer_marketing_spend": {"alpha": 0.7,  "gamma": 1000},
    "email_marketing_spend":      {"alpha": 0.8,  "gamma": 500},
}

# Default budget (daily average spend * 90 days)
DEFAULT_CURRENT_BUDGET = {
    "google_ads_spend": 135_000,
    "facebook_ads_spend": 108_000,
    "youtube_ads_spend": 72_000,
    "tv_ads_spend": 180_000,
    "influencer_marketing_spend": 54_000,
    "email_marketing_spend": 27_000,
}


def channel_revenue(spend: float, channel: str) -> float:
    """
    Expected revenue from a channel given spend.
    Uses Hill saturation + ROI scaling.
    """
    p = SAT_PARAMS[channel]
    sat = hill_saturation(np.array([spend]), p["alpha"], p["gamma"])[0]
    roi = CHANNEL_ROI[channel]
    # Scale: max possible daily revenue = roi * mean_spend * days
    base_mean = spend / 90 if spend > 0 else 1000
    return sat * roi * max(spend, 1)


def total_revenue(allocations: np.ndarray) -> float:
    """Total expected revenue across all channels."""
    rev = 0.0
    for i, ch in enumerate(CHANNELS):
        rev += channel_revenue(allocations[i], ch)
    return rev


def negative_revenue(allocations: np.ndarray) -> float:
    """Negative total revenue for minimization."""
    return -total_revenue(allocations)


def optimize_budget(
    total_budget: float,
    current_allocation: Optional[Dict[str, float]] = None,
    min_spend_pct: float = 0.05,
    max_spend_pct: float = 0.50,
) -> dict:
    """
    Optimize budget allocation to maximize expected revenue.

    Args:
        total_budget: Total marketing budget
        current_allocation: Current spend per channel (dict)
        min_spend_pct: Minimum fraction of total budget per channel
        max_spend_pct: Maximum fraction of total budget per channel

    Returns:
        dict with optimized allocation, revenue uplift, metrics
    """
    current_allocation = current_allocation or DEFAULT_CURRENT_BUDGET.copy()

    # Adjust current allocation to match total_budget
    current_total = sum(current_allocation.values())
    scaling = total_budget / current_total
    current_scaled = {ch: v * scaling for ch, v in current_allocation.items()}

    # Bounds
    min_spend = total_budget * min_spend_pct
    max_spend = total_budget * max_spend_pct
    bounds = [(min_spend, max_spend)] * len(CHANNELS)

    # Budget constraint
    constraints = [
        {"type": "eq", "fun": lambda x: np.sum(x) - total_budget}
    ]

    # Initial guess: proportional to ROI
    rois = np.array([CHANNEL_ROI[ch] for ch in CHANNELS])
    x0 = total_budget * (rois / rois.sum())
    x0 = np.clip(x0, min_spend, max_spend)
    x0 = x0 / x0.sum() * total_budget  # re-normalize

    # Optimize
    result = minimize(
        negative_revenue,
        x0=x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-9, "maxiter": 1000},
    )

    optimized_alloc = dict(zip(CHANNELS, result.x))
    baseline_revenue = total_revenue(np.array([current_scaled[ch] for ch in CHANNELS]))
    optimized_revenue = total_revenue(result.x)
    uplift = (optimized_revenue - baseline_revenue) / (baseline_revenue + 1e-9) * 100

    return {
        "current_allocation": current_scaled,
        "optimized_allocation": optimized_alloc,
        "total_budget": total_budget,
        "baseline_revenue": round(baseline_revenue, 2),
        "optimized_revenue": round(optimized_revenue, 2),
        "revenue_uplift": round(uplift, 3),
        "revenue_lift_abs": round(optimized_revenue - baseline_revenue, 2),
        "success": result.success,
        "channels": CHANNELS,
    }


def build_response_curves(
    channel: str,
    budget_range: Tuple[float, float] = (0, 200_000),
    n_points: int = 100,
) -> pd.DataFrame:
    """Build response curve for a single channel."""
    spends = np.linspace(budget_range[0], budget_range[1], n_points)
    revenues = [channel_revenue(s, channel) for s in spends]
    marginal = np.gradient(revenues, spends)
    return pd.DataFrame({
        "spend": spends,
        "revenue": revenues,
        "marginal_roi": marginal,
    })


def allocation_summary(opt_result: dict) -> pd.DataFrame:
    """Create a summary DataFrame for visualization."""
    rows = []
    for ch in CHANNELS:
        current_s = opt_result["current_allocation"].get(ch, 0)
        optimized_s = opt_result["optimized_allocation"].get(ch, 0)
        ch_label = ch.replace("_spend", "").replace("_", " ").title()
        rows.append({
            "channel": ch_label,
            "channel_raw": ch,
            "current_spend": round(current_s, 2),
            "optimized_spend": round(optimized_s, 2),
            "change": round(optimized_s - current_s, 2),
            "change_pct": round((optimized_s - current_s) / (current_s + 1e-9) * 100, 2),
            "current_revenue": round(channel_revenue(current_s, ch), 2),
            "optimized_revenue": round(channel_revenue(optimized_s, ch), 2),
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    result = optimize_budget(total_budget=576_000)
    print(f"\nRevenue uplift: {result['revenue_uplift']:.2f}%")
    summary = allocation_summary(result)
    print(summary[["channel", "current_spend", "optimized_spend", "change_pct"]].to_string())
