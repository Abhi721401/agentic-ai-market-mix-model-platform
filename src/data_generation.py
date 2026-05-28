"""
data_generation.py
------------------
Generates 3 years of realistic synthetic marketing mix data with:
- Trend, yearly + weekly seasonality
- Lag/carryover effects per channel
- Nonlinear saturation (Hill function)
- External factors: holiday, competitor, inflation, weather, traffic
- Controlled noise for statistical realism
"""

import numpy as np
import pandas as pd
from pathlib import Path

# ─── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42
rng = np.random.default_rng(SEED)

# ─── Config ───────────────────────────────────────────────────────────────────
START_DATE = "2021-01-01"
N_DAYS = 365 * 3  # 3 years

CHANNEL_CONFIG = {
    "google_ads_spend": {
        "base_mean": 1500, "base_std": 400,
        "roi_true": 3.5, "adstock_decay": 0.4, "sat_alpha": 0.6, "sat_gamma": 2000,
    },
    "facebook_ads_spend": {
        "base_mean": 1200, "base_std": 350,
        "roi_true": 2.8, "adstock_decay": 0.3, "sat_alpha": 0.55, "sat_gamma": 1800,
    },
    "youtube_ads_spend": {
        "base_mean": 800, "base_std": 300,
        "roi_true": 3.0, "adstock_decay": 0.5, "sat_alpha": 0.65, "sat_gamma": 1500,
    },
    "tv_ads_spend": {
        "base_mean": 2000, "base_std": 600,
        "roi_true": 2.2, "adstock_decay": 0.6, "sat_alpha": 0.5, "sat_gamma": 3000,
    },
    "influencer_marketing_spend": {
        "base_mean": 600, "base_std": 200,
        "roi_true": 4.0, "adstock_decay": 0.25, "sat_alpha": 0.7, "sat_gamma": 1000,
    },
    "email_marketing_spend": {
        "base_mean": 300, "base_std": 100,
        "roi_true": 5.0, "adstock_decay": 0.15, "sat_alpha": 0.8, "sat_gamma": 500,
    },
}

CHANNELS = list(CHANNEL_CONFIG.keys())


# ─── Helpers ──────────────────────────────────────────────────────────────────

def hill_saturation(x: np.ndarray, alpha: float, gamma: float) -> np.ndarray:
    """Hill function saturation: x^alpha / (x^alpha + gamma^alpha)"""
    x_safe = np.where(x < 0, 0, x)
    return (x_safe ** alpha) / (x_safe ** alpha + gamma ** alpha + 1e-9)


def adstock_transform(x: np.ndarray, decay: float) -> np.ndarray:
    """Geometric adstock with decay rate `decay` ∈ (0,1)."""
    adstocked = np.zeros_like(x, dtype=float)
    adstocked[0] = x[0]
    for t in range(1, len(x)):
        adstocked[t] = x[t] + decay * adstocked[t - 1]
    return adstocked


def generate_spend(cfg: dict, n: int, dates: pd.DatetimeIndex) -> np.ndarray:
    """Generate channel spend with weekly spikes and campaign bursts."""
    base = rng.normal(cfg["base_mean"], cfg["base_std"], n).clip(0)

    # Campaign bursts: ~8 per year
    n_bursts = int(n / 365 * 8)
    burst_days = rng.integers(0, n, n_bursts)
    for bd in burst_days:
        duration = rng.integers(7, 21)
        end = min(bd + duration, n)
        base[bd:end] *= rng.uniform(1.5, 2.5)

    # Zero spend on some weekends (realistic)
    weekend_mask = dates.dayofweek >= 5
    zero_weekend = rng.random(n) < 0.3
    base[weekend_mask & zero_weekend] = 0

    return base.clip(0)


# ─── Main Generator ───────────────────────────────────────────────────────────

def generate_marketing_data(save_path: str = "data/marketing_mmm_data.csv") -> pd.DataFrame:
    dates = pd.date_range(START_DATE, periods=N_DAYS, freq="D")
    df = pd.DataFrame({"date": dates})
    t = np.arange(N_DAYS)

    # ── External / Control Variables ──────────────────────────────────────────
    # Yearly seasonality (stronger in Nov-Dec)
    yearly_season = (
        3000 * np.sin(2 * np.pi * t / 365 - np.pi / 2) +
        2000 * np.cos(4 * np.pi * t / 365)
    )
    # Weekly seasonality (higher mid-week for B2B, higher weekend for B2C mix)
    weekly_season = 800 * np.sin(2 * np.pi * np.array(dates.dayofweek) / 7)

    # Trend: gentle linear + slight acceleration
    trend = 5000 + 1.8 * t + 0.0003 * t ** 2

    # Holiday flag: major US/retail holidays
    holiday_dates = pd.to_datetime([
        "2021-01-01","2021-07-04","2021-11-25","2021-12-24","2021-12-25","2021-12-31",
        "2022-01-01","2022-07-04","2022-11-24","2022-12-24","2022-12-25","2022-12-31",
        "2023-01-01","2023-07-04","2023-11-23","2023-12-24","2023-12-25","2023-12-31",
        # Black Friday windows
        "2021-11-26","2021-11-27","2021-11-28","2021-11-29",
        "2022-11-25","2022-11-26","2022-11-27","2022-11-28",
        "2023-11-24","2023-11-25","2023-11-26","2023-11-27",
    ])
    holiday_flag = dates.isin(holiday_dates).astype(int)

    # Competitor promotions (random bursts)
    competitor_promotion = np.zeros(N_DAYS)
    comp_starts = rng.integers(0, N_DAYS, 20)
    for cs in comp_starts:
        end = min(cs + rng.integers(5, 15), N_DAYS)
        competitor_promotion[cs:end] = 1

    # Discount percentage (0–40%)
    discount_pct = rng.uniform(0, 40, N_DAYS)
    # Promotional spikes
    disc_spikes = rng.integers(0, N_DAYS, 30)
    for ds in disc_spikes:
        end = min(ds + rng.integers(3, 10), N_DAYS)
        discount_pct[ds:end] = rng.uniform(25, 45)
    discount_pct = discount_pct.clip(0, 45)

    # Inflation rate (monthly, interpolated)
    monthly_inflation = np.linspace(2.0, 4.5, N_DAYS // 30 + 2)
    inflation_rate = np.interp(t, np.linspace(0, N_DAYS, len(monthly_inflation)), monthly_inflation)
    inflation_rate += rng.normal(0, 0.1, N_DAYS)

    # Weather index (-1 to 1, correlated with season)
    weather_index = (
        np.sin(2 * np.pi * t / 365) * 0.6 +
        rng.normal(0, 0.2, N_DAYS)
    ).clip(-1, 1)

    # Website traffic (correlated with ads + seasonality)
    base_traffic = 5000 + 1.5 * t + 1500 * np.sin(2 * np.pi * t / 365)
    website_traffic = (base_traffic + rng.normal(0, 800, N_DAYS)).clip(1000)

    # ── Spend Generation ──────────────────────────────────────────────────────
    raw_spends = {}
    for ch, cfg in CHANNEL_CONFIG.items():
        raw_spends[ch] = generate_spend(cfg, N_DAYS, dates)
        df[ch] = raw_spends[ch]

    # ── Sales Revenue Construction ────────────────────────────────────────────
    # Base sales — force numpy arrays
    trend = np.array(trend, dtype=float)
    yearly_season = np.array(yearly_season, dtype=float)
    weekly_season = np.array(weekly_season, dtype=float)
    sales = trend + yearly_season + weekly_season

    # Channel contributions via adstock + saturation
    for ch, cfg in CHANNEL_CONFIG.items():
        adstocked = adstock_transform(raw_spends[ch], cfg["adstock_decay"])
        saturated = hill_saturation(adstocked, cfg["sat_alpha"], cfg["sat_gamma"])
        # Scale: saturated output is (0,1) → multiply by max contribution
        contribution = saturated * cfg["roi_true"] * cfg["base_mean"] * 1.5
        sales += contribution

    # External variable effects
    sales += holiday_flag * rng.uniform(3000, 8000, N_DAYS)
    sales += discount_pct * rng.uniform(100, 200, N_DAYS)
    sales -= competitor_promotion * rng.uniform(500, 2000, N_DAYS)
    sales -= inflation_rate * rng.uniform(200, 500, N_DAYS)
    sales += weather_index * rng.uniform(500, 1500, N_DAYS)
    sales += (website_traffic / 10000) * rng.uniform(1000, 3000, N_DAYS)

    # Multiplicative noise
    noise = rng.normal(1.0, 0.05, N_DAYS)
    sales = (sales * noise).clip(5000)

    # ── Assemble DataFrame ────────────────────────────────────────────────────
    df["discount_percentage"] = np.round(discount_pct, 2)
    df["holiday_flag"] = holiday_flag
    df["competitor_promotion"] = competitor_promotion.astype(int)
    df["inflation_rate"] = np.round(inflation_rate, 3)
    df["weather_index"] = np.round(weather_index, 4)
    df["website_traffic"] = website_traffic.astype(int)
    df["sales_revenue"] = np.round(sales, 2)

    # Save
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"[DataGen] Saved {len(df)} rows → {save_path}")
    return df


if __name__ == "__main__":
    df = generate_marketing_data()
    print(df.describe())
