"""
dashboard/app.py  —  Agentic AI-Powered Market Mix Modeling Platform
Light theme | All Plotly color bugs fixed | rgba() everywhere
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import warnings; warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import joblib
from pathlib import Path

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MMM Platform | AI-Powered",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Light Theme CSS ───────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* ── Global ── */
  [data-testid="stAppViewContainer"] {
    background: #f8fafc;
    color: #1e293b;
  }
  [data-testid="stSidebar"] {
    background: #ffffff;
    border-right: 1px solid #e2e8f0;
  }
  [data-testid="stSidebar"] * { color: #1e293b !important; }

  /* ── Metric Cards ── */
  .metric-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-top: 4px solid #3b82f6;
    border-radius: 10px;
    padding: 18px 16px;
    text-align: center;
    box-shadow: 0 1px 3px rgba(0,0,0,0.07);
  }
  .metric-value { font-size: 1.75rem; font-weight: 700; color: #1d4ed8; }
  .metric-label { font-size: 0.8rem; color: #64748b; margin-top: 5px; font-weight: 500; text-transform: uppercase; letter-spacing: 0.05em; }
  .metric-delta-pos { font-size: 0.8rem; color: #16a34a; margin-top: 4px; font-weight: 600; }
  .metric-delta-neg { font-size: 0.8rem; color: #dc2626; margin-top: 4px; font-weight: 600; }

  /* ── Section Headers ── */
  .section-header {
    font-size: 1.35rem;
    font-weight: 700;
    color: #0f172a;
    border-left: 4px solid #3b82f6;
    padding-left: 12px;
    margin-bottom: 18px;
  }

  /* ── Buttons ── */
  .stButton > button {
    background: linear-gradient(135deg, #3b82f6, #6366f1);
    color: #ffffff !important;
    border: none;
    border-radius: 8px;
    padding: 8px 20px;
    font-weight: 600;
    width: 100%;
  }
  .stButton > button:hover { opacity: 0.9; }

  /* ── Agent Boxes ── */
  .agent-box {
    background: #f0f9ff;
    border: 1px solid #bae6fd;
    border-left: 4px solid #0ea5e9;
    border-radius: 8px;
    padding: 14px 16px;
    margin: 8px 0;
    color: #0c4a6e;
    font-size: 0.95rem;
    line-height: 1.6;
  }

  /* ── Chat Messages ── */
  .chat-user {
    background: #eff6ff;
    border: 1px solid #bfdbfe;
    border-radius: 10px 10px 2px 10px;
    padding: 10px 14px;
    margin: 6px 0 2px 30px;
    color: #1e40af;
    font-size: 0.93rem;
  }
  .chat-bot {
    background: #f0fdf4;
    border: 1px solid #bbf7d0;
    border-radius: 2px 10px 10px 10px;
    padding: 10px 14px;
    margin: 2px 30px 6px 0;
    color: #14532d;
    font-size: 0.93rem;
  }

  /* ── Info box ── */
  .info-box {
    background: #fefce8;
    border: 1px solid #fde68a;
    border-radius: 8px;
    padding: 12px 16px;
    color: #78350f;
    font-size: 0.9rem;
  }

  /* ── Divider ── */
  hr { border-color: #e2e8f0; }

  /* ── Tab labels ── */
  .stTabs [data-baseweb="tab"] { color: #475569 !important; font-weight: 600; }
  .stTabs [aria-selected="true"] { color: #1d4ed8 !important; border-bottom-color: #1d4ed8 !important; }

  /* ── Dataframe ── */
  [data-testid="stDataFrame"] { border: 1px solid #e2e8f0; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
CHANNELS = [
    "google_ads_spend", "facebook_ads_spend", "youtube_ads_spend",
    "tv_ads_spend", "influencer_marketing_spend", "email_marketing_spend",
]
CHANNEL_LABELS = {
    "google_ads_spend":           "Google Ads",
    "facebook_ads_spend":         "Facebook Ads",
    "youtube_ads_spend":          "YouTube Ads",
    "tv_ads_spend":               "TV Ads",
    "influencer_marketing_spend": "Influencer",
    "email_marketing_spend":      "Email",
}
# Solid colors (no transparency in the color itself)
COLORS = ["#3b82f6", "#8b5cf6", "#ec4899", "#f97316", "#22c55e", "#eab308"]
# Pre-built rgba fill versions
COLORS_FILL = [
    "rgba(59,130,246,0.15)", "rgba(139,92,246,0.15)", "rgba(236,72,153,0.15)",
    "rgba(249,115,22,0.15)", "rgba(34,197,94,0.15)",  "rgba(234,179,8,0.15)",
]
TARGET = "sales_revenue"

# Light plotly layout — applied to every figure
def light_layout(**extra):
    base = dict(
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        font=dict(color="#1e293b", family="Inter, Arial, sans-serif", size=12),
        xaxis=dict(gridcolor="#f1f5f9", linecolor="#e2e8f0", zerolinecolor="#e2e8f0"),
        yaxis=dict(gridcolor="#f1f5f9", linecolor="#e2e8f0", zerolinecolor="#e2e8f0"),
        legend=dict(bgcolor="#f8fafc", bordercolor="#e2e8f0", borderwidth=1),
        margin=dict(t=48, b=32, l=16, r=16),
    )
    base.update(extra)
    return base


# ── Helpers ───────────────────────────────────────────────────────────────────
def kpi_card(label, value, delta="", positive=True, col=None):
    delta_cls = "metric-delta-pos" if positive else "metric-delta-neg"
    delta_html = f'<div class="{delta_cls}">{delta}</div>' if delta else ""
    html = f"""
    <div class="metric-card">
      <div class="metric-value">{value}</div>
      <div class="metric-label">{label}</div>
      {delta_html}
    </div>"""
    target = col if col else st
    target.markdown(html, unsafe_allow_html=True)


def fmt_currency(val):
    if abs(val) >= 1_000_000:
        return f"${val/1_000_000:.2f}M"
    if abs(val) >= 1_000:
        return f"${val/1_000:.1f}K"
    return f"${val:.0f}"


def ts_to_ms(dt):
    """Convert datetime / Timestamp to milliseconds (for add_vline on time axes)."""
    return int(pd.Timestamp(dt).timestamp() * 1000)


# ── Data & Model Loaders ──────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_raw_data():
    path = Path("data/marketing_mmm_data.csv")
    if not path.exists():
        from src.data_generation import generate_marketing_data
        return generate_marketing_data(str(path))
    return pd.read_csv(path, parse_dates=["date"])


@st.cache_data(show_spinner=False)
def load_engineered_data():
    from src.preprocessing import engineer_features
    return engineer_features(load_raw_data())


@st.cache_resource(show_spinner=False)
def load_models():
    if not Path("models/all_results.pkl").exists():
        _run_mini_pipeline()
    try:
        return joblib.load("models/all_results.pkl"), joblib.load("models/feat_cols.pkl")
    except Exception:
        return {}, []


@st.cache_resource(show_spinner=False)
def load_forecast_results():
    try:
        return (joblib.load("models/prophet_result.pkl"),
                joblib.load("models/xgb_forecast_result.pkl"),
                joblib.load("models/arima_result.pkl"))
    except Exception:
        return None, None, None


@st.cache_resource(show_spinner=False)
def load_opt_result():
    try:
        return joblib.load("models/opt_result.pkl")
    except Exception:
        from src.optimization import optimize_budget
        df = load_raw_data()
        return optimize_budget(float(df[CHANNELS].mean().sum() * 90))


@st.cache_resource(show_spinner=False)
def load_shap_result():
    try:
        return joblib.load("models/shap_result.pkl")
    except Exception:
        return None


def _run_mini_pipeline():
    with st.spinner("First run: training models (~2 min)…"):
        from src.preprocessing import engineer_features, train_test_split_temporal, get_feature_sets
        from src.mmm_model import train_all_models
        Path("models").mkdir(exist_ok=True)
        Path("reports").mkdir(exist_ok=True)
        df = engineer_features(load_raw_data())
        train, test = train_test_split_temporal(df)
        feat_cols = [c for c in get_feature_sets(df)["with_saturation"] if c in df.columns]
        results, feat_cols = train_all_models(df, feat_cols, train, test)
        joblib.dump(results, "models/all_results.pkl")
        joblib.dump(feat_cols, "models/feat_cols.pkl")


# ── Sidebar ───────────────────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.markdown("## 📊 MMM Platform")
        st.markdown("*AI-Powered Marketing Analytics*")
        st.divider()
        page = st.radio("Navigate", [
            "📈 Executive Overview",
            "📣 Marketing Analytics",
            "🔮 Forecasting",
            "💰 Optimization",
            "🔍 Explainability",
            "🤖 AI Assistant",
        ], label_visibility="collapsed")
        st.divider()
        st.markdown("#### ⚙️ Settings")
        date_range = st.select_slider("Analysis Period",
            options=["6M", "1Y", "2Y", "3Y"], value="1Y")
        st.session_state["date_range"] = date_range
        st.divider()
        st.caption("🔑 Add GROQ_API_KEY to .env for live AI responses.")
    return page


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 1 — Executive Overview
# ══════════════════════════════════════════════════════════════════════════════
def page_executive_overview(df):
    st.markdown('<div class="section-header">📈 Executive Overview</div>', unsafe_allow_html=True)

    period_map = {"6M": 180, "1Y": 365, "2Y": 730, "3Y": 1095}
    days = period_map.get(st.session_state.get("date_range", "1Y"), 365)
    dft = df.tail(days).copy()

    total_rev   = dft[TARGET].sum()
    avg_daily   = dft[TARGET].mean()
    total_spend = dft[CHANNELS].sum().sum()
    roi         = total_rev / (total_spend + 1e-9)
    best_ch     = max(CHANNELS, key=lambda c: dft[c].mean())
    prior_rev   = df.iloc[-(days*2):-days][TARGET].sum() if len(df) >= days*2 else total_rev
    trend_pct   = (total_rev - prior_rev) / (prior_rev + 1e-9) * 100

    c1, c2, c3, c4, c5 = st.columns(5)
    kpi_card("Total Revenue",    fmt_currency(total_rev),
             f"{'▲' if trend_pct>=0 else '▼'} {abs(trend_pct):.1f}% vs prior",
             trend_pct >= 0, c1)
    kpi_card("Avg Daily Revenue", fmt_currency(avg_daily), "", True, c2)
    kpi_card("Total Ad Spend",   fmt_currency(total_spend), "", True, c3)
    kpi_card("Blended ROI",      f"{roi:.2f}x", "", roi >= 1.0, c4)
    kpi_card("Top Channel",      CHANNEL_LABELS.get(best_ch, best_ch), "", True, c5)

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns([2, 1])

    # Revenue trend
    with col1:
        wk = dft.set_index("date")[TARGET].resample("W").sum().reset_index()
        wk.columns = ["date", "revenue"]
        wk["ma"] = wk["revenue"].rolling(4).mean()
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=wk["date"], y=wk["revenue"],
            fill="tozeroy", fillcolor="rgba(59,130,246,0.12)",
            line=dict(color="#3b82f6", width=2), name="Weekly Revenue"))
        fig.add_trace(go.Scatter(
            x=wk["date"], y=wk["ma"],
            line=dict(color="#ec4899", width=2, dash="dash"), name="4-Wk MA"))
        fig.update_layout(title="Weekly Revenue Trend", height=320, **light_layout())
        st.plotly_chart(fig, use_container_width=True)

    # Spend pie
    with col2:
        vals = [dft[ch].sum() for ch in CHANNELS]
        lbls = [CHANNEL_LABELS[ch] for ch in CHANNELS]
        fig2 = go.Figure(go.Pie(
            labels=lbls, values=vals, hole=0.45,
            marker=dict(colors=COLORS), textinfo="label+percent",
            textfont=dict(size=10)))
        fig2.update_layout(title="Spend Mix", height=320,
                           showlegend=False, **light_layout())
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)

    # Spend vs Revenue scatter
    with col3:
        dm = dft.set_index("date").resample("M").agg(
            {TARGET: "sum", **{ch: "sum" for ch in CHANNELS}}).reset_index()
        dm["total_spend"] = dm[CHANNELS].sum(axis=1)
        fig3 = px.scatter(dm, x="total_spend", y=TARGET, trendline="ols",
            color_discrete_sequence=["#3b82f6"],
            labels={"total_spend": "Monthly Spend ($)", TARGET: "Monthly Revenue ($)"},
            title="Monthly Spend vs Revenue")
        fig3.update_layout(height=300, **light_layout())
        st.plotly_chart(fig3, use_container_width=True)

    # Stacked area channel spend
    with col4:
        dp = dft.set_index("date")[CHANNELS].resample("W").sum().reset_index()
        fig4 = go.Figure()
        for i, ch in enumerate(CHANNELS):
            fig4.add_trace(go.Scatter(
                x=dp["date"], y=dp[ch],
                stackgroup="one", name=CHANNEL_LABELS[ch],
                fillcolor=COLORS_FILL[i],
                line=dict(color=COLORS[i], width=0.8)))
        fig4.update_layout(title="Channel Spend Over Time",
                           height=300, **light_layout())
        st.plotly_chart(fig4, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 2 — Marketing Analytics
# ══════════════════════════════════════════════════════════════════════════════
def page_marketing_analytics(df):
    st.markdown('<div class="section-header">📣 Marketing Analytics</div>', unsafe_allow_html=True)

    results, feat_cols = load_models()
    from src.optimization import CHANNEL_ROI

    roi_path = Path("reports/channel_roi.csv")
    if roi_path.exists():
        roi_df = pd.read_csv(roi_path)
    else:
        roi_df = pd.DataFrame({
            "channel": CHANNELS,
            "ROI":         [CHANNEL_ROI[c] for c in CHANNELS],
            "marginal_ROI":[CHANNEL_ROI[c]*0.7 for c in CHANNELS],
        })

    contrib_path = Path("reports/channel_contributions.csv")
    if contrib_path.exists():
        contrib_df = pd.read_csv(contrib_path)
    else:
        contrib_df = pd.DataFrame({"channel": CHANNELS, "contribution_pct": [16.7]*6})

    roi_df["Channel"] = roi_df["channel"].map(CHANNEL_LABELS).fillna(roi_df["channel"])
    contrib_df["Channel"] = contrib_df["channel"].map(CHANNEL_LABELS).fillna(contrib_df["channel"])

    col1, col2 = st.columns(2)

    with col1:
        roi_sorted = roi_df.sort_values("ROI", ascending=True)
        fig = go.Figure(go.Bar(
            x=roi_sorted["ROI"], y=roi_sorted["Channel"],
            orientation="h",
            marker=dict(color=roi_sorted["ROI"], colorscale="Blues"),
            text=[f"{r:.2f}x" for r in roi_sorted["ROI"]],
            textposition="outside"))
        fig.update_layout(title="Channel ROI", height=350,
                          showlegend=False, **light_layout())
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        pcts = contrib_df["contribution_pct"].abs()
        fig2 = go.Figure(go.Pie(
            labels=contrib_df["Channel"], values=pcts,
            hole=0.5, marker=dict(colors=COLORS),
            textinfo="label+percent"))
        fig2.update_layout(title="Revenue Attribution",
                           height=350, showlegend=False, **light_layout())
        st.plotly_chart(fig2, use_container_width=True)

    # Correlation heatmap
    st.markdown("#### Channel Correlations with Revenue")
    corr = df[CHANNELS + [TARGET]].corr()
    lbls = [CHANNEL_LABELS.get(c, c) for c in corr.columns]
    fig3 = go.Figure(go.Heatmap(
        z=corr.values, x=lbls, y=lbls,
        colorscale="RdBu_r", zmid=0, zmin=-1, zmax=1,
        text=corr.values.round(2).astype(str),
        texttemplate="%{text}", showscale=True))
    fig3.update_layout(title="Correlation Heatmap",
                       height=420, **light_layout())
    st.plotly_chart(fig3, use_container_width=True)

    # Model comparison
    metrics_path = Path("reports/model_metrics.csv")
    if metrics_path.exists():
        st.markdown("#### Model Performance Comparison")
        mdf = pd.read_csv(metrics_path)
        print(mdf.columns.tolist())
        st.write(mdf.columns.tolist())
        col3, col4 = st.columns(2)
        with col3:
            fig4 = go.Figure()
            fig4.add_trace(go.Bar(name="Train R²", x=mdf["Model"],
                y=mdf["Train R2"], marker_color="#3b82f6"))
            fig4.add_trace(go.Bar(name="Test R²", x=mdf["Model"],
                y=mdf["Test R2"], marker_color="#22c55e"))
            fig4.update_layout(barmode="group", title="R² Score",
                               height=300, **light_layout())
            st.plotly_chart(fig4, use_container_width=True)
        with col4:
            fig5 = go.Figure(go.Bar(
                x=mdf["Model"], y=mdf["Test RMSE"],
                marker=dict(color=mdf["Test RMSE"], colorscale="Reds_r"),
                text=mdf["Test RMSE"].round(0),
                textposition="outside"))
            fig5.update_layout(title="Test RMSE (lower = better)",
                               height=300, showlegend=False, **light_layout())
            st.plotly_chart(fig5, use_container_width=True)

    # Saturation curve
    st.markdown("#### Channel Response (Saturation) Curve")
    from src.saturation import hill_saturation
    from src.optimization import SAT_PARAMS
    sel = st.selectbox("Channel", CHANNELS,
                       format_func=lambda c: CHANNEL_LABELS.get(c, c),
                       key="sat_ch")
    max_s = df[sel].quantile(0.99) * 2
    xs = np.linspace(0, max_s, 300)
    p = SAT_PARAMS.get(sel, {"alpha": 0.6, "gamma": 1500})
    ys = hill_saturation(xs, p["alpha"], p["gamma"])
    cur_mean = float(df[sel].mean())

    fig6 = go.Figure()
    fig6.add_trace(go.Scatter(
        x=xs, y=ys, fill="tozeroy",
        fillcolor="rgba(59,130,246,0.10)",
        line=dict(color="#3b82f6", width=2.5),
        name="Saturation"))
    # mark current spend with a vertical line using shapes (avoids vline timestamp issue)
    fig6.add_shape(type="line",
        x0=cur_mean, x1=cur_mean, y0=0, y1=1,
        line=dict(color="#ec4899", width=2, dash="dash"))
    fig6.add_annotation(x=cur_mean, y=0.95,
        text=f"Current avg ${cur_mean:,.0f}",
        showarrow=True, arrowhead=2,
        font=dict(color="#ec4899"), bgcolor="#fff")
    fig6.update_layout(
        title=f"{CHANNEL_LABELS.get(sel)} — Hill Saturation Curve",
        xaxis_title="Spend ($)", yaxis_title="Saturation Index (0–1)",
        height=350, **light_layout())
    st.plotly_chart(fig6, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 3 — Forecasting
# ══════════════════════════════════════════════════════════════════════════════
def page_forecasting(df):
    st.markdown('<div class="section-header">🔮 Forecasting & Scenario Simulator</div>',
                unsafe_allow_html=True)

    prophet_r, xgb_r, arima_r = load_forecast_results()
    tab1, tab2 = st.tabs(["📅 Sales Forecast", "🎛️ Scenario Simulator"])

    with tab1:
        # Metric row
        c1, c2, c3 = st.columns(3)
        for r, col, name in [(prophet_r, c1, "Prophet"),
                             (arima_r,   c2, "ARIMA"),
                             (xgb_r,     c3, "XGBoost")]:
            m = r.get("metrics", {}) if r else {}
            with col:
                st.markdown(f"**{name}**")
                st.metric("R²",    f"{float(m.get('R2',0)):.4f}")
                st.metric("RMSE",  f"${float(m.get('RMSE',0)):,.0f}")
                st.metric("MAPE",  f"{float(m.get('MAPE',0)):.2f}%")

        # Forecast chart — NO add_vline on date axis
        fig = go.Figure()
        hist = df.tail(180)
        fig.add_trace(go.Scatter(
            x=hist["date"], y=hist[TARGET],
            name="Historical",
            line=dict(color="#94a3b8", width=1.5)))

        forecast_start = df["date"].max()

        for r, name, color, fill in [
            (prophet_r, "Prophet", "#3b82f6", "rgba(59,130,246,0.10)"),
            (xgb_r,     "XGBoost","#22c55e", "rgba(34,197,94,0.10)"),
            (arima_r,   "ARIMA",  "#ec4899", "rgba(236,72,153,0.10)"),
        ]:
            if r is None:
                continue
            ff = r.get("future_forecast")
            if ff is None or len(ff) == 0:
                continue
            x_col = "ds" if "ds" in ff.columns else "date"
            fig.add_trace(go.Scatter(
                x=ff[x_col], y=ff["yhat"],
                name=f"{name} Forecast",
                line=dict(color=color, width=2, dash="dot")))
            # Prophet CI band — use rgba strings, never hex+alpha
            if name == "Prophet" and "yhat_lower" in ff.columns:
                x_ci = list(ff[x_col]) + list(ff[x_col])[::-1]
                y_ci = list(ff["yhat_upper"]) + list(ff["yhat_lower"])[::-1]
                fig.add_trace(go.Scatter(
                    x=x_ci, y=y_ci,
                    fill="toself",
                    fillcolor="rgba(59,130,246,0.10)",
                    line=dict(color="rgba(0,0,0,0)"),
                    showlegend=False, name="Prophet CI"))

        # Use shape instead of add_vline to avoid timestamp arithmetic issues
        fig.add_shape(type="line",
            xref="x", yref="paper",
            x0=forecast_start, x1=forecast_start,
            y0=0, y1=1,
            line=dict(color="#f97316", width=2, dash="dash"))
        fig.add_annotation(
            x=forecast_start, y=1.02, yref="paper",
            text="Forecast Start", showarrow=False,
            font=dict(color="#f97316", size=11))

        fig.update_layout(title="90-Day Sales Forecast",
                          height=460, **light_layout())
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.markdown("#### What-If Scenario Simulator")
        st.info("Drag the sliders to change each channel's spend multiplier, then click **Run Scenario**.")

        col_sl, col_res = st.columns([1, 1])
        with col_sl:
            st.markdown("**Spend Multipliers (1.0 = current)**")
            changes = {}
            for ch in CHANNELS:
                changes[ch] = st.slider(
                    CHANNEL_LABELS.get(ch, ch),
                    0.0, 3.0, 1.0, 0.05, key=f"scen_{ch}")

        with col_res:
            if st.button("▶ Run Scenario", type="primary"):
                from src.forecasting import scenario_simulator
                results, feat_cols = load_models()
                if results:
                    best_name = max(results, key=lambda k: results[k]["test_metrics"]["R2"])
                    model  = results[best_name]["model"]
                    scaler = results[best_name]["scaler"]
                    valid  = [c for c in feat_cols if c in df.columns]
                    out = scenario_simulator(df, changes, model, valid, scaler)

                    st.markdown("### Results")
                    ca, cb = st.columns(2)
                    ca.metric("Baseline Avg/Day",
                              fmt_currency(out["baseline_avg"]))
                    cb.metric("Scenario Avg/Day",
                              fmt_currency(out["scenario_avg"]),
                              delta=f"{out['uplift_pct']:+.2f}%")

                    lift = out["revenue_lift"]
                    color = "#16a34a" if lift >= 0 else "#dc2626"
                    sign  = "+" if lift >= 0 else ""
                    st.markdown(f"""
                    <div style='background:#f0fdf4;border:1px solid #bbf7d0;
                         border-radius:8px;padding:14px;margin-top:8px'>
                      <b>Total Revenue Lift:</b>
                      <span style='color:{color};font-size:1.25rem;font-weight:700'>
                        {sign}{fmt_currency(lift)}
                      </span><br>
                      <small style='color:#64748b'>Over full data period at scenario spend</small>
                    </div>""", unsafe_allow_html=True)

                    # Change bars
                    cdf = pd.DataFrame({
                        "Channel": [CHANNEL_LABELS.get(c, c) for c in CHANNELS],
                        "Multiplier": [changes[c] for c in CHANNELS],
                    })
                    fig_s = go.Figure(go.Bar(
                        x=cdf["Channel"], y=cdf["Multiplier"],
                        marker=dict(
                            color=["#22c55e" if v >= 1 else "#ef4444"
                                   for v in cdf["Multiplier"]]),
                        text=[f"{v:.2f}x" for v in cdf["Multiplier"]],
                        textposition="outside"))
                    fig_s.add_shape(type="line",
                        x0=-0.5, x1=len(CHANNELS)-0.5, y0=1, y1=1,
                        line=dict(color="#94a3b8", dash="dash", width=1.5))
                    fig_s.update_layout(title="Spend Multipliers vs Baseline",
                        height=280, **light_layout())
                    st.plotly_chart(fig_s, use_container_width=True)
                else:
                    st.warning("Run `python main.py` first to train models.")


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 4 — Optimization
# ══════════════════════════════════════════════════════════════════════════════
def page_optimization():
    st.markdown('<div class="section-header">💰 Budget Optimization</div>',
                unsafe_allow_html=True)

    from src.optimization import optimize_budget, allocation_summary, build_response_curves

    df_raw = load_raw_data()
    default_budget = float(df_raw[CHANNELS].mean().sum() * 90)

    col_ctrl, col_info = st.columns([1, 2])
    with col_ctrl:
        st.markdown("#### Controls")
        budget = st.number_input(
            "Total Budget (90-day, $)", min_value=100_000.0,
            max_value=5_000_000.0, value=round(default_budget, -3),
            step=10_000.0, format="%.0f")
        min_pct = st.slider("Min % per channel", 1, 20, 5) / 100
        max_pct = st.slider("Max % per channel", 20, 80, 50) / 100
        run_btn = st.button("⚡ Optimize Budget", type="primary")

    with col_info:
        st.markdown("""
        **Optimization Engine**
        - Algorithm: `scipy.optimize.minimize` (SLSQP)
        - Objective: maximize Hill-saturated, ROI-weighted revenue
        - Constraints: total budget equality + per-channel min/max bounds
        - Response curves calibrated from 3-year data
        """)

    opt_result = None
    if run_btn:
        with st.spinner("Optimizing…"):
            opt_result = optimize_budget(budget, min_spend_pct=min_pct, max_spend_pct=max_pct)
            joblib.dump(opt_result, "models/opt_result.pkl")
    elif Path("models/opt_result.pkl").exists():
        opt_result = load_opt_result()

    if opt_result:
        alloc = allocation_summary(opt_result)

        c1, c2, c3 = st.columns(3)
        kpi_card("Baseline Revenue",  fmt_currency(opt_result["baseline_revenue"]), "", True, c1)
        kpi_card("Optimized Revenue", fmt_currency(opt_result["optimized_revenue"]),
                 f"▲ {opt_result['revenue_uplift']:.1f}% uplift", True, c2)
        kpi_card("Revenue Lift",      fmt_currency(opt_result["revenue_lift_abs"]),
                 "90-day absolute", opt_result["revenue_lift_abs"] >= 0, c3)

        st.markdown("<br>", unsafe_allow_html=True)
        col_l, col_r = st.columns(2)

        with col_l:
            fig = go.Figure()
            fig.add_trace(go.Bar(name="Current",
                x=alloc["channel"], y=alloc["current_spend"],
                marker_color="#94a3b8"))
            fig.add_trace(go.Bar(name="Optimized",
                x=alloc["channel"], y=alloc["optimized_spend"],
                marker_color="#3b82f6"))
            fig.update_layout(barmode="group",
                title="Current vs Optimized Allocation",
                xaxis_tickangle=-30, height=380, **light_layout())
            st.plotly_chart(fig, use_container_width=True)

        with col_r:
            fig2 = go.Figure(go.Bar(
                x=alloc["channel"], y=alloc["change_pct"],
                marker_color=["#22c55e" if v >= 0 else "#ef4444"
                              for v in alloc["change_pct"]],
                text=[f"{v:+.1f}%" for v in alloc["change_pct"]],
                textposition="outside"))
            fig2.add_shape(type="line",
                x0=-0.5, x1=len(alloc)-0.5, y0=0, y1=0,
                line=dict(color="#94a3b8", width=1))
            fig2.update_layout(title="% Change from Current",
                xaxis_tickangle=-30, height=380, **light_layout())
            st.plotly_chart(fig2, use_container_width=True)

        # Response curve
        st.markdown("#### Channel Response Curve")
        resp_ch = st.selectbox("Channel", CHANNELS,
            format_func=lambda c: CHANNEL_LABELS.get(c, c), key="resp_ch")
        curve = build_response_curves(resp_ch, n_points=200)
        opt_s  = opt_result["optimized_allocation"].get(resp_ch, 0)
        curr_s = opt_result["current_allocation"].get(resp_ch, 0)

        fig3 = make_subplots(specs=[[{"secondary_y": True}]])
        fig3.add_trace(go.Scatter(
            x=curve["spend"], y=curve["revenue"],
            fill="tozeroy", fillcolor="rgba(59,130,246,0.10)",
            line=dict(color="#3b82f6", width=2), name="Revenue"),
            secondary_y=False)
        fig3.add_trace(go.Scatter(
            x=curve["spend"], y=curve["marginal_roi"],
            line=dict(color="#ec4899", width=1.5, dash="dot"),
            name="Marginal ROI"),
            secondary_y=True)
        # Use shapes instead of add_vline
        for xv, clr, lbl in [(curr_s, "#f97316", "Current"),
                              (opt_s,  "#22c55e", "Optimized")]:
            fig3.add_shape(type="line",
                x0=xv, x1=xv, y0=0, y1=1, yref="paper",
                line=dict(color=clr, width=2, dash="dash"))
            fig3.add_annotation(x=xv, y=1.04, yref="paper",
                text=lbl, showarrow=False,
                font=dict(color=clr, size=10))
        fig3.update_layout(title=f"{CHANNEL_LABELS.get(resp_ch)} Response Curve",
            height=380, **light_layout())
        st.plotly_chart(fig3, use_container_width=True)

        # Summary table
        st.markdown("#### Allocation Summary Table")
        disp = alloc[["channel", "current_spend", "optimized_spend",
                       "change", "change_pct"]].copy()
        disp.columns = ["Channel", "Current ($)", "Optimized ($)", "Change ($)", "Change %"]
        for col in ["Current ($)", "Optimized ($)", "Change ($)"]:
            disp[col] = disp[col].apply(lambda x: f"${x:,.0f}")
        disp["Change %"] = disp["Change %"].apply(lambda x: f"{x:+.1f}%")
        st.dataframe(disp, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 5 — Explainability
# ══════════════════════════════════════════════════════════════════════════════
def page_explainability(df):
    st.markdown('<div class="section-header">🔍 Explainability (SHAP)</div>',
                unsafe_allow_html=True)

    shap_result = load_shap_result()

    if shap_result is None:
        results, feat_cols = load_models()
        if results:
            with st.spinner("Computing SHAP values…"):
                from src.explainability import compute_shap_values
                ridge  = results.get("Ridge Regression", list(results.values())[0])
                model  = ridge["model"]
                scaler = ridge["scaler"]
                valid  = [c for c in feat_cols if c in df.columns]
                X = df[valid].fillna(0)
                if scaler is not None:
                    X = pd.DataFrame(scaler.transform(X.values), columns=valid)
                shap_result = compute_shap_values(model, X, model_type="linear")
                joblib.dump(shap_result, "models/shap_result.pkl")
        else:
            st.error("Run `python main.py` first.")
            return

    from src.explainability import (
        plot_feature_importance, plot_shap_summary,
        plot_waterfall, get_top_sales_drivers
    )

    # --- patch all figures to light layout before rendering ---
    def lightify(fig):
        fig.update_layout(
            plot_bgcolor="#ffffff", paper_bgcolor="#ffffff",
            font=dict(color="#1e293b"),
            xaxis=dict(gridcolor="#f1f5f9"),
            yaxis=dict(gridcolor="#f1f5f9"))
        return fig

    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Feature Importance", "🐝 SHAP Summary",
        "🌊 Waterfall", "📋 Sales Drivers"])

    with tab1:
        fig = lightify(plot_feature_importance(shap_result, top_n=20))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Mean absolute SHAP — higher = more impact on predictions.")

    with tab2:
        fig2 = lightify(plot_shap_summary(shap_result, top_n=15))
        st.plotly_chart(fig2, use_container_width=True)
        st.caption("Right of zero = positive sales impact. Colour shows feature value magnitude.")

    with tab3:
        n = len(shap_result["X_sample"])
        idx = st.slider("Sample index", 0, n - 1, 0)
        fig3 = lightify(plot_waterfall(shap_result, sample_idx=idx))
        st.plotly_chart(fig3, use_container_width=True)

    with tab4:
        drivers = get_top_sales_drivers(shap_result, top_n=8)
        fig4 = go.Figure(go.Bar(
            x=drivers["mean_shap"],
            y=drivers["feature"].str.replace("_"," ").str.title(),
            orientation="h",
            marker_color=["#22c55e" if d == "Positive" else "#ef4444"
                          for d in drivers["direction"]],
            text=drivers["mean_shap"].round(2),
            textposition="outside"))
        fig4 = lightify(fig4)

        fig4.update_layout(
            title="Mean SHAP per Feature",
            height=420,
            yaxis=dict(
                categoryorder="total ascending",
                gridcolor="#f1f5f9"
                )
        )
        st.plotly_chart(fig4, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 6 — AI Assistant
# ══════════════════════════════════════════════════════════════════════════════
def page_ai_assistant(df):
    st.markdown('<div class="section-header">🤖 AI Marketing Assistant</div>',
                unsafe_allow_html=True)

    from src.agents import (chat_agent, analyst_agent, forecast_agent,
                            optimization_agent, reporting_agent)
    from src.groq_utils import GROQ_API_KEY

    opt_r = load_opt_result()
    roi_path = Path("reports/channel_roi.csv")
    if roi_path.exists():
        rdf = pd.read_csv(roi_path)
        best_ch  = rdf.loc[rdf["ROI"].idxmax(), "channel"] if "ROI" in rdf else "email_marketing_spend"
        worst_ch = rdf.loc[rdf["ROI"].idxmin(), "channel"] if "ROI" in rdf else "tv_ads_spend"
        best_roi = float(rdf["ROI"].max()) if "ROI" in rdf else 5.0
    else:
        best_ch, worst_ch, best_roi = "email_marketing_spend", "tv_ads_spend", 5.0

    ctx = {
        "avg_revenue":       round(float(df[TARGET].mean()), 2),
        "total_revenue":     round(float(df[TARGET].sum()), 2),
        "best_roi_channel":  CHANNEL_LABELS.get(best_ch, best_ch),
        "worst_roi_channel": CHANNEL_LABELS.get(worst_ch, worst_ch),
        "best_roi":          round(best_roi, 2),
    }

    if not GROQ_API_KEY:
        st.markdown("""
        <div class="info-box">
        ⚠️ <b>Groq API key not set.</b> Responses use smart rule-based fallback.
        Add <code>GROQ_API_KEY=gsk_xxx</code> to <code>.env</code> for live LLM responses.
        <a href="https://console.groq.com" target="_blank">Get free key →</a>
        </div>""", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

    # Agent buttons
    st.markdown("#### 🧑‍💼 Run AI Agents")
    c1, c2 = st.columns(2)
    agent_configs = [
        ("📊 Analyst Agent",     c1, analyst_agent,
         {"channel_roi": {CHANNEL_LABELS.get(c,c): 3.5 for c in CHANNELS},
          "trend": "upward +8% MoM", "anomaly": "none detected"}),
        ("🔮 Forecast Agent",    c2, forecast_agent,
         {"avg_daily_sales": ctx["avg_revenue"], "yoy_growth_pct": 8.5,
          "seasonality_note": "Q4 is 20% above average"}),
        ("💰 Optimization Agent",c1, optimization_agent,
         {"revenue_uplift": opt_r.get("revenue_uplift", 48.2) if opt_r else 48.2,
          "top_roi_channel": CHANNEL_LABELS.get(best_ch, best_ch),
          "low_roi_channel": CHANNEL_LABELS.get(worst_ch, worst_ch),
          "total_budget": opt_r.get("total_budget", 714290) if opt_r else 714290}),
        ("📝 Reporting Agent",   c2, reporting_agent,
         {"total_revenue": ctx["total_revenue"],
          "best_channel":  CHANNEL_LABELS.get(best_ch, best_ch),
          "worst_channel": CHANNEL_LABELS.get(worst_ch, worst_ch),
          "forecast_note": "projected 8–10% growth next quarter"}),
    ]
    for label, col, fn, agent_ctx in agent_configs:
        if col.button(label, key=f"btn_{label}"):
            with st.spinner("Agent running…"):
                out = fn(agent_ctx)
            st.markdown(f'<div class="agent-box">{out}</div>', unsafe_allow_html=True)

    st.divider()
    st.markdown("#### 💬 Chat with Your Marketing Data")

    # Quick questions
    st.markdown("**Quick questions:**")
    quick_qs = [
        "Which channel has the highest ROI?",
        "Why did sales drop last quarter?",
        "How should I allocate budget next quarter?",
        "What is the impact of email marketing?",
        "How does seasonality affect our sales?",
        "Should I increase TV ad spend?",
    ]
    qcols = st.columns(3)
    for i, q in enumerate(quick_qs):
        if qcols[i % 3].button(q, key=f"q_{i}"):
            st.session_state["pending_q"] = q

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    # Handle pending quick-question
    if "pending_q" in st.session_state:
        q = st.session_state.pop("pending_q")
        with st.spinner("Thinking…"):
            ans = chat_agent(q, context_metrics=ctx)
        st.session_state["chat_history"].append({"user": q, "bot": ans})

    # Free-form input
    with st.form("chat_form", clear_on_submit=True):
        user_in = st.text_input("Ask anything about your marketing data…",
                                placeholder="e.g. Which channel has the highest ROI?")
        submitted = st.form_submit_button("Send →")
    if submitted and user_in.strip():
        with st.spinner("Thinking…"):
            ans = chat_agent(user_in, context_metrics=ctx)
        st.session_state["chat_history"].append({"user": user_in, "bot": ans})

    # Chat history
    for item in reversed(st.session_state["chat_history"]):
        st.markdown(f'<div class="chat-user"><b>You:</b> {item["user"]}</div>',
                    unsafe_allow_html=True)
        st.markdown(f'<div class="chat-bot"><b>AI:</b> {item["bot"]}</div>',
                    unsafe_allow_html=True)

    if st.session_state["chat_history"] and st.button("🗑️ Clear Chat"):
        st.session_state["chat_history"] = []
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    page = render_sidebar()
    with st.spinner("Loading data…"):
        df = load_engineered_data()

    if   "Executive"     in page: page_executive_overview(df)
    elif "Analytics"     in page: page_marketing_analytics(df)
    elif "Forecasting"   in page: page_forecasting(df)
    elif "Optimization"  in page: page_optimization()
    elif "Explainability"in page: page_explainability(df)
    elif "Assistant"     in page: page_ai_assistant(df)


if __name__ == "__main__":
    main()
