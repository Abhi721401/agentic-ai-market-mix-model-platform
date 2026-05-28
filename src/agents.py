"""
agents.py
---------
AI Agents using Groq LLM (LangChain integration).
Agents:
  1. Analyst Agent    — campaign performance + anomaly detection
  2. Forecast Agent   — sales prediction
  3. Optimization Agent — budget recommendations
  4. Reporting Agent  — summaries
  5. Chat Agent       — Q&A interface

Token-efficient: short prompts + context injection only.
Uses free Groq llama3-8b-8192 model.
"""

import os
from typing import Optional
from src.groq_utils import groq_complete, build_context_string

# ─── Shared System Prompts (short) ────────────────────────────────────────────

SYS_ANALYST = (
    "You are an expert marketing analyst. "
    "Give 2-3 sentence data-driven insights. "
    "Be specific with numbers. No fluff."
)

SYS_FORECAST = (
    "You are a sales forecasting expert. "
    "Give concrete predictions with % changes. "
    "2-3 sentences max."
)

SYS_OPT = (
    "You are a marketing budget optimizer. "
    "Give specific budget reallocation advice. "
    "Use percentages and channel names. 2-3 sentences."
)

SYS_REPORT = (
    "You are a business reporting specialist. "
    "Write executive-level summaries in 3-4 sentences. "
    "Focus on business impact."
)

SYS_CHAT = (
    "You are a marketing analytics assistant for an MMM platform. "
    "Answer questions about marketing ROI, budget allocation, and sales performance. "
    "Be concise (2-4 sentences) and data-driven."
)


# ─── Agent Functions ──────────────────────────────────────────────────────────

def analyst_agent(context: dict) -> str:
    """
    Analyst Agent: Analyze campaign performance + detect anomalies.
    context: dict with roi_data, channel_contributions, recent_sales_trend
    """
    # Build compact prompt
    roi_str = " | ".join([
        f"{ch}: {roi:.1f}x"
        for ch, roi in context.get("channel_roi", {}).items()
    ])
    trend = context.get("trend", "stable")
    anomaly = context.get("anomaly", "none detected")

    prompt = (
        f"Channel ROIs: {roi_str}. "
        f"Sales trend: {trend}. "
        f"Anomaly: {anomaly}. "
        "Provide campaign performance analysis and key insights."
    )
    return groq_complete(prompt, system=SYS_ANALYST)


def forecast_agent(context: dict) -> str:
    """
    Forecast Agent: Predict future sales.
    context: dict with current_avg_sales, growth_rate, seasonality_note
    """
    avg = context.get("avg_daily_sales", 25000)
    growth = context.get("yoy_growth_pct", 8.5)
    season = context.get("seasonality_note", "Q4 typically 20% above average")

    prompt = (
        f"Current avg daily sales: ${avg:,.0f}. "
        f"YoY growth: {growth:.1f}%. "
        f"Seasonality: {season}. "
        "Forecast next quarter sales and identify key drivers."
    )
    return groq_complete(prompt, system=SYS_FORECAST)


def optimization_agent(context: dict) -> str:
    """
    Optimization Agent: Budget recommendations.
    context: dict with current_allocation, optimized_allocation, uplift_pct
    """
    uplift = context.get("revenue_uplift", 12.5)
    top_channel = context.get("top_roi_channel", "Email Marketing")
    low_channel = context.get("low_roi_channel", "TV Ads")
    budget = context.get("total_budget", 576000)

    prompt = (
        f"Budget: ${budget:,.0f}. "
        f"Optimization shows {uplift:.1f}% revenue uplift possible. "
        f"Highest ROI: {top_channel}. Lowest ROI: {low_channel}. "
        "Give specific budget reallocation advice."
    )
    return groq_complete(prompt, system=SYS_OPT)


def reporting_agent(context: dict) -> str:
    """
    Reporting Agent: Executive summary.
    context: dict with total_revenue, best_channel, worst_channel, forecast_note
    """
    total = context.get("total_revenue", 25_000_000)
    best = context.get("best_channel", "Email Marketing")
    worst = context.get("worst_channel", "TV Ads")
    forecast = context.get("forecast_note", "projected 10% growth")

    prompt = (
        f"Period revenue: ${total:,.0f}. "
        f"Best performing: {best}. "
        f"Underperforming: {worst}. "
        f"Outlook: {forecast}. "
        "Write an executive summary."
    )
    return groq_complete(prompt, system=SYS_REPORT)


def chat_agent(
    user_question: str,
    context_metrics: Optional[dict] = None,
) -> str:
    """
    Chat Agent: General Q&A.
    Injects compact context to avoid large history.
    """
    # Build short context
    ctx_parts = []
    if context_metrics:
        ctx_parts.append(
            f"[Data: avg_revenue=${context_metrics.get('avg_revenue', 25000):,.0f}/day, "
            f"best_roi={context_metrics.get('best_roi_channel', 'Email')} {context_metrics.get('best_roi', 5.0):.1f}x, "
            f"total_period_revenue=${context_metrics.get('total_revenue', 25_000_000):,.0f}]"
        )

    ctx_str = " ".join(ctx_parts)
    prompt = f"{ctx_str} User: {user_question}"

    return groq_complete(prompt, system=SYS_CHAT, max_tokens=400)


def run_all_agents(context: dict) -> dict:
    """Run all agents and return their outputs."""
    return {
        "analyst":      analyst_agent(context),
        "forecast":     forecast_agent(context),
        "optimization": optimization_agent(context),
        "report":       reporting_agent(context),
    }


# ─── LangChain Integration (optional, token-efficient) ────────────────────────

def get_langchain_chat_chain(context_metrics: dict = None):
    """
    Returns a LangChain chain for the chat agent.
    Uses Groq as the LLM backend.
    """
    try:
        from langchain_groq import ChatGroq
        from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
        from langchain.chains import LLMChain

        groq_key = os.getenv("GROQ_API_KEY", "")
        if not groq_key:
            return None

        llm = ChatGroq(
            api_key=groq_key,
            model_name="llama3-8b-8192",
            temperature=0.2,
            max_tokens=400,
        )

        ctx = ""
        if context_metrics:
            ctx = (
                f"MMM context: avg_daily_sales=${context_metrics.get('avg_revenue', 25000):,.0f}, "
                f"best_channel={context_metrics.get('best_roi_channel', 'Email')}, "
                f"best_roi={context_metrics.get('best_roi', 5.0):.1f}x. "
            )

        system_msg = SystemMessagePromptTemplate.from_template(
            f"You are a marketing mix modeling analyst. {ctx}"
            "Answer concisely in 2-4 sentences. Use numbers from context."
        )
        human_msg = HumanMessagePromptTemplate.from_template("{question}")
        prompt = ChatPromptTemplate.from_messages([system_msg, human_msg])
        chain = LLMChain(llm=llm, prompt=prompt)
        return chain

    except ImportError:
        return None


if __name__ == "__main__":
    ctx = {
        "channel_roi": {
            "google_ads_spend": 3.5,
            "email_marketing_spend": 5.0,
            "tv_ads_spend": 2.2,
        },
        "trend": "upward, +8% MoM",
        "anomaly": "sales dip on 2023-03-15 (-22%)",
        "avg_daily_sales": 27500,
        "yoy_growth_pct": 9.2,
        "revenue_uplift": 14.5,
        "top_roi_channel": "Email Marketing",
        "low_roi_channel": "TV Ads",
        "total_budget": 576000,
        "total_revenue": 26_000_000,
    }

    outputs = run_all_agents(ctx)
    for name, output in outputs.items():
        print(f"\n=== {name.upper()} ===")
        print(output)
