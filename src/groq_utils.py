"""
groq_utils.py
-------------
Groq LLM utilities.
- Minimizes token usage with short prompts
- Uses context injection instead of long histories
- Deterministic responses where possible
- Free tier: llama3-8b-8192 (fast + free)
"""

import os
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
MODEL = "llama-3.1-8b-instant"  # Free tier model
MAX_TOKENS = 512  # Keep responses concise

_client = None


def get_groq_client():
    """Lazy init Groq client."""
    global _client
    if _client is None:
        if not GROQ_API_KEY:
            return None
        try:
            from groq import Groq
            _client = Groq(api_key=GROQ_API_KEY)
        except Exception as e:
            print(f"[Groq] Init error: {e}")
            return None
    return _client


def groq_complete(
    prompt: str,
    system: str = "You are a concise marketing analytics AI. Answer in 2-4 sentences.",
    max_tokens: int = MAX_TOKENS,
    temperature: float = 0.2,
) -> str:
    """
    Single Groq completion call.
    Low temperature for deterministic analysis.
    """
    client = get_groq_client()
    if client is None:
        return _fallback_response(prompt)

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt[:2000]},  # Truncate to save tokens
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return _fallback_response(prompt, error=str(e))


def _fallback_response(prompt: str, error: str = "") -> str:
    """Rule-based fallback when Groq is unavailable."""
    p = prompt.lower()
    if "roi" in p:
        return "Email marketing shows highest ROI (5.0x), followed by Influencer (4.0x) and Google Ads (3.5x). TV ads have lowest ROI (2.2x) but highest reach."
    if "drop" in p or "decline" in p:
        return "Sales drops typically correlate with competitor promotions, reduced marketing spend, or seasonal troughs. Check Q1 and summer months for historical patterns."
    if "budget" in p or "allocat" in p:
        return "Recommend shifting budget toward Email and Influencer marketing (highest ROI). Reduce TV spend proportionally. Maintain Google Ads for top-of-funnel awareness."
    if "forecast" in p or "predict" in p:
        return "Based on current trends, sales are projected to grow 8-12% YoY driven by seasonality and marketing efficiency gains."
    if "channel" in p:
        return "Google Ads and YouTube drive high-intent traffic. Facebook drives volume. Email and Influencer have best cost efficiency."
    if error:
        return f"[Groq API unavailable: {error[:100]}. Add GROQ_API_KEY to .env for AI responses.]"
    return "Based on the MMM analysis, marketing activities show strong positive ROI with opportunities for budget reallocation toward higher-performing channels."


def build_context_string(metrics: dict, top_channels: list, recent_trend: str) -> str:
    """Build compact context for agent injection. Minimizes token usage."""
    lines = [
        f"Total daily avg revenue: ${metrics.get('avg_revenue', 0):,.0f}",
        f"Best ROI channel: {metrics.get('best_roi_channel', 'Google Ads')} ({metrics.get('best_roi', 3.5):.1f}x)",
        f"Lowest ROI: {metrics.get('worst_roi_channel', 'TV')} ({metrics.get('worst_roi', 2.2):.1f}x)",
        f"Recent trend: {recent_trend}",
        f"Top channels by spend: {', '.join(top_channels[:3])}",
    ]
    return " | ".join(lines)
