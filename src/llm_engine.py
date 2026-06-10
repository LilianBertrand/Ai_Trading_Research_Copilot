from __future__ import annotations

import os
from typing import Dict


def deterministic_commentary(ticker: str, fundamentals: Dict, risk: Dict, signal: Dict, prompt: str) -> str:
    return f"""
The copilot identifies {ticker} as the primary asset requested in the prompt: "{prompt}".

Analytical conclusion: {signal.get('action', 'Neutral')}. The current technical configuration is {signal.get('trend', 'mixed')} with {signal.get('momentum', 'mixed')} momentum. The risk engine indicates that the {signal.get('risk_comment', 'risk profile is mixed')}, with annualized volatility around {risk.get('Annual Volatility', 0):.2%} and maximum drawdown around {risk.get('Max Drawdown', 0):.2%} over the available history.

Fundamental view: the valuation screen is {signal.get('valuation', 'neutral')}. Key inputs include P/E, profit margin, revenue growth and leverage where available. The output should be interpreted as a structured research assistant view, not as a standalone investment recommendation.

Possible strategy framework: build exposure only if price action confirms the trend, size the position according to volatility, define a stop-loss level before entry, and compare the opportunity with the benchmark and with alternative assets in the portfolio.
""".strip()


def llm_commentary(ticker: str, fundamentals: Dict, risk: Dict, signal: Dict, prompt: str, model: str = "gpt-4o-mini") -> tuple[str, str]:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return deterministic_commentary(ticker, fundamentals, risk, signal, prompt), "rules-based fallback"
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        system = (
            "You are a cautious buy-side research copilot. Produce concise, structured financial analysis. "
            "Do not present the output as investment advice. Mention key risks and position sizing discipline."
        )
        user = f"""
Prompt: {prompt}
Ticker: {ticker}
Fundamentals: {fundamentals}
Risk metrics: {risk}
Quant signal: {signal}

Write a professional analyst-style research note with: thesis, risk, strategy, and watchlist levels.
"""
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0.25,
        )
        return resp.choices[0].message.content.strip(), f"OpenAI model: {model}"
    except Exception as exc:
        return deterministic_commentary(ticker, fundamentals, risk, signal, prompt) + f"\n\nLLM fallback reason: {exc}", "rules-based fallback after LLM error"
