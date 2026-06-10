from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except Exception:  # pragma: no cover
    yf = None


COMMON_TICKERS = {
    "apple": "AAPL", "aapl": "AAPL", "microsoft": "MSFT", "msft": "MSFT",
    "google": "GOOGL", "alphabet": "GOOGL", "amazon": "AMZN", "tesla": "TSLA",
    "nvidia": "NVDA", "meta": "META", "facebook": "META", "netflix": "NFLX",
    "sp500": "SPY", "s&p": "SPY", "nasdaq": "QQQ", "gold": "GC=F",
    "oil": "CL=F", "wti": "CL=F", "brent": "BZ=F", "bitcoin": "BTC-USD",
}

DEFAULT_UNIVERSE = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "SPY", "TLT", "GLD", "CL=F"]
COMMODITIES = ["GC=F", "CL=F", "BZ=F", "NG=F", "HG=F", "SI=F"]
BENCHMARK = "SPY"


@dataclass
class MarketDataBundle:
    prices: pd.DataFrame
    primary: str
    benchmark: str
    source: str
    warnings: List[str]


def parse_tickers(prompt: str) -> List[str]:
    """Extract tickers without confusing normal English words with symbols.

    The previous version uppercased the whole sentence, so a prompt such as
    "Analyze Apple and propose a strategy" became APPLE and AND, which
    Yahoo Finance interpreted as invalid tickers. This parser first maps
    known company names, then only accepts explicit ticker-looking tokens
    already written in uppercase by the user, e.g. AAPL, MSFT, BTC-USD.
    """
    text = prompt.lower().strip()
    found: List[str] = []

    # Natural-language aliases: "Apple" -> AAPL, "oil" -> CL=F, etc.
    for key, ticker in COMMON_TICKERS.items():
        if re.search(rf"\b{re.escape(key)}\b", text) and ticker not in found:
            found.append(ticker)

    # Explicit tickers only if the original text already contains uppercase tokens.
    # This avoids turning words like "and" into the fake ticker AND.
    stopwords = {
        "AI", "LLM", "PDF", "CEO", "CFO", "GDP", "USA", "USD", "EUR",
        "AND", "THE", "FOR", "WITH", "RISK", "BUY", "SELL", "HOLD",
        "APPLE", "MICROSOFT", "GOOGLE", "AMAZON", "TESLA", "NVIDIA",
        "META", "FACEBOOK", "NETFLIX", "ANALYZE", "STRATEGY",
    }
    explicit = re.findall(r"\b[A-Z]{1,5}(?:[-=][A-Z]{1,4})?\b", prompt)
    for ticker in explicit:
        ticker = ticker.strip().upper()
        if ticker not in stopwords and ticker not in found:
            found.append(ticker)

    return found[:8] or ["AAPL"]


def _demo_series(tickers: List[str], years: int = 4) -> pd.DataFrame:
    np.random.seed(42)
    n = 252 * years
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n)
    out = {}
    for i, ticker in enumerate(tickers):
        drift = 0.00025 + i * 0.00003
        vol = 0.012 + (i % 5) * 0.002
        returns = np.random.normal(drift, vol, n)
        shock = np.sin(np.linspace(0, 10, n)) * 0.0008
        price = 100 * np.exp(np.cumsum(returns + shock))
        out[ticker] = price
    return pd.DataFrame(out, index=dates)


def load_prices(tickers: List[str], period: str = "5y") -> Tuple[pd.DataFrame, str, List[str]]:
    warnings: List[str] = []
    clean = []
    for t in tickers:
        if isinstance(t, str) and t.strip() and t.strip().upper() not in clean:
            clean.append(t.strip().upper())
    if BENCHMARK not in clean:
        clean.append(BENCHMARK)

    if yf is None:
        warnings.append("yfinance is not installed. Demo market data is being used.")
        return _demo_series(clean), "demo", warnings

    try:
        raw = yf.download(clean, period=period, auto_adjust=True, progress=False, group_by="column", threads=True)
        if raw.empty:
            raise ValueError("Empty Yahoo Finance response")
        if isinstance(raw.columns, pd.MultiIndex):
            prices = raw["Close"].copy() if "Close" in raw.columns.get_level_values(0) else raw.xs("Close", level=1, axis=1)
        else:
            prices = raw[["Close"]].rename(columns={"Close": clean[0]})
        prices = prices.dropna(how="all").ffill().dropna(axis=1, how="all")
        if prices.shape[0] < 30 or prices.shape[1] == 0:
            raise ValueError("Insufficient price history")
        missing = [t for t in clean if t not in prices.columns]
        if missing:
            warnings.append("Missing or unavailable tickers: " + ", ".join(missing))
        return prices, "yfinance", warnings
    except Exception as exc:
        warnings.append(f"Live data failed ({exc}). Demo market data is being used.")
        return _demo_series(clean), "demo", warnings


def load_market_bundle(prompt: str, extra_tickers: List[str] | None = None, period: str = "5y") -> MarketDataBundle:
    tickers = parse_tickers(prompt)
    if extra_tickers:
        tickers.extend([t.strip().upper() for t in extra_tickers if t.strip()])
    prices, source, warnings = load_prices(tickers, period=period)
    primary = tickers[0] if tickers[0] in prices.columns else prices.columns[0]
    benchmark = BENCHMARK if BENCHMARK in prices.columns else primary
    return MarketDataBundle(prices=prices, primary=primary, benchmark=benchmark, source=source, warnings=warnings)


def load_fundamentals(ticker: str) -> Dict[str, object]:
    demo = {
        "ticker": ticker, "company_name": ticker, "sector": "Technology / Demo", "market_cap": 2_500_000_000_000,
        "trailing_pe": 28.5, "forward_pe": 24.1, "price_to_book": 14.2, "profit_margin": 0.24,
        "revenue_growth": 0.06, "debt_to_equity": 1.6, "free_cash_flow": 85_000_000_000,
        "dividend_yield": 0.005, "analyst_target_price": None, "source": "demo"
    }
    if yf is None:
        return demo
    try:
        info = yf.Ticker(ticker).info or {}
        return {
            "ticker": ticker,
            "company_name": info.get("longName") or info.get("shortName") or ticker,
            "sector": info.get("sector") or "Unavailable",
            "industry": info.get("industry") or "Unavailable",
            "market_cap": info.get("marketCap"),
            "trailing_pe": info.get("trailingPE"),
            "forward_pe": info.get("forwardPE"),
            "price_to_book": info.get("priceToBook"),
            "profit_margin": info.get("profitMargins"),
            "revenue_growth": info.get("revenueGrowth"),
            "debt_to_equity": info.get("debtToEquity"),
            "free_cash_flow": info.get("freeCashflow"),
            "dividend_yield": info.get("dividendYield"),
            "analyst_target_price": info.get("targetMeanPrice"),
            "recommendation": info.get("recommendationKey"),
            "source": "yfinance",
        }
    except Exception:
        return demo


def load_news(ticker: str, limit: int = 12) -> List[Dict[str, object]]:
    if yf is None:
        return []
    try:
        news = yf.Ticker(ticker).news or []
        rows = []
        for item in news[:limit]:
            content = item.get("content", item) if isinstance(item, dict) else {}
            rows.append({
                "title": content.get("title") or item.get("title") or "Untitled headline",
                "publisher": content.get("provider", {}).get("displayName") if isinstance(content.get("provider"), dict) else item.get("publisher", "Unavailable"),
                "link": content.get("canonicalUrl", {}).get("url") if isinstance(content.get("canonicalUrl"), dict) else item.get("link"),
                "published": content.get("pubDate") or item.get("providerPublishTime"),
                "summary": content.get("summary") or item.get("summary") or "",
            })
        return rows
    except Exception:
        return []
