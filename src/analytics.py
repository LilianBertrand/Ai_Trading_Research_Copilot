from __future__ import annotations

from typing import Dict
import numpy as np
import pandas as pd


def returns(prices: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    return prices.pct_change().replace([np.inf, -np.inf], np.nan).dropna(how="all")


def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0).rolling(window).mean()
    down = (-delta.clip(upper=0)).rolling(window).mean()
    rs = up / down.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def macd(series: pd.Series) -> pd.DataFrame:
    ema12 = series.ewm(span=12, adjust=False).mean()
    ema26 = series.ewm(span=26, adjust=False).mean()
    line = ema12 - ema26
    signal = line.ewm(span=9, adjust=False).mean()
    return pd.DataFrame({"MACD": line, "Signal": signal, "Histogram": line - signal})


def technical_indicators(price: pd.Series) -> pd.DataFrame:
    out = pd.DataFrame(index=price.index)
    out["Close"] = price
    out["SMA_20"] = price.rolling(20).mean()
    out["SMA_50"] = price.rolling(50).mean()
    out["SMA_200"] = price.rolling(200).mean()
    out["RSI_14"] = rsi(price)
    m = macd(price)
    out = out.join(m)
    out["Volatility_21D"] = price.pct_change().rolling(21).std() * np.sqrt(252)
    out["Drawdown"] = price / price.cummax() - 1
    return out


def risk_metrics(asset: pd.Series, benchmark: pd.Series | None = None, rf: float = 0.02) -> Dict[str, float]:
    ret = returns(asset).dropna()
    if ret.empty:
        return {}
    ann_ret = (1 + ret).prod() ** (252 / len(ret)) - 1
    ann_vol = ret.std() * np.sqrt(252)
    downside = ret[ret < 0].std() * np.sqrt(252)
    sharpe = (ann_ret - rf) / ann_vol if ann_vol and ann_vol > 0 else np.nan
    sortino = (ann_ret - rf) / downside if downside and downside > 0 else np.nan
    dd = asset / asset.cummax() - 1
    var95 = ret.quantile(0.05)
    cvar95 = ret[ret <= var95].mean() if len(ret[ret <= var95]) else np.nan
    beta = np.nan
    corr = np.nan
    if benchmark is not None and not benchmark.equals(asset):
        br = returns(benchmark).dropna()
        joined = pd.concat([ret, br], axis=1).dropna()
        if joined.shape[0] > 30 and joined.iloc[:, 1].var() > 0:
            beta = joined.iloc[:, 0].cov(joined.iloc[:, 1]) / joined.iloc[:, 1].var()
            corr = joined.iloc[:, 0].corr(joined.iloc[:, 1])
    return {
        "Annual Return": ann_ret,
        "Annual Volatility": ann_vol,
        "Sharpe Ratio": sharpe,
        "Sortino Ratio": sortino,
        "Max Drawdown": dd.min(),
        "VaR 95% Daily": var95,
        "CVaR 95% Daily": cvar95,
        "Beta vs Benchmark": beta,
        "Correlation vs Benchmark": corr,
        "Last Price": float(asset.dropna().iloc[-1]),
    }


def valuation_score(fund: Dict[str, object]) -> Dict[str, object]:
    pe = fund.get("trailing_pe") or fund.get("forward_pe")
    margin = fund.get("profit_margin")
    growth = fund.get("revenue_growth")
    debt = fund.get("debt_to_equity")
    score = 50
    notes = []
    if pe is not None:
        if pe < 18: score += 12; notes.append("valuation multiple appears moderate")
        elif pe > 35: score -= 10; notes.append("valuation multiple appears demanding")
    if margin is not None:
        if margin > 0.20: score += 10; notes.append("profitability is strong")
        elif margin < 0.05: score -= 8; notes.append("profitability is weak")
    if growth is not None:
        if growth > 0.08: score += 8; notes.append("revenue growth is supportive")
        elif growth < 0: score -= 8; notes.append("revenue growth is negative")
    if debt is not None:
        d = debt / 100 if debt and debt > 10 else debt
        if d and d > 2.5: score -= 7; notes.append("balance sheet leverage is elevated")
        elif d and d < 1.0: score += 5; notes.append("leverage looks contained")
    score = max(0, min(100, score))
    return {"score": score, "notes": notes or ["fundamental data is limited"]}


def trading_signal(ind: pd.DataFrame, risk: Dict[str, float], fundamental_score: int) -> Dict[str, str]:
    latest = ind.dropna().iloc[-1]
    close = latest["Close"]
    sma50 = latest.get("SMA_50", np.nan)
    sma200 = latest.get("SMA_200", np.nan)
    rsi_latest = latest.get("RSI_14", np.nan)
    macd_hist = latest.get("Histogram", np.nan)
    risk_ok = risk.get("Annual Volatility", 0.99) < 0.40 and risk.get("Max Drawdown", -1) > -0.45
    trend = "bullish" if close > sma50 > sma200 else "bearish" if close < sma50 < sma200 else "mixed"
    momentum = "positive" if macd_hist > 0 else "negative"
    valuation = "supportive" if fundamental_score >= 60 else "stretched" if fundamental_score < 45 else "neutral"

    if trend == "bullish" and momentum == "positive" and risk_ok and fundamental_score >= 50:
        action = "Constructive / Buy-on-weakness"
    elif trend == "bearish" or not risk_ok:
        action = "Defensive / Wait or reduce exposure"
    else:
        action = "Neutral / Monitor key levels"

    risk_comment = "risk profile is acceptable" if risk_ok else "risk profile is elevated"
    rsi_comment = "overbought" if rsi_latest > 70 else "oversold" if rsi_latest < 30 else "not extreme"
    return {
        "action": action,
        "trend": trend,
        "momentum": momentum,
        "valuation": valuation,
        "risk_comment": risk_comment,
        "rsi_comment": rsi_comment,
    }


def sentiment_score(text: str) -> float:
    positive = {"beat", "beats", "growth", "strong", "upgrade", "surge", "record", "profit", "positive", "bullish", "gain", "higher", "outperform"}
    negative = {"miss", "weak", "downgrade", "fall", "lawsuit", "risk", "loss", "negative", "bearish", "drop", "lower", "underperform", "concern"}
    words = set(str(text).lower().replace("-", " ").split())
    pos = len(words & positive)
    neg = len(words & negative)
    total = pos + neg
    return 0.0 if total == 0 else (pos - neg) / total


def news_sentiment(news: list[dict]) -> pd.DataFrame:
    rows = []
    for n in news:
        txt = f"{n.get('title','')} {n.get('summary','')}"
        rows.append({**n, "sentiment": sentiment_score(txt)})
    return pd.DataFrame(rows)
