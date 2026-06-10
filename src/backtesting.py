from __future__ import annotations

from typing import Dict
import numpy as np
import pandas as pd
from .analytics import rsi, macd


def _metrics(strategy_returns: pd.Series) -> Dict[str, float]:
    r = strategy_returns.replace([np.inf, -np.inf], np.nan).dropna()
    if r.empty:
        return {}
    equity = (1 + r).cumprod()
    ann_return = equity.iloc[-1] ** (252 / len(r)) - 1
    ann_vol = r.std() * np.sqrt(252)
    sharpe = ann_return / ann_vol if ann_vol > 0 else np.nan
    dd = equity / equity.cummax() - 1
    hit_rate = (r > 0).mean()
    return {
        "Total Return": equity.iloc[-1] - 1,
        "Annual Return": ann_return,
        "Annual Volatility": ann_vol,
        "Sharpe": sharpe,
        "Max Drawdown": dd.min(),
        "Hit Rate": hit_rate,
    }


def run_strategies(price: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    px = price.dropna().copy()
    ret = px.pct_change().fillna(0)
    signals = pd.DataFrame(index=px.index)

    sma_fast = px.rolling(50).mean()
    sma_slow = px.rolling(200).mean()
    signals["SMA Crossover"] = (sma_fast > sma_slow).astype(float)

    r = rsi(px)
    meanrev = pd.Series(0.0, index=px.index)
    meanrev[r < 35] = 1.0
    meanrev[r > 70] = 0.0
    signals["RSI Mean Reversion"] = meanrev.ffill().fillna(0)

    m = macd(px)
    signals["MACD Trend"] = (m["MACD"] > m["Signal"]).astype(float)
    signals["Buy & Hold"] = 1.0

    returns = signals.shift(1).fillna(0).mul(ret, axis=0)
    equity = (1 + returns).cumprod()
    stats = pd.DataFrame({name: _metrics(returns[name]) for name in returns.columns}).T
    return equity, stats


def monte_carlo(price: pd.Series, days: int = 252, sims: int = 500) -> pd.DataFrame:
    px = price.dropna()
    ret = px.pct_change().dropna()
    mu, sigma = ret.mean(), ret.std()
    last = px.iloc[-1]
    rng = np.random.default_rng(7)
    paths = rng.normal(mu, sigma, size=(days, sims))
    prices = last * np.exp(np.cumsum(paths, axis=0))
    idx = pd.bdate_range(start=px.index[-1], periods=days)
    return pd.DataFrame(prices, index=idx)
