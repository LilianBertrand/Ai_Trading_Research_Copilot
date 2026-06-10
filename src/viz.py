from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def price_chart(ind: pd.DataFrame, ticker: str):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ind.index, y=ind["Close"], name=ticker, line=dict(width=2)))
    for col in ["SMA_50", "SMA_200"]:
        if col in ind:
            fig.add_trace(go.Scatter(x=ind.index, y=ind[col], name=col, line=dict(width=1)))
    fig.update_layout(title=f"{ticker} Price & Moving Averages", height=430, margin=dict(l=10, r=10, t=50, b=10))
    return fig


def drawdown_chart(ind: pd.DataFrame, ticker: str):
    fig = px.area(ind, x=ind.index, y="Drawdown", title=f"{ticker} Drawdown")
    fig.update_layout(height=330, margin=dict(l=10, r=10, t=50, b=10), yaxis_tickformat=".0%")
    return fig


def correlation_heatmap(prices: pd.DataFrame):
    corr = prices.pct_change().corr()
    fig = px.imshow(corr, text_auto=".2f", aspect="auto", title="Cross-Asset Correlation Matrix")
    fig.update_layout(height=440, margin=dict(l=10, r=10, t=50, b=10))
    return fig


def equity_chart(equity: pd.DataFrame):
    fig = px.line(equity, title="Backtesting Equity Curves")
    fig.update_layout(height=430, margin=dict(l=10, r=10, t=50, b=10), yaxis_title="Growth of $1")
    return fig


def monte_carlo_chart(paths: pd.DataFrame):
    sample = paths.iloc[:, :80]
    fig = px.line(sample, title="Monte Carlo Simulation Paths")
    fig.update_layout(height=430, showlegend=False, margin=dict(l=10, r=10, t=50, b=10), yaxis_title="Simulated price")
    return fig
