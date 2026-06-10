# AI Trading & Research Copilot

A professional Streamlit research platform that combines market data, fundamentals, risk analytics, commodities, backtesting, sentiment analysis, report generation and an optional LLM layer.

The objective is to simulate a lightweight financial analyst assistant. A user can write a natural-language request such as:

> Analyze Apple and propose a strategy

The application then identifies the asset, retrieves market data, computes technical indicators, evaluates risk, checks fundamental metrics, runs several backtests, analyzes news sentiment, generates charts and exports a research report.

---

## Key Features

### Natural-Language Research Input

- Prompt-based workflow
- Automatic ticker detection for common assets
- Additional ticker input for portfolio comparison
- Works with or without an LLM API key

### Optional LLM Layer

- Uses an OpenAI-compatible workflow when `OPENAI_API_KEY` is available
- Falls back to a deterministic analyst-style commentary if no key is provided
- Keeps the project usable for GitHub demos and local testing

### Market Dashboard

- Historical price data through Yahoo Finance via `yfinance`
- Demo-data fallback if live data is unavailable
- Price chart with moving averages
- Normalized performance comparison
- Drawdown visualization

### Technical Indicators

- SMA 20, 50 and 200
- RSI 14
- MACD and signal line
- Rolling volatility
- Drawdown analysis

### Risk Engine

- Annualized return
- Annualized volatility
- Sharpe ratio
- Sortino ratio
- Maximum drawdown
- Historical VaR 95%
- Historical CVaR 95%
- Beta and correlation versus benchmark
- Cross-asset correlation matrix

### Fundamentals Module

- Company name, sector and industry
- Market capitalization
- P/E ratios
- Price-to-book
- Profit margin
- Revenue growth
- Debt-to-equity
- Free cash flow
- Dividend yield
- Analyst target price when available
- Fundamental scoring logic

### News & Sentiment

- Yahoo Finance news headlines when available
- Lightweight sentiment scoring
- Source and link display
- Works even if no news is returned

### Multi-Strategy Backtesting

The platform compares several simple strategies:

- Buy & Hold
- SMA crossover
- RSI mean reversion
- MACD trend following

For each strategy, the app computes:

- Total return
- Annualized return
- Annualized volatility
- Sharpe ratio
- Maximum drawdown
- Hit rate

### Commodities Dashboard

- Gold
- WTI crude oil
- Brent crude oil
- Natural gas
- Copper
- Silver

Includes normalized commodity performance and correlation analysis.

### Scenario Lab

- Instant price shock simulation
- Monte Carlo paths
- Median, 5th percentile and 95th percentile terminal price

### Report Export

- Markdown export
- PDF export without external PDF dependencies
- Analyst commentary
- Fundamentals
- Risk metrics
- Strategy stance
- Educational disclaimer

---

## Project Structure

```text
ai_trading_research_copilot_ultimate_v2_1_fixed/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── app_assets/
└── src/
    ├── __init__.py
    ├── analytics.py
    ├── backtesting.py
    ├── data_loader.py
    ├── llm_engine.py
    ├── reporting.py
    └── viz.py
```

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Run the App

```bash
streamlit run app.py
```

---

## Optional LLM Setup

The app works without a key. To activate LLM commentary, set an environment variable before launching Streamlit:

```bash
export OPENAI_API_KEY="your_api_key_here"
streamlit run app.py
```

If no key is configured, the application automatically uses a rules-based research note.

---

## Example Prompts

```text
Analyze Apple and propose a strategy
```

```text
Compare Nvidia and Microsoft from a risk and momentum perspective
```

```text
Analyze oil, gold and copper and explain the macro risk setup
```

```text
Build a defensive portfolio view with Apple, Microsoft, TLT and gold
```

---

## Important Disclaimer

This project is for educational and portfolio demonstration purposes only. It is not investment advice and should not be used to make real trading or investment decisions.


## V2.1 Fixes

- Improved natural-language ticker parsing. Example: `Analyze Apple and propose a strategy` now resolves to `AAPL`, not `APPLE` or `AND`.
- Removed Streamlit `use_container_width` warnings by using `width="stretch"` with backwards-compatible fallbacks.
- Kept demo-data fallbacks for environments where Yahoo Finance is unavailable.
