from __future__ import annotations

import pandas as pd
import streamlit as st
import plotly.express as px

from src.data_loader import load_market_bundle, load_fundamentals, load_news, COMMODITIES
from src.analytics import technical_indicators, risk_metrics, valuation_score, trading_signal, news_sentiment
from src.backtesting import run_strategies, monte_carlo
from src.llm_engine import llm_commentary
from src.reporting import build_markdown_report, markdown_to_simple_pdf_bytes
from src.viz import price_chart, drawdown_chart, correlation_heatmap, equity_chart, monte_carlo_chart

st.set_page_config(page_title="AI Trading & Research Copilot", page_icon="📈", layout="wide")

CSS = """
<style>
.block-container {padding-top: 1.5rem;}
.metric-card {border: 1px solid rgba(125,125,125,.25); border-radius: 16px; padding: 16px; background: rgba(125,125,125,.05);}
.small-muted {color: #888; font-size: 0.88rem;}
.big-title {font-size: 2.2rem; font-weight: 800; margin-bottom: 0.1rem;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


def fmt_pct(x):
    try:
        return f"{float(x):.2%}"
    except Exception:
        return "N/A"


def fmt_num(x):
    try:
        return f"{float(x):,.2f}"
    except Exception:
        return "N/A"


def safe_df(df: pd.DataFrame, **kwargs):
    try:
        st.dataframe(df, width="stretch", **kwargs)
    except TypeError:
        st.dataframe(df, use_container_width=True, **kwargs)


def safe_plotly_chart(fig):
    try:
        st.plotly_chart(fig, width="stretch")
    except TypeError:
        safe_plotly_chart(fig)


st.markdown('<div class="big-title">AI Trading & Research Copilot</div>', unsafe_allow_html=True)
st.markdown("Institutional-style research assistant combining market data, fundamentals, risk, backtesting, commodities, sentiment and optional LLM commentary.")

with st.sidebar:
    st.header("Research Input")
    prompt = st.text_area(
        "Ask the copilot",
        value="Analyze Apple and propose a strategy",
        height=110,
        help="Examples: Analyze Apple / Compare Nvidia and Microsoft / Build a risk view on oil and gold",
    )
    period = st.selectbox("Historical window", ["1y", "2y", "5y", "10y"], index=2)
    extra = st.text_input("Extra tickers, comma-separated", value="MSFT,NVDA,GLD,TLT")
    llm_model = st.text_input("LLM model name", value="gpt-4o-mini")
    st.caption("LLM is optional. Add OPENAI_API_KEY in your environment to activate it. Without a key, the app uses a robust rules-based analyst note.")
    run = st.button("Run full research", type="primary")

if "last_prompt" not in st.session_state or run:
    st.session_state.last_prompt = prompt

extras = [x.strip() for x in extra.split(",") if x.strip()]
bundle = load_market_bundle(st.session_state.last_prompt, extras, period=period)
prices = bundle.prices
primary = bundle.primary
benchmark = bundle.benchmark
fund = load_fundamentals(primary)
ind = technical_indicators(prices[primary])
risk = risk_metrics(prices[primary], prices[benchmark] if benchmark in prices else None)
val = valuation_score(fund)
signal = trading_signal(ind, risk, val["score"])
llm_text, llm_source = llm_commentary(primary, fund, risk, signal, st.session_state.last_prompt, model=llm_model)
report_md = build_markdown_report(primary, fund, risk, signal, llm_text, bundle.source)
report_pdf = markdown_to_simple_pdf_bytes(report_md)

if bundle.warnings:
    for w in bundle.warnings:
        st.warning(w)

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Primary Asset", primary)
k2.metric("Copilot Stance", signal["action"])
k3.metric("Fundamental Score", f"{val['score']}/100")
k4.metric("Annual Volatility", fmt_pct(risk.get("Annual Volatility")))
k5.metric("Max Drawdown", fmt_pct(risk.get("Max Drawdown")))

st.divider()

tabs = st.tabs([
    "Copilot Report", "Market Dashboard", "Fundamentals", "Risk Engine", "Backtesting",
    "News & Sentiment", "Commodities", "Scenario Lab", "Downloads"
])

with tabs[0]:
    c1, c2 = st.columns([1.2, 0.8])
    with c1:
        st.subheader("Analyst Commentary")
        st.caption(f"Commentary source: {llm_source}")
        st.markdown(llm_text)
    with c2:
        st.subheader("Signal Breakdown")
        breakdown = pd.DataFrame([
            ["Trend", signal["trend"]], ["Momentum", signal["momentum"]], ["Valuation", signal["valuation"]],
            ["Risk", signal["risk_comment"]], ["RSI", signal["rsi_comment"]]
        ], columns=["Dimension", "Assessment"])
        safe_df(breakdown, hide_index=True)
        st.markdown("### Fundamental notes")
        for note in val["notes"]:
            st.write("- " + note)
        st.info("Educational project only. This is not financial advice.")

with tabs[1]:
    safe_plotly_chart(price_chart(ind, primary))
    c1, c2 = st.columns(2)
    with c1:
        safe_plotly_chart(drawdown_chart(ind, primary))
    with c2:
        normalized = prices / prices.iloc[0] * 100
        fig = px.line(normalized, title="Normalized Performance Base 100")
        fig.update_layout(height=330, margin=dict(l=10, r=10, t=50, b=10))
        safe_plotly_chart(fig)

with tabs[2]:
    st.subheader("Fundamental Snapshot")
    items = {
        "Company": fund.get("company_name"), "Sector": fund.get("sector"), "Industry": fund.get("industry"),
        "Market Cap": fmt_num(fund.get("market_cap")), "Trailing P/E": fmt_num(fund.get("trailing_pe")),
        "Forward P/E": fmt_num(fund.get("forward_pe")), "Price/Book": fmt_num(fund.get("price_to_book")),
        "Profit Margin": fmt_pct(fund.get("profit_margin")), "Revenue Growth": fmt_pct(fund.get("revenue_growth")),
        "Debt/Equity": fmt_num(fund.get("debt_to_equity")), "Free Cash Flow": fmt_num(fund.get("free_cash_flow")),
        "Dividend Yield": fmt_pct(fund.get("dividend_yield")), "Analyst Target": fmt_num(fund.get("analyst_target_price")),
        "Recommendation": fund.get("recommendation", "N/A"), "Data Source": fund.get("source")
    }
    fund_df = pd.DataFrame(items.items(), columns=["Metric", "Value"])
    safe_df(fund_df, hide_index=True)

with tabs[3]:
    st.subheader("Risk Engine")
    risk_df = pd.DataFrame(risk.items(), columns=["Metric", "Value"])
    safe_df(risk_df, hide_index=True)
    safe_plotly_chart(correlation_heatmap(prices))

with tabs[4]:
    st.subheader("Multi-Strategy Backtesting")
    equity, stats = run_strategies(prices[primary])
    safe_plotly_chart(equity_chart(equity))
    display_stats = stats.copy()
    for col in ["Total Return", "Annual Return", "Annual Volatility", "Max Drawdown", "Hit Rate"]:
        if col in display_stats:
            display_stats[col] = display_stats[col].map(lambda x: f"{x:.2%}" if pd.notna(x) else "N/A")
    if "Sharpe" in display_stats:
        display_stats["Sharpe"] = display_stats["Sharpe"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
    safe_df(display_stats, hide_index=False)

with tabs[5]:
    st.subheader("News & Sentiment")
    news = load_news(primary)
    if news:
        ns = news_sentiment(news)
        avg_sent = ns["sentiment"].mean() if "sentiment" in ns else 0
        st.metric("Average headline sentiment", f"{avg_sent:.2f}")
        safe_df(ns[["title", "publisher", "sentiment", "link"]], hide_index=True)
    else:
        st.info("No live news available from yfinance in this environment. The rest of the research engine remains functional.")

with tabs[6]:
    st.subheader("Commodity Dashboard")
    c_bundle = load_market_bundle("commodities", COMMODITIES, period=period)
    comm = c_bundle.prices
    norm = comm / comm.iloc[0] * 100
    fig = px.line(norm, title="Commodity Performance Base 100")
    fig.update_layout(height=430, margin=dict(l=10, r=10, t=50, b=10))
    safe_plotly_chart(fig)
    safe_plotly_chart(correlation_heatmap(comm))

with tabs[7]:
    st.subheader("Scenario Lab")
    shock = st.slider("Instant price shock", -30, 30, -10, 1)
    shocked_price = risk.get("Last Price", prices[primary].iloc[-1]) * (1 + shock / 100)
    st.metric("Shocked price", fmt_num(shocked_price), delta=f"{shock}%")
    paths = monte_carlo(prices[primary], days=252, sims=500)
    safe_plotly_chart(monte_carlo_chart(paths))
    terminal = paths.iloc[-1]
    c1, c2, c3 = st.columns(3)
    c1.metric("MC Median 1Y", fmt_num(terminal.median()))
    c2.metric("MC 5th Percentile", fmt_num(terminal.quantile(0.05)))
    c3.metric("MC 95th Percentile", fmt_num(terminal.quantile(0.95)))

with tabs[8]:
    st.subheader("Export Research Report")
    st.download_button("Download Markdown report", data=report_md, file_name=f"{primary}_research_report.md", mime="text/markdown")
    st.download_button("Download PDF report", data=report_pdf, file_name=f"{primary}_research_report.pdf", mime="application/pdf")
    st.markdown("### Report preview")
    st.markdown(report_md)
