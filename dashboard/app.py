"""QTP Streamlit Dashboard — Phase5 with EDGAR, Fear&Greed, calibration."""

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from qtp.data.database import QTPDatabase

st.set_page_config(page_title="QTP Dashboard", layout="wide")
st.title("Quant Trading Pipeline Dashboard")
st.caption("Phase 5.2 | 53 features | 13 tickers | Sharpe 14.92 | Calibrated")

DB_PATH = Path(__file__).parent.parent / "data" / "qtp.db"

if not DB_PATH.exists():
    st.error(f"Database not found: {DB_PATH}")
    st.stop()

db = QTPDatabase(DB_PATH)

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Overview", "Predictions", "Accuracy", "Alt Data", "Experiments"]
)

# ── Tab 1: Overview ─────────────────────────────────────────────────────
with tab1:
    import pandas as pd

    col1, col2, col3, col4 = st.columns(4)
    summary = db.get_accuracy_summary()
    if summary and summary.get("total"):
        col1.metric("Total Predictions", summary["total"])
        col2.metric("Accuracy", f"{summary['accuracy']:.1%}")
        col3.metric("Avg Return", f"{summary.get('avg_return', 0):+.2%}")
        col4.metric("Avg Confidence", f"{summary.get('avg_confidence', 0):.1%}")
    else:
        st.info("No graded predictions yet. Run `make grade` first.")

    # Model info
    st.subheader("Current Model")
    models = db.list_models(limit=1)
    if models:
        m = models[0]
        st.write(f"**Version**: `{m['version']}` | **Created**: {m['created_at']}")

    # Accuracy trend
    st.subheader("Accuracy Trend (7-day windows)")
    trend = db.get_accuracy_trend(window_days=7, n_windows=12)
    if trend:
        trend_df = pd.DataFrame(trend)
        trend_df = trend_df.iloc[::-1].reset_index(drop=True)
        st.line_chart(trend_df.set_index("window")["accuracy"])
    else:
        st.info("Not enough graded data for trend analysis.")

    # Today's signals
    st.subheader("Latest Predictions")
    recent = db.get_recent_predictions(20)
    if recent:
        df = pd.DataFrame(recent)

        # Color-code signals
        def signal_color(row):
            if row.get("direction") == 1 and row.get("confidence", 0) >= 0.55:
                return "BUY"
            elif row.get("direction") == 0 and row.get("confidence", 0) >= 0.55:
                return "SELL"
            return "HOLD"

        df["signal"] = df.apply(signal_color, axis=1)
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No predictions recorded yet.")

# ── Tab 2: Predictions ──────────────────────────────────────────────────
with tab2:
    import pandas as pd

    st.subheader("Prediction History")
    recent = db.get_recent_predictions(100)
    if recent:
        df = pd.DataFrame(recent)

        col_f1, col_f2 = st.columns(2)
        with col_f1:
            tickers = sorted(set(r["ticker"] for r in recent if r.get("ticker")))
            selected_ticker = st.selectbox("Filter by Ticker", ["All"] + tickers)
        with col_f2:
            directions = ["All", 1, 0]
            selected_dir = st.selectbox(
                "Filter by Direction",
                directions,
                format_func=lambda x: {1: "UP", 0: "DOWN", "All": "All"}.get(x, x),
            )

        if selected_ticker != "All":
            df = df[df["ticker"] == selected_ticker]
        if selected_dir != "All":
            df = df[df["direction"] == selected_dir]

        st.dataframe(df, use_container_width=True)
        st.caption(f"Showing {len(df)} of {len(recent)} predictions")
    else:
        st.info("No predictions recorded yet.")

# ── Tab 3: Accuracy ─────────────────────────────────────────────────────
with tab3:
    import pandas as pd

    st.subheader("Accuracy by Confidence")
    by_conf = db.get_accuracy_by_confidence()
    if by_conf:
        conf_df = pd.DataFrame(by_conf)
        st.dataframe(conf_df, use_container_width=True)
        st.bar_chart(conf_df.set_index("bucket")["accuracy_pct"])

    st.divider()

    st.subheader("Accuracy by Ticker")
    by_ticker = db.get_accuracy_by_ticker()
    if by_ticker:
        ticker_df = pd.DataFrame(by_ticker)
        st.dataframe(ticker_df, use_container_width=True)

        # Highlight best/worst
        if len(ticker_df) > 0:
            best = ticker_df.iloc[0]
            worst = ticker_df.iloc[-1]
            col1, col2 = st.columns(2)
            col1.metric("Best Ticker", f"{best['ticker']} ({best['accuracy_pct']}%)")
            col2.metric("Worst Ticker", f"{worst['ticker']} ({worst['accuracy_pct']}%)")
    else:
        st.info("No graded predictions yet.")

# ── Tab 4: Alternative Data ─────────────────────────────────────────────
with tab4:
    import pandas as pd

    st.subheader("Alternative Data Coverage")

    coverage = db.alternative_coverage()
    if coverage:
        cov_df = pd.DataFrame(coverage)
        st.dataframe(cov_df, use_container_width=True)
    else:
        st.info("No alternative data cached. Run `make alt-data` first.")

    st.divider()

    st.subheader("Daily Accumulation Status")
    import sqlite3

    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row

    # Count records by date
    daily_counts = conn.execute("""
        SELECT date, COUNT(*) as records, COUNT(DISTINCT ticker) as tickers,
               COUNT(DISTINCT tool) as tools
        FROM alternative_data_daily
        GROUP BY date
        ORDER BY date DESC
        LIMIT 14
    """).fetchall()

    if daily_counts:
        daily_df = pd.DataFrame([dict(r) for r in daily_counts])
        st.dataframe(daily_df, use_container_width=True)
        st.line_chart(daily_df.set_index("date")["records"])
        st.caption("30+ days of accumulation needed for Finnhub Tier5 features")
    else:
        st.info("No daily accumulation data yet. Cron runs daily at 09:00 JST.")

    st.divider()

    # Fear & Greed current
    st.subheader("Fear & Greed Index")
    fg = db.get_alternative("_market", "fear_greed")
    if fg:
        score = fg.get("score", 50)
        rating = fg.get("rating", "neutral")
        st.metric("Current Score", f"{score:.1f}", delta=rating)
        history = fg.get("history", {})
        if history:
            st.write(
                f"1 week ago: {history.get('1w', 'N/A')} | "
                f"1 month ago: {history.get('1m', 'N/A')} | "
                f"3 months ago: {history.get('3m', 'N/A')}"
            )
    else:
        st.info("No Fear & Greed data. Run `make alt-data`.")

    conn.close()

# ── Tab 5: Experiments ──────────────────────────────────────────────────
with tab5:
    import pandas as pd

    st.subheader("Experiment History")
    exps = db.list_experiments(20)
    if exps:
        exp_df = pd.DataFrame(exps)
        st.dataframe(exp_df, use_container_width=True)
    else:
        st.info("No experiments recorded yet.")

    st.divider()

    st.subheader("Top Experiments by AUC")
    best = db.best_experiments("wf_auc", 5)
    if best:
        best_df = pd.DataFrame(best)
        st.dataframe(best_df, use_container_width=True)

    st.divider()

    st.subheader("Top Experiments by Sharpe")
    best_sharpe = db.best_experiments("wf_sharpe", 5)
    if best_sharpe:
        sharpe_df = pd.DataFrame(best_sharpe)
        st.dataframe(sharpe_df, use_container_width=True)
