"""QTP Streamlit Dashboard — predictions, accuracy, and experiment monitoring."""

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from qtp.data.database import QTPDatabase

st.set_page_config(page_title="QTP Dashboard", layout="wide")
st.title("Quant Trading Pipeline Dashboard")

DB_PATH = Path(__file__).parent.parent / "data" / "qtp.db"

if not DB_PATH.exists():
    st.error(f"Database not found: {DB_PATH}")
    st.stop()

db = QTPDatabase(DB_PATH)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Predictions", "Accuracy", "Experiments"])

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

    # Accuracy trend
    st.subheader("Accuracy Trend (7-day windows)")
    trend = db.get_accuracy_trend(window_days=7, n_windows=8)
    if trend:
        trend_df = pd.DataFrame(trend)
        trend_df = trend_df.iloc[::-1].reset_index(drop=True)  # oldest first for chart
        st.line_chart(trend_df.set_index("window")["accuracy"])
    else:
        st.info("Not enough graded data for trend analysis.")

    # Recent predictions
    st.subheader("Recent Predictions")
    recent = db.get_recent_predictions(20)
    if recent:
        df = pd.DataFrame(recent)
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

        # Filters
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            tickers = sorted(set(r["ticker"] for r in recent if r.get("ticker")))
            selected_ticker = st.selectbox("Filter by Ticker", ["All"] + tickers)
        with col_f2:
            directions = sorted(set(r["direction"] for r in recent if r.get("direction")))
            selected_dir = st.selectbox("Filter by Direction", ["All"] + directions)

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

    st.subheader("Accuracy Analysis")

    # By confidence bucket
    by_conf = db.get_accuracy_by_confidence()
    if by_conf:
        st.write("**By Confidence Bucket**")
        conf_df = pd.DataFrame(by_conf)
        st.dataframe(conf_df, use_container_width=True)
        st.bar_chart(conf_df.set_index("bucket")["accuracy_pct"])
    else:
        st.info("No graded predictions yet.")

    st.divider()

    # By ticker
    by_ticker = db.get_accuracy_by_ticker()
    if by_ticker:
        st.write("**By Ticker**")
        ticker_df = pd.DataFrame(by_ticker)
        st.dataframe(ticker_df, use_container_width=True)
        st.bar_chart(ticker_df.set_index("ticker")["accuracy_pct"])
    else:
        st.info("No graded predictions yet.")

# ── Tab 4: Experiments ──────────────────────────────────────────────────
with tab4:
    import pandas as pd

    st.subheader("Experiment History")
    exps = db.list_experiments(20)
    if exps:
        exp_df = pd.DataFrame(exps)
        st.dataframe(exp_df, use_container_width=True)
    else:
        st.info("No experiments recorded yet.")

    st.divider()

    # Best experiments
    st.subheader("Top Experiments by AUC")
    best = db.best_experiments("wf_auc", 5)
    if best:
        best_df = pd.DataFrame(best)
        st.dataframe(best_df, use_container_width=True)
    else:
        st.info("No experiments with AUC scores yet.")
