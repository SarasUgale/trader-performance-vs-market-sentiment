from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from primetrade_analysis import run_full_analysis

st.set_page_config(
    page_title="Primetrade Sentiment Dashboard",
    page_icon=":bar_chart:",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def load_dashboard_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    dashboard_dir = PROJECT_ROOT / "outputs" / "dashboard"
    merged_path = dashboard_dir / "merged_account_daily.csv"
    account_path = dashboard_dir / "account_segments.csv"

    if not merged_path.exists() or not account_path.exists():
        run_full_analysis(output_root=PROJECT_ROOT / "outputs")

    merged = pd.read_csv(merged_path, parse_dates=["date"])
    account = pd.read_csv(account_path)
    return merged, account


def metric_card(label: str, value: float, format_string: str = "{:,.2f}") -> None:
    st.metric(label, format_string.format(value))


merged, account_level = load_dashboard_frames()

st.title("Trader Performance vs Market Sentiment")
st.caption(
    "Interactive view of Hyperliquid trader behavior conditioned on Bitcoin Fear & Greed sentiment."
)

with st.sidebar:
    st.header("Filters")
    sentiment_mode = st.radio("Sentiment view", ["Binary", "5-class"], index=0)
    if sentiment_mode == "Binary":
        available_sentiments = ["Fear", "Greed"]
        selected_sentiments = st.multiselect(
            "Binary sentiment",
            options=available_sentiments,
            default=available_sentiments,
        )
        working = merged[merged["sentiment_binary"].isin(selected_sentiments)].copy()
        color_column = "sentiment_binary"
    else:
        available_sentiments = sorted(
            merged["sentiment_5class"].dropna().unique().tolist()
        )
        selected_sentiments = st.multiselect(
            "5-class sentiment",
            options=available_sentiments,
            default=available_sentiments,
        )
        working = merged[merged["sentiment_5class"].isin(selected_sentiments)].copy()
        color_column = "sentiment_5class"

    segment_dimension = st.selectbox(
        "Account segment",
        options=["size_segment", "frequency_segment", "consistency_segment"],
        index=0,
    )
    available_segments = sorted(working[segment_dimension].dropna().unique().tolist())
    selected_segments = st.multiselect(
        "Segment values",
        options=available_segments,
        default=available_segments,
    )
    working = working[working[segment_dimension].isin(selected_segments)].copy()

    min_date = working["date"].min().date()
    max_date = working["date"].max().date()
    date_range = st.date_input(
        "Date range", value=(min_date, max_date), min_value=min_date, max_value=max_date
    )
    if len(date_range) == 2:
        start_date, end_date = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
        working = working[working["date"].between(start_date, end_date)].copy()

if working.empty:
    st.warning("No rows match the current filter set.")
    st.stop()

kpi_1, kpi_2, kpi_3, kpi_4 = st.columns(4)
with kpi_1:
    metric_card("Avg daily PnL", working["daily_closed_pnl"].mean())
with kpi_2:
    metric_card("Avg win rate", working["win_rate"].mean(), "{:.2%}")
with kpi_3:
    metric_card("Avg trade count", working["trade_count"].mean())
with kpi_4:
    metric_card("Avg trade size (USD)", working["avg_trade_size_usd"].mean())

trend = working.groupby(["date", color_column], as_index=False).agg(
    avg_daily_closed_pnl=("daily_closed_pnl", "mean")
)
fig_trend = px.line(
    trend,
    x="date",
    y="avg_daily_closed_pnl",
    color=color_column,
    title="Average daily closed PnL over time",
)
st.plotly_chart(fig_trend, use_container_width=True)

behavior = working.groupby(color_column, as_index=False).agg(
    avg_trade_count=("trade_count", "mean"),
    avg_trade_size_usd=("avg_trade_size_usd", "mean"),
    avg_drawdown_proxy=("drawdown_proxy", "mean"),
)
behavior_long = behavior.melt(
    id_vars=color_column, var_name="metric", value_name="value"
)
fig_behavior = px.bar(
    behavior_long,
    x="metric",
    y="value",
    color=color_column,
    barmode="group",
    title="Behavior metrics by sentiment",
)
st.plotly_chart(fig_behavior, use_container_width=True)

segment_perf = working.groupby([segment_dimension, color_column], as_index=False).agg(
    avg_daily_closed_pnl=("daily_closed_pnl", "mean")
)
fig_segment = px.bar(
    segment_perf,
    x=segment_dimension,
    y="avg_daily_closed_pnl",
    color=color_column,
    barmode="group",
    title="Segment-level PnL comparison",
)
st.plotly_chart(fig_segment, use_container_width=True)

with st.expander("Filtered data preview", expanded=False):
    preview_cols = [
        "account",
        "date",
        "daily_closed_pnl",
        "win_rate",
        "trade_count",
        "avg_trade_size_usd",
        "sentiment_5class",
        "sentiment_binary",
        segment_dimension,
    ]
    st.dataframe(working[preview_cols].head(100), use_container_width=True)
