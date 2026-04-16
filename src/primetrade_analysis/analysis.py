from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", palette="deep")


def ensure_directory(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def export_table(frame: pd.DataFrame, path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)


def plot_sentiment_timeline(sentiment: pd.DataFrame, path: str | Path) -> None:
    figure, axis = plt.subplots(figsize=(12, 5))
    counts = (
        sentiment.assign(quarter=sentiment["date"].dt.to_period("Q").dt.to_timestamp())
        .groupby(["quarter", "sentiment_5class"])
        .size()
        .rename("count")
        .reset_index()
    )
    sns.lineplot(data=counts, x="quarter", y="count", hue="sentiment_5class", ax=axis)
    axis.set_title("Quarterly sentiment distribution over time")
    axis.set_xlabel("Date")
    axis.set_ylabel("Observation count")
    figure.tight_layout()
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def plot_pnl_distribution(merged: pd.DataFrame, path: str | Path) -> None:
    figure, axis = plt.subplots(figsize=(10, 5))
    subset = merged[merged["sentiment_binary"].isin(["Fear", "Greed"])].copy()
    sns.violinplot(
        data=subset,
        x="sentiment_binary",
        y="daily_closed_pnl",
        inner="quartile",
        ax=axis,
    )
    axis.set_title("Daily closed PnL distribution on Fear vs Greed days")
    axis.set_xlabel("Binary sentiment")
    axis.set_ylabel("Daily closed PnL")
    figure.tight_layout()
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def plot_behavior_comparison(merged: pd.DataFrame, path: str | Path) -> None:
    subset = (
        merged[merged["sentiment_binary"].isin(["Fear", "Greed"])]
        .groupby("sentiment_binary")
        .agg(
            avg_trade_count=("trade_count", "mean"),
            avg_trade_size_usd=("avg_trade_size_usd", "mean"),
            avg_long_ratio=("long_ratio", "mean"),
        )
        .reset_index()
        .melt(id_vars="sentiment_binary", var_name="metric", value_name="value")
    )
    figure, axis = plt.subplots(figsize=(10, 5))
    sns.barplot(data=subset, x="metric", y="value", hue="sentiment_binary", ax=axis)
    axis.set_title("Behavior differences by binary sentiment")
    axis.set_xlabel("Metric")
    axis.set_ylabel("Average value")
    axis.tick_params(axis="x", rotation=20)
    figure.tight_layout()
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def plot_segment_heatmap(
    segment_summary: pd.DataFrame, segment_column: str, path: str | Path
) -> None:
    pivoted = segment_summary.pivot(
        index=segment_column, columns="sentiment_binary", values="avg_pnl"
    )
    figure, axis = plt.subplots(figsize=(8, 4))
    sns.heatmap(pivoted, annot=True, fmt=".2f", cmap="RdYlGn", center=0, ax=axis)
    axis.set_title(f"Average daily PnL by {segment_column} and sentiment")
    figure.tight_layout()
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def plot_cumulative_pnl(
    merged: pd.DataFrame, account_level: pd.DataFrame, path: str | Path
) -> None:
    top_accounts = account_level.sort_values("total_daily_pnl", ascending=False).head(
        5
    )["account"]
    subset = merged[merged["account"].isin(top_accounts)].copy()
    subset = subset.sort_values(["account", "date"])
    figure, axis = plt.subplots(figsize=(12, 5))
    sns.lineplot(
        data=subset, x="date", y="cumulative_closed_pnl", hue="account", ax=axis
    )
    axis.set_title("Cumulative closed PnL for top accounts")
    axis.set_xlabel("Date")
    axis.set_ylabel("Cumulative closed PnL")
    axis.legend(title="Account", bbox_to_anchor=(1.02, 1), loc="upper left")
    figure.tight_layout()
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)
