from __future__ import annotations

import numpy as np
import pandas as pd

LONG_DIRECTIONS = {"Open Long", "Close Short"}
SHORT_DIRECTIONS = {"Open Short", "Close Long"}
OPEN_LONG_DIRECTIONS = {"Open Long"}
OPEN_SHORT_DIRECTIONS = {"Open Short"}
CLOSE_LONG_DIRECTIONS = {"Close Long"}
CLOSE_SHORT_DIRECTIONS = {"Close Short"}


def add_trade_features(trades: pd.DataFrame) -> pd.DataFrame:
    frame = trades.copy()
    frame["abs_trade_size_usd"] = frame["size_usd"].abs()
    frame["signed_notional_usd"] = np.where(
        frame["side"].eq("BUY"), frame["size_usd"], -frame["size_usd"]
    )
    frame["is_profitable_trade"] = (frame["closed_pnl"] > 0).astype(int)
    frame["is_loss_trade"] = (frame["closed_pnl"] < 0).astype(int)
    frame["is_buy"] = frame["side"].eq("BUY").astype(int)
    frame["is_sell"] = frame["side"].eq("SELL").astype(int)
    frame["is_long_bias_trade"] = frame["direction"].isin(LONG_DIRECTIONS).astype(int)
    frame["is_short_bias_trade"] = frame["direction"].isin(SHORT_DIRECTIONS).astype(int)
    frame["is_open_long"] = frame["direction"].isin(OPEN_LONG_DIRECTIONS).astype(int)
    frame["is_open_short"] = frame["direction"].isin(OPEN_SHORT_DIRECTIONS).astype(int)
    frame["is_close_long"] = frame["direction"].isin(CLOSE_LONG_DIRECTIONS).astype(int)
    frame["is_close_short"] = (
        frame["direction"].isin(CLOSE_SHORT_DIRECTIONS).astype(int)
    )
    return frame


def aggregate_daily_metrics(trades: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["account", "date"]
    aggregated = (
        trades.groupby(group_cols, dropna=False)
        .agg(
            daily_closed_pnl=("closed_pnl", "sum"),
            trade_count=("account", "size"),
            avg_trade_size_usd=("abs_trade_size_usd", "mean"),
            median_trade_size_usd=("abs_trade_size_usd", "median"),
            total_notional_usd=("abs_trade_size_usd", "sum"),
            total_signed_notional_usd=("signed_notional_usd", "sum"),
            fees_paid=("fee", "sum"),
            profitable_trade_count=("is_profitable_trade", "sum"),
            loss_trade_count=("is_loss_trade", "sum"),
            buy_count=("is_buy", "sum"),
            sell_count=("is_sell", "sum"),
            long_bias_trade_count=("is_long_bias_trade", "sum"),
            short_bias_trade_count=("is_short_bias_trade", "sum"),
            open_long_count=("is_open_long", "sum"),
            open_short_count=("is_open_short", "sum"),
            close_long_count=("is_close_long", "sum"),
            close_short_count=("is_close_short", "sum"),
            active_coins=("coin", "nunique"),
        )
        .reset_index()
    )

    aggregated["win_rate"] = aggregated["profitable_trade_count"] / aggregated[
        "trade_count"
    ].replace(0, np.nan)
    aggregated["buy_ratio"] = aggregated["buy_count"] / aggregated[
        "trade_count"
    ].replace(0, np.nan)
    aggregated["sell_ratio"] = aggregated["sell_count"] / aggregated[
        "trade_count"
    ].replace(0, np.nan)
    aggregated["long_ratio"] = aggregated["long_bias_trade_count"] / aggregated[
        "trade_count"
    ].replace(0, np.nan)
    aggregated["short_ratio"] = aggregated["short_bias_trade_count"] / aggregated[
        "trade_count"
    ].replace(0, np.nan)
    aggregated["long_short_ratio"] = aggregated["long_bias_trade_count"] / aggregated[
        "short_bias_trade_count"
    ].replace(0, np.nan)
    aggregated["long_open_ratio"] = aggregated["open_long_count"] / aggregated[
        "trade_count"
    ].replace(0, np.nan)
    aggregated["short_open_ratio"] = aggregated["open_short_count"] / aggregated[
        "trade_count"
    ].replace(0, np.nan)
    aggregated["close_long_ratio"] = aggregated["close_long_count"] / aggregated[
        "trade_count"
    ].replace(0, np.nan)
    aggregated["close_short_ratio"] = aggregated["close_short_count"] / aggregated[
        "trade_count"
    ].replace(0, np.nan)
    aggregated["net_positioning_bias"] = aggregated["buy_ratio"].fillna(0) - aggregated[
        "sell_ratio"
    ].fillna(0)
    return aggregated


def add_rolling_account_metrics(
    daily_account: pd.DataFrame, window: int = 7
) -> pd.DataFrame:
    frame = daily_account.sort_values(["account", "date"]).copy()
    grouped = frame.groupby("account", group_keys=False)
    frame["rolling_pnl_mean"] = grouped["daily_closed_pnl"].transform(
        lambda series: series.rolling(window=window, min_periods=2).mean()
    )
    frame["pnl_volatility_proxy"] = grouped["daily_closed_pnl"].transform(
        lambda series: series.rolling(window=window, min_periods=2).std()
    )
    frame["cumulative_closed_pnl"] = grouped["daily_closed_pnl"].cumsum()
    frame["rolling_peak_pnl"] = grouped["cumulative_closed_pnl"].cummax()
    frame["drawdown_proxy"] = frame["rolling_peak_pnl"] - frame["cumulative_closed_pnl"]
    frame["positive_pnl_day"] = (frame["daily_closed_pnl"] > 0).astype(int)
    return frame


def add_account_segments(
    daily_account: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    account_level = (
        daily_account.groupby("account")
        .agg(
            active_days=("date", "nunique"),
            total_daily_pnl=("daily_closed_pnl", "sum"),
            avg_daily_pnl=("daily_closed_pnl", "mean"),
            pnl_std=("daily_closed_pnl", "std"),
            median_daily_notional=("total_notional_usd", "median"),
            median_trade_count=("trade_count", "median"),
            positive_day_share=("positive_pnl_day", "mean"),
            avg_win_rate=("win_rate", "mean"),
            max_drawdown_proxy=("drawdown_proxy", "max"),
        )
        .reset_index()
    )

    size_threshold = account_level["median_daily_notional"].median()
    freq_threshold = account_level["median_trade_count"].median()
    winner_threshold = account_level["positive_day_share"].median()

    account_level["size_segment"] = np.where(
        account_level["median_daily_notional"] >= size_threshold,
        "high_size",
        "low_size",
    )
    account_level["frequency_segment"] = np.where(
        account_level["median_trade_count"] >= freq_threshold,
        "frequent",
        "infrequent",
    )
    account_level["consistency_segment"] = np.where(
        account_level["positive_day_share"] >= winner_threshold,
        "consistent_winner",
        "inconsistent",
    )

    enriched = daily_account.merge(
        account_level, on="account", how="left", suffixes=("", "_account")
    )
    return enriched, account_level


def merge_with_sentiment(
    daily_account: pd.DataFrame, sentiment: pd.DataFrame
) -> pd.DataFrame:
    sentiment_cols = ["date", "value", "sentiment_5class", "sentiment_binary"]
    merged = daily_account.merge(sentiment[sentiment_cols], on="date", how="inner")
    merged = merged.sort_values(["date", "account"]).reset_index(drop=True)
    return merged


def summarize_sentiment_performance(
    merged: pd.DataFrame, binary_only: bool = False
) -> pd.DataFrame:
    frame = merged.copy()
    sentiment_col = "sentiment_binary" if binary_only else "sentiment_5class"
    if binary_only:
        frame = frame[frame["sentiment_binary"].isin(["Fear", "Greed"])].copy()

    summary = (
        frame.groupby(sentiment_col)
        .agg(
            account_days=("account", "size"),
            avg_daily_closed_pnl=("daily_closed_pnl", "mean"),
            median_daily_closed_pnl=("daily_closed_pnl", "median"),
            avg_win_rate=("win_rate", "mean"),
            avg_drawdown_proxy=("drawdown_proxy", "mean"),
            avg_trade_count=("trade_count", "mean"),
            avg_trade_size_usd=("avg_trade_size_usd", "mean"),
            avg_long_ratio=("long_ratio", "mean"),
            avg_short_ratio=("short_ratio", "mean"),
        )
        .reset_index()
        .sort_values(sentiment_col)
    )
    return summary


def summarize_segment_behavior(
    merged: pd.DataFrame, segment_column: str
) -> pd.DataFrame:
    summary = (
        merged.groupby([segment_column, "sentiment_binary"])
        .agg(
            avg_pnl=("daily_closed_pnl", "mean"),
            avg_win_rate=("win_rate", "mean"),
            avg_drawdown_proxy=("drawdown_proxy", "mean"),
            avg_trade_count=("trade_count", "mean"),
            avg_trade_size_usd=("avg_trade_size_usd", "mean"),
        )
        .reset_index()
    )
    return summary


def build_insight_frames(merged: pd.DataFrame, segment_column: str) -> pd.DataFrame:
    frame = summarize_segment_behavior(merged, segment_column)
    pivoted = frame.pivot(
        index=segment_column, columns="sentiment_binary", values="avg_pnl"
    )
    pivoted["fear_minus_greed_pnl"] = pivoted.get("Fear", np.nan) - pivoted.get(
        "Greed", np.nan
    )
    return pivoted.reset_index()
