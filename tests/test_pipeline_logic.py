from __future__ import annotations

import pandas as pd

from primetrade_analysis.cleaning import SENTIMENT_BINARY_MAP, clean_trades_data
from primetrade_analysis.features import (
    add_rolling_account_metrics,
    add_trade_features,
    aggregate_daily_metrics,
)


def test_scientific_notation_timestamp_parses_to_datetime() -> None:
    raw = pd.DataFrame(
        {
            "Account": ["0xabc"],
            "Coin": ["BTC"],
            "Execution Price": ["100.0"],
            "Size Tokens": ["1.0"],
            "Size USD": ["100.0"],
            "Side": ["BUY"],
            "Timestamp IST": ["02-12-2024 22:50"],
            "Start Position": ["0"],
            "Direction": ["Open Long"],
            "Closed PnL": ["0"],
            "Transaction Hash": ["0xhash"],
            "Order ID": ["123"],
            "Crossed": ["TRUE"],
            "Fee": ["0.1"],
            "Trade ID": ["1.0E+12"],
            "Timestamp": ["1.73E+12"],
        }
    )

    cleaned = clean_trades_data(raw)
    assert cleaned.loc[0, "timestamp_dt"].year == 2024
    assert cleaned.loc[0, "date"].strftime("%Y-%m-%d") == "2024-12-02"
    assert bool(cleaned.loc[0, "timestamp_date_mismatch"]) is True


def test_sentiment_binary_map_keeps_neutral_and_collapses_extremes() -> None:
    assert SENTIMENT_BINARY_MAP["Extreme Fear"] == "Fear"
    assert SENTIMENT_BINARY_MAP["Fear"] == "Fear"
    assert SENTIMENT_BINARY_MAP["Neutral"] == "Neutral"
    assert SENTIMENT_BINARY_MAP["Extreme Greed"] == "Greed"


def test_daily_aggregation_is_account_date_grain() -> None:
    trades = pd.DataFrame(
        {
            "account": ["a", "a", "a"],
            "date": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-02"]),
            "closed_pnl": [10.0, -4.0, 7.0],
            "size_usd": [100.0, 50.0, 30.0],
            "fee": [1.0, 1.0, 1.0],
            "side": ["BUY", "SELL", "BUY"],
            "direction": ["Open Long", "Close Long", "Open Short"],
            "coin": ["BTC", "BTC", "ETH"],
        }
    )

    featured = add_trade_features(trades)
    daily = aggregate_daily_metrics(featured)

    assert len(daily) == 2
    first_day = daily.loc[daily["date"] == pd.Timestamp("2024-01-01")].iloc[0]
    assert first_day["trade_count"] == 2
    assert first_day["daily_closed_pnl"] == 6.0
    assert round(first_day["win_rate"], 4) == 0.5


def test_drawdown_proxy_uses_cumulative_peak_minus_current() -> None:
    daily = pd.DataFrame(
        {
            "account": ["a", "a", "a"],
            "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "daily_closed_pnl": [10.0, -6.0, -2.0],
            "trade_count": [1, 1, 1],
            "avg_trade_size_usd": [100.0, 100.0, 100.0],
            "median_trade_size_usd": [100.0, 100.0, 100.0],
            "total_notional_usd": [100.0, 100.0, 100.0],
            "total_signed_notional_usd": [100.0, -100.0, -100.0],
            "fees_paid": [1.0, 1.0, 1.0],
            "profitable_trade_count": [1, 0, 0],
            "loss_trade_count": [0, 1, 1],
            "buy_count": [1, 0, 0],
            "sell_count": [0, 1, 1],
            "long_bias_trade_count": [1, 0, 0],
            "short_bias_trade_count": [0, 1, 1],
            "open_long_count": [1, 0, 0],
            "open_short_count": [0, 1, 1],
            "close_long_count": [0, 0, 0],
            "close_short_count": [0, 0, 0],
            "active_coins": [1, 1, 1],
            "win_rate": [1.0, 0.0, 0.0],
            "buy_ratio": [1.0, 0.0, 0.0],
            "sell_ratio": [0.0, 1.0, 1.0],
            "long_ratio": [1.0, 0.0, 0.0],
            "short_ratio": [0.0, 1.0, 1.0],
            "long_short_ratio": [None, 0.0, 0.0],
            "long_open_ratio": [1.0, 0.0, 0.0],
            "short_open_ratio": [0.0, 1.0, 1.0],
            "close_long_ratio": [0.0, 0.0, 0.0],
            "close_short_ratio": [0.0, 0.0, 0.0],
            "net_positioning_bias": [1.0, -1.0, -1.0],
        }
    )

    enriched = add_rolling_account_metrics(daily)
    assert list(enriched["cumulative_closed_pnl"]) == [10.0, 4.0, 2.0]
    assert list(enriched["drawdown_proxy"]) == [0.0, 6.0, 8.0]
