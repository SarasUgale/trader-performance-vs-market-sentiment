from __future__ import annotations

import re

import numpy as np
import pandas as pd

SENTIMENT_BINARY_MAP = {
    "Extreme Fear": "Fear",
    "Fear": "Fear",
    "Neutral": "Neutral",
    "Greed": "Greed",
    "Extreme Greed": "Greed",
}


def to_snake_case(value: str) -> str:
    cleaned = re.sub(r"[^0-9a-zA-Z]+", "_", value.strip())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned.lower()


def standardize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    renamed = frame.copy()
    renamed.columns = [to_snake_case(column) for column in renamed.columns]
    return renamed


def parse_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def clean_sentiment_data(raw: pd.DataFrame) -> pd.DataFrame:
    frame = standardize_columns(raw)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    frame["value"] = parse_numeric(frame["value"])
    frame["timestamp"] = parse_numeric(frame["timestamp"])
    frame["sentiment_5class"] = frame["classification"].astype("string").str.strip()
    frame["sentiment_binary"] = (
        frame["sentiment_5class"].map(SENTIMENT_BINARY_MAP).astype("string")
    )
    frame = (
        frame.sort_values("date")
        .drop_duplicates(subset=["date"], keep="last")
        .reset_index(drop=True)
    )
    return frame


def normalize_direction(direction: pd.Series) -> pd.Series:
    normalized = direction.astype("string").fillna("Unknown").str.strip()
    replacements = {
        "Buy": "Open Long",
        "Sell": "Open Short",
    }
    normalized = normalized.replace(replacements)
    return normalized


def clean_trades_data(raw: pd.DataFrame) -> pd.DataFrame:
    frame = standardize_columns(raw)
    numeric_columns = [
        "execution_price",
        "size_tokens",
        "size_usd",
        "start_position",
        "closed_pnl",
        "fee",
        "timestamp",
    ]
    for column in numeric_columns:
        frame[column] = parse_numeric(frame[column])

    frame["timestamp_ist_dt"] = pd.to_datetime(
        frame["timestamp_ist"],
        format="%d-%m-%Y %H:%M",
        errors="coerce",
    )
    frame["timestamp_dt"] = pd.to_datetime(
        frame["timestamp"], unit="ms", utc=True, errors="coerce"
    )
    frame["timestamp_dt_naive"] = frame["timestamp_dt"].dt.tz_convert(None)
    frame["event_dt"] = frame["timestamp_ist_dt"].combine_first(
        frame["timestamp_dt_naive"]
    )
    frame["date"] = frame["event_dt"].dt.normalize()
    frame["side"] = frame["side"].astype("string").str.upper().str.strip()
    frame["direction"] = normalize_direction(frame["direction"])
    frame["crossed"] = (
        frame["crossed"]
        .astype("string")
        .str.upper()
        .map({"TRUE": True, "FALSE": False})
    )
    frame["account"] = frame["account"].astype("string").str.lower().str.strip()
    frame["coin"] = frame["coin"].astype("string").str.strip()
    frame["timestamp_date_mismatch"] = (
        frame["timestamp_ist_dt"].notna()
        & frame["timestamp_dt_naive"].notna()
        & (
            frame["timestamp_ist_dt"].dt.normalize()
            != frame["timestamp_dt_naive"].dt.normalize()
        )
    )
    frame = frame.sort_values(["event_dt", "account"], kind="stable").reset_index(
        drop=True
    )
    return frame


def summarize_quality(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    summary = pd.DataFrame(
        {
            "dataset": [name],
            "rows": [len(frame)],
            "columns": [frame.shape[1]],
            "duplicate_rows": [int(frame.duplicated().sum())],
            "missing_cells": [int(frame.isna().sum().sum())],
        }
    )
    return summary


def invalid_trade_rows(frame: pd.DataFrame) -> pd.DataFrame:
    invalid_mask = (
        frame["event_dt"].isna()
        | frame["date"].isna()
        | (frame["size_usd"].fillna(0) < 0)
        | (frame["fee"].fillna(0) < 0)
    )
    return frame.loc[invalid_mask].copy()


def overlap_window(
    trades: pd.DataFrame, sentiment: pd.DataFrame
) -> dict[str, pd.Timestamp | None]:
    trade_dates = trades["date"].dropna()
    sentiment_dates = sentiment["date"].dropna()
    if trade_dates.empty or sentiment_dates.empty:
        return {"start": None, "end": None}

    start = max(trade_dates.min(), sentiment_dates.min())
    end = min(trade_dates.max(), sentiment_dates.max())
    return {"start": start, "end": end}


def describe_missing_values(frame: pd.DataFrame) -> pd.DataFrame:
    missing = frame.isna().sum().rename("missing_count").reset_index()
    missing.columns = ["column", "missing_count"]
    missing["missing_pct"] = np.where(
        len(frame) > 0, missing["missing_count"] / len(frame), 0.0
    )
    return missing.sort_values(
        ["missing_count", "column"], ascending=[False, True]
    ).reset_index(drop=True)
