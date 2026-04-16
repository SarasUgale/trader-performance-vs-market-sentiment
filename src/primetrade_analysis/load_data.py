from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

SENTIMENT_FILENAME = "fear_greed_index.csv"
TRADES_FILENAME = "historical_data.csv"


def default_data_search_paths() -> list[Path]:
    home = Path.home()
    return [
        Path.cwd() / "data",
        home / "Downloads",
        home / "Desktop",
    ]


def resolve_data_file(
    filename: str, search_paths: Iterable[Path] | None = None
) -> Path:
    paths = list(search_paths or default_data_search_paths())
    for root in paths:
        candidate = root / filename
        if candidate.exists():
            return candidate
    searched = ", ".join(str(path) for path in paths)
    raise FileNotFoundError(f"Could not find {filename!r} in: {searched}")


def load_sentiment_raw(path: str | Path | None = None) -> pd.DataFrame:
    file_path = Path(path) if path else resolve_data_file(SENTIMENT_FILENAME)
    return pd.read_csv(file_path)


def load_trades_raw(path: str | Path | None = None) -> pd.DataFrame:
    file_path = Path(path) if path else resolve_data_file(TRADES_FILENAME)
    dtype_map = {
        "Account": "string",
        "Coin": "string",
        "Side": "string",
        "Timestamp IST": "string",
        "Direction": "string",
        "Transaction Hash": "string",
        "Order ID": "string",
        "Crossed": "string",
        "Trade ID": "string",
    }
    return pd.read_csv(file_path, dtype=dtype_map, low_memory=False)
