from __future__ import annotations

from pathlib import Path

import pandas as pd


def save_dashboard_data(
    merged: pd.DataFrame, account_level: pd.DataFrame, output_dir: str | Path
) -> dict[str, Path]:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    merged_path = output_root / "merged_account_daily.csv"
    account_path = output_root / "account_segments.csv"

    merged.to_csv(merged_path, index=False)
    account_level.to_csv(account_path, index=False)

    return {
        "merged_account_daily": merged_path,
        "account_segments": account_path,
    }
