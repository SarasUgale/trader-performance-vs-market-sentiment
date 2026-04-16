from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from primetrade_analysis import run_full_analysis


def main() -> None:
    outputs = run_full_analysis(output_root=PROJECT_ROOT / "outputs")

    merged = outputs["merged"]
    print("Analysis complete.")
    print(f"Merged account-day rows: {len(merged):,}")
    print(f"Date range: {merged['date'].min().date()} -> {merged['date'].max().date()}")


if __name__ == "__main__":
    main()
