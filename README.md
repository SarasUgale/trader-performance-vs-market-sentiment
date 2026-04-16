# Primetrade Sentiment vs Trader Performance

This repo implements the Data Science Intern assignment: analyze how Bitcoin market sentiment relates to Hyperliquid trader behavior and performance, then convert the findings into actionable trading rules.


## Executive Summary

- The usable overlap window is `2023-05-01` to `2025-05-01`, producing `2,340` merged account-day observations across `32` trader accounts.
- Fear days are more active than Greed days: average trade count is `105.4` vs `76.9`, and average trade size is about `$8.5k` vs `$6.0k`.
- Mean daily closed PnL is higher on Fear days (`~5.2k`) than Greed days (`~4.1k`), but the median account-day PnL is higher on Greed days (`~265`) than Fear days (`~123`). This suggests Fear regimes create larger upside outliers while Greed regimes are more stable for the typical trader-day.
- Fear days show a stronger long bias (`0.521` long ratio vs `0.456` short ratio), while Greed days tilt slightly short (`0.472` long vs `0.511` short).
- Segment behavior matters:
  - `high_size` traders outperform sharply on Fear days (`8.8k` mean daily PnL) relative to Greed days (`4.8k`)
  - `low_size` traders do better on Greed days (`3.7k`) than Fear days (`1.7k`)
  - `frequent` traders maintain the best win rates in both regimes, especially on Greed days (`43.3%`)

These results support regime-specific rules instead of one generic trading policy.

## Project Structure

```text
.
|-- app.py
|-- notebooks/analysis.ipynb
|-- outputs/
|-- scripts/run_analysis.py
|-- src/primetrade_analysis/
|-- tests/
|-- requirements.txt
`-- pyproject.toml
```

## Setup

1. Create and activate a Python environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
pip install -e .
```

3. Place `fear_greed_index.csv` and `historical_data.csv` in either:
   - `data/`
   - your `~/Downloads` folder

The pipeline already checks both locations.

## How To Run

### Generate analysis outputs

```bash
python scripts/run_analysis.py
```

This exports:
- cleaned tables to `outputs/tables/`
- figures to `outputs/figures/`
- dashboard-ready datasets to `outputs/dashboard/`
- bonus model outputs to `outputs/model/`

### Open the notebook

```bash
jupyter notebook notebooks/analysis.ipynb
```

### Launch the dashboard

```bash
streamlit run app.py
```
## Output
<p align="center">
<img width="48%" alt="image" src="https://github.com/user-attachments/assets/a8172271-90a6-4f82-819c-db2ee24834d3" />
<img width="48%" alt="image" src="https://github.com/user-attachments/assets/c4c43dd7-416d-4a7a-b304-dd24af312274" />
<img width="48%" alt="image" src="https://github.com/user-attachments/assets/50845b27-35e8-458d-b584-117208022dc4" />
</p>


## Methodology

### 1. Data preparation
- Load both datasets with explicit dtype handling to preserve scientific-notation fields.
- Standardize all columns to `snake_case`.
- Parse both trade timestamp fields and use `Timestamp IST` as the canonical event time because the Unix timestamp column is rounded in scientific notation.
- Create per-trade helper features such as signed notional, long/short markers, and absolute trade size.
- Aggregate to `account x date` granularity for sentiment-aware analysis.

### 2. Performance and behavior metrics
- `daily_closed_pnl`
- `win_rate`
- `trade_count`
- `avg_trade_size_usd`
- buy/sell ratios
- long/short positioning ratios
- `pnl_volatility_proxy`
- `drawdown_proxy`

### 3. Trader segmentation
- `size_segment`: `high_size` vs `low_size`
- `frequency_segment`: `frequent` vs `infrequent`
- `consistency_segment`: `consistent_winner` vs `inconsistent`

### 4. Sentiment framing
- 5-class sentiment: `Extreme Fear`, `Fear`, `Neutral`, `Greed`, `Extreme Greed`
- Binary sentiment:
  - `Fear` = `Fear` + `Extreme Fear`
  - `Greed` = `Greed` + `Extreme Greed`
  - `Neutral` is preserved for descriptive analysis but excluded from strict Fear-vs-Greed comparisons

## Key Insights

The current outputs support these final assignment insights:

1. Trade frequency increases during Fear, but the typical trader-day is more profitable during Greed.
   - Average trade count rises from `76.9` on Greed days to `105.4` on Fear days.
   - Average trade size rises from about `$5.95k` to `$8.53k`.
   - Median daily closed PnL is still higher on Greed days: `265.2` vs `122.7`.
   - Interpretation: Fear increases activity, but not the typical account-day payoff.

2. High-size traders are much more regime-sensitive than low-size traders.
   - `high_size` traders perform much better on Fear days (`8.76k`) than Greed days (`4.75k`).
   - `low_size` traders perform better on Greed days (`3.67k`) than Fear days (`1.70k`).
   - Interpretation: the best action depends on trader type, not just market mood.

3. Frequent traders keep their edge during Greed, while infrequent traders do not gain much from trading less.
   - Frequent Greed-day win rate is `43.3%` versus `30.7%` for infrequent traders.
   - Interpretation: Greed rewards disciplined repeatability more than simple inactivity.

4. Winners reduce risk better during Fear, while inconsistent traders still use large size without a matching edge.
   - Consistent winners on Fear days average `6.53k` daily PnL at about `$8.13k` average size.
   - Inconsistent traders on Fear days average only `3.00k` daily PnL despite using about `$9.18k` average size.
   - Interpretation: weaker traders are not under-sizing risk; they are over-sizing it.

## Strategy Recommendations

The exported `outputs/tables/strategy_recommendations.csv` captures the final rule set. The repo currently frames two required recommendations:

1. On Fear days, high-size traders should reduce trade size and directional exposure when drawdown is already elevated.
   - Rationale: this segment is the most profitable in Fear regimes, but it also carries the largest drawdown burden, so risk control matters most when activity expands.
2. On Greed days, avoid overtrading; let frequent traders keep normal activity, but require stricter stop-losses and tighter setup selection for everyone else.
   - Rationale: Greed rewards disciplined repeatability, not automatic increases in activity by weaker traders.

## Bonus Model

The repo includes a simple predictive baseline for next-day positive profitability at the account-day level.

- Model: logistic regression
- Features: sentiment, trade count, trade size, win rate, long/short mix, drawdown proxy, volatility proxy, and segment labels
- Time split: first 80% of dates for training, final 20% for testing
- Current test performance:
  - accuracy: `0.670`
  - ROC-AUC: `0.655`

This is intentionally lightweight, but it shows that sentiment plus trader behavior has usable signal for a basic predictive task.

## Limitations

- The provided trade dataset does not include a true leverage field, so this project uses trade-size and activity proxies instead of claiming leverage analysis.
- `Closed PnL` is treated as the realized performance signal; open mark-to-market exposure is out of scope.
- Some `Direction` values are noisy, so the pipeline normalizes the common categories and safely handles rare residual labels.
- The raw Unix timestamp field loses day-level precision because it is stored in scientific notation, so the project uses `Timestamp IST` for alignment and keeps the Unix field only as a quality check.


