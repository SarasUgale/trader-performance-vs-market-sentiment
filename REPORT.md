# Primetrade.ai Assignment Write-Up

## Objective

This project analyzes how Bitcoin market sentiment relates to Hyperliquid trader behavior and realized performance. The goal is to identify regime-dependent trader patterns that can be translated into practical trading rules.

## Data Preparation

- Sentiment dataset: `2,644` rows, `6` columns, no missing cells, no duplicate rows
- Trade dataset: `211,224` rows, `22` columns after cleaning and feature preparation, no duplicate rows
- Overlap window used for merged analysis: `2023-05-01` to `2025-05-01`
- Final merged dataset: `2,340` account-day observations across `32` accounts

Important cleaning note:
- The Unix timestamp field in the trade file is stored in rounded scientific notation and loses day-level precision.
- Because of that, `Timestamp IST` was used as the canonical event timestamp for daily alignment, while the Unix field was retained only for validation checks.

## Methodology

1. Standardized both datasets to a consistent schema and parsed all timestamps.
2. Aggregated Hyperliquid trades to daily `account x date` granularity.
3. Built performance metrics:
   - `daily_closed_pnl`
   - `win_rate`
   - `drawdown_proxy`
   - `pnl_volatility_proxy`
4. Built behavior metrics:
   - `trade_count`
   - `avg_trade_size_usd`
   - buy/sell ratios
   - long/short ratios
5. Collapsed sentiment into both:
   - 5-class labels
   - binary `Fear` vs `Greed` labels for strict comparison
6. Segmented traders into:
   - `high_size` vs `low_size`
   - `frequent` vs `infrequent`
   - `consistent_winner` vs `inconsistent`

## Findings

### 1. Trade frequency increases during Fear, but the typical trader-day is more profitable during Greed

- Average trade count rises from `76.9` on Greed days to `105.4` on Fear days
- Average trade size rises from about `$5.95k` to `$8.53k`
- But median daily closed PnL is higher on Greed days: `265.2` vs `122.7`

Interpretation:
Fear does not reduce activity. Traders become more active and trade larger size during fearful regimes, but that does not translate into better typical day-to-day outcomes.

### 2. High-size traders are much more regime-sensitive than low-size traders

- `high_size` traders:
  - Fear: `8.76k` mean daily PnL
  - Greed: `4.75k`
- `low_size` traders:
  - Fear: `1.70k`
  - Greed: `3.67k`

Interpretation:
There is no single best response to sentiment. Large, aggressive-risk traders benefit most from Fear regimes, while smaller traders perform better in Greed.

### 3. Frequent traders keep their edge during Greed, while infrequent traders do not gain much from trading less

- Frequent traders on Greed days:
  - win rate: `43.3%`
  - mean daily PnL: `4.81k`
- Infrequent traders on Greed days:
  - win rate: `30.7%`
  - mean daily PnL: `3.61k`

Interpretation:
Greed does not automatically reward doing fewer trades. The better outcome comes from disciplined, repeatable execution rather than simple inactivity.

### 4. Winners reduce risk better during Fear, while inconsistent traders still use large size without a matching edge

- Consistent winners on Fear days:
  - mean daily PnL: `6.53k`
  - average trade size: `$8.13k`
- Inconsistent traders on Fear days:
  - mean daily PnL: `3.00k`
  - average trade size: `$9.18k`

Interpretation:
The weaker group is not trading smaller. They are taking large risk without converting it into comparable performance.

## Actionable Strategy Ideas

### Rule 1

During Fear days, reduce trade size for high-size traders and focus on fewer, higher-conviction setups when drawdown is elevated.

Why:
- High-size traders post about `8.76k` average daily PnL on Fear days versus `4.75k` on Greed days.
- They also carry the largest drawdown burden, so Fear is profitable but unstable.

### Rule 2

During Greed days, avoid overtrading. Let frequent traders keep normal activity, but require stricter stop-losses and tighter setup selection for everyone else.

Why:
- Frequent traders still keep the strongest Greed-day win rate (`43.3%`) even while trading much more often.
- Infrequent traders do not gain enough edge from lower activity alone, so simply trading less is not the answer.

## Bonus

A lightweight logistic-regression baseline was added to predict whether an account will have a positive PnL day next time it appears.

- Test accuracy: `0.670`
- Test ROC-AUC: `0.655`

This is not meant to be production-grade, but it confirms that sentiment and behavior features contain predictive signal.
