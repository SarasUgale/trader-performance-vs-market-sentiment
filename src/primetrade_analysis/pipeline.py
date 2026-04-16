from __future__ import annotations

from pathlib import Path

import pandas as pd

from .analysis import (
    ensure_directory,
    export_table,
    plot_behavior_comparison,
    plot_cumulative_pnl,
    plot_pnl_distribution,
    plot_segment_heatmap,
    plot_sentiment_timeline,
)
from .cleaning import (
    clean_sentiment_data,
    clean_trades_data,
    describe_missing_values,
    invalid_trade_rows,
    overlap_window,
    summarize_quality,
)
from .dashboard_data import save_dashboard_data
from .features import (
    add_account_segments,
    add_rolling_account_metrics,
    add_trade_features,
    aggregate_daily_metrics,
    build_insight_frames,
    merge_with_sentiment,
    summarize_segment_behavior,
    summarize_sentiment_performance,
)
from .load_data import load_sentiment_raw, load_trades_raw
from .modeling import run_profitability_baseline, save_model_outputs


def build_quality_tables(
    sentiment: pd.DataFrame, trades: pd.DataFrame
) -> dict[str, pd.DataFrame]:
    overlap = overlap_window(trades, sentiment)
    overlap_frame = pd.DataFrame(
        {
            "overlap_start": [overlap["start"]],
            "overlap_end": [overlap["end"]],
            "invalid_trade_rows": [len(invalid_trade_rows(trades))],
            "timestamp_date_mismatch_rows": [
                int(trades["timestamp_date_mismatch"].sum())
            ],
        }
    )
    quality_overview = pd.concat(
        [
            summarize_quality(sentiment, "sentiment"),
            summarize_quality(trades, "trades"),
        ],
        ignore_index=True,
    )
    return {
        "quality_overview": quality_overview,
        "sentiment_missing": describe_missing_values(sentiment),
        "trade_missing": describe_missing_values(trades),
        "overlap_window": overlap_frame,
    }


def generate_strategy_recommendations(merged: pd.DataFrame) -> pd.DataFrame:
    subset = merged[merged["sentiment_binary"].isin(["Fear", "Greed"])].copy()
    size_summary = (
        subset.groupby(["size_segment", "sentiment_binary"])
        .agg(
            avg_pnl=("daily_closed_pnl", "mean"),
            avg_drawdown=("drawdown_proxy", "mean"),
            avg_trade_count=("trade_count", "mean"),
            avg_trade_size_usd=("avg_trade_size_usd", "mean"),
        )
        .reset_index()
    )
    frequency_summary = (
        subset.groupby(["frequency_segment", "sentiment_binary"])
        .agg(
            avg_pnl=("daily_closed_pnl", "mean"),
            avg_win_rate=("win_rate", "mean"),
            avg_trade_count=("trade_count", "mean"),
        )
        .reset_index()
    )

    high_size_fear = size_summary[
        (size_summary["size_segment"] == "high_size")
        & (size_summary["sentiment_binary"] == "Fear")
    ].iloc[0]
    high_size_greed = size_summary[
        (size_summary["size_segment"] == "high_size")
        & (size_summary["sentiment_binary"] == "Greed")
    ].iloc[0]
    frequent_greed = frequency_summary[
        (frequency_summary["frequency_segment"] == "frequent")
        & (frequency_summary["sentiment_binary"] == "Greed")
    ].iloc[0]
    infrequent_greed = frequency_summary[
        (frequency_summary["frequency_segment"] == "infrequent")
        & (frequency_summary["sentiment_binary"] == "Greed")
    ].iloc[0]

    recommendations = [
        {
            "strategy_id": "S1",
            "rule_of_thumb": (
                "During Fear days, reduce trade size for high-size traders and "
                "focus on fewer, higher-conviction setups when drawdown is elevated."
            ),
            "target_segment": "high_size traders",
            "market_regime": "Fear",
            "decision": "Reduce risk concentration instead of scaling up with the crowd.",
            "why_this_rule": (
                f"High-size traders earn about {high_size_fear['avg_pnl']:.0f} average "
                f"daily PnL on Fear days versus {high_size_greed['avg_pnl']:.0f} on Greed "
                f"days, but they also carry very large drawdown proxy "
                f"({high_size_fear['avg_drawdown']:.0f})."
            ),
            "evidence_table": "outputs/tables/size_segment_summary.csv",
            "evidence_chart": "outputs/figures/size_segment_heatmap.png",
        },
        {
            "strategy_id": "S2",
            "rule_of_thumb": (
                "During Greed days, avoid overtrading; let frequent traders keep normal "
                "activity, but require stricter stop-losses and tighter setup selection "
                "for everyone else."
            ),
            "target_segment": "frequent vs infrequent traders",
            "market_regime": "Greed",
            "decision": "Do not assume more trades will automatically improve profitability.",
            "why_this_rule": (
                f"On Greed days, frequent traders keep a stronger win rate "
                f"({frequent_greed['avg_win_rate']:.3f}) than infrequent traders "
                f"({infrequent_greed['avg_win_rate']:.3f}), even though frequent traders "
                f"already trade far more often ({frequent_greed['avg_trade_count']:.1f} "
                f"vs {infrequent_greed['avg_trade_count']:.1f})."
            ),
            "evidence_table": "outputs/tables/frequency_segment_summary.csv",
            "evidence_chart": "outputs/figures/behavior_comparison_fear_vs_greed.png",
        },
    ]

    return pd.DataFrame(recommendations)


def generate_key_insights(merged: pd.DataFrame) -> pd.DataFrame:
    subset = merged[merged["sentiment_binary"].isin(["Fear", "Greed"])].copy()
    sentiment_summary = (
        subset.groupby("sentiment_binary")
        .agg(
            avg_pnl=("daily_closed_pnl", "mean"),
            median_pnl=("daily_closed_pnl", "median"),
            avg_win_rate=("win_rate", "mean"),
            avg_trade_count=("trade_count", "mean"),
            avg_trade_size_usd=("avg_trade_size_usd", "mean"),
            avg_long_ratio=("long_ratio", "mean"),
            avg_short_ratio=("short_ratio", "mean"),
        )
        .reset_index()
        .set_index("sentiment_binary")
    )
    size_summary = (
        subset.groupby(["size_segment", "sentiment_binary"])["daily_closed_pnl"]
        .mean()
        .unstack("sentiment_binary")
    )
    frequency_summary = (
        subset.groupby(["frequency_segment", "sentiment_binary"])
        .agg(avg_pnl=("daily_closed_pnl", "mean"), avg_win_rate=("win_rate", "mean"))
        .reset_index()
    )
    consistency_summary = (
        subset.groupby(["consistency_segment", "sentiment_binary"])
        .agg(
            avg_pnl=("daily_closed_pnl", "mean"),
            avg_trade_size_usd=("avg_trade_size_usd", "mean"),
        )
        .reset_index()
    )

    fear = sentiment_summary.loc["Fear"]
    greed = sentiment_summary.loc["Greed"]
    frequent_greed = frequency_summary[
        (frequency_summary["frequency_segment"] == "frequent")
        & (frequency_summary["sentiment_binary"] == "Greed")
    ].iloc[0]
    infrequent_greed = frequency_summary[
        (frequency_summary["frequency_segment"] == "infrequent")
        & (frequency_summary["sentiment_binary"] == "Greed")
    ].iloc[0]
    consistent_fear = consistency_summary[
        (consistency_summary["consistency_segment"] == "consistent_winner")
        & (consistency_summary["sentiment_binary"] == "Fear")
    ].iloc[0]
    inconsistent_fear = consistency_summary[
        (consistency_summary["consistency_segment"] == "inconsistent")
        & (consistency_summary["sentiment_binary"] == "Fear")
    ].iloc[0]

    insights = [
        {
            "insight_id": "I1",
            "insight_statement": "Trade frequency increases during Fear, but the typical trader-day is more profitable during Greed.",
            "why_non_obvious": (
                "Fear raises average trade count and size, yet median PnL is still higher on Greed days."
            ),
            "evidence_numbers": (
                f"Fear trade_count={fear['avg_trade_count']:.1f}, Greed trade_count={greed['avg_trade_count']:.1f}; "
                f"Fear median_pnl={fear['median_pnl']:.1f}, Greed median_pnl={greed['median_pnl']:.1f}."
            ),
            "evidence_table": "outputs/tables/sentiment_summary_binary.csv",
            "evidence_chart": "outputs/figures/pnl_distribution_fear_vs_greed.png",
        },
        {
            "insight_id": "I2",
            "insight_statement": "High-size traders are much more regime-sensitive than low-size traders.",
            "why_non_obvious": (
                "Large traders outperform strongly during Fear, while smaller traders do better during Greed."
            ),
            "evidence_numbers": (
                f"High-size Fear pnl={size_summary.loc['high_size', 'Fear']:.0f}, "
                f"High-size Greed pnl={size_summary.loc['high_size', 'Greed']:.0f}; "
                f"Low-size Fear pnl={size_summary.loc['low_size', 'Fear']:.0f}, "
                f"Low-size Greed pnl={size_summary.loc['low_size', 'Greed']:.0f}."
            ),
            "evidence_table": "outputs/tables/size_segment_summary.csv",
            "evidence_chart": "outputs/figures/size_segment_heatmap.png",
        },
        {
            "insight_id": "I3",
            "insight_statement": "Frequent traders keep their edge during Greed, while infrequent traders do not gain much from trading less.",
            "why_non_obvious": (
                "Greed does not reward inactivity by default; disciplined frequent traders still post the best win rates."
            ),
            "evidence_numbers": (
                f"Frequent Greed win_rate={frequent_greed['avg_win_rate']:.3f}, "
                f"Infrequent Greed win_rate={infrequent_greed['avg_win_rate']:.3f}; "
                f"Frequent Greed pnl={frequent_greed['avg_pnl']:.0f}, "
                f"Infrequent Greed pnl={infrequent_greed['avg_pnl']:.0f}."
            ),
            "evidence_table": "outputs/tables/frequency_segment_summary.csv",
            "evidence_chart": "outputs/figures/behavior_comparison_fear_vs_greed.png",
        },
        {
            "insight_id": "I4",
            "insight_statement": "Winners reduce risk better during Fear, while inconsistent traders take larger size without a matching win-rate edge.",
            "why_non_obvious": (
                "Consistent winners keep better outcomes on Fear days even though inconsistent traders still trade relatively large size."
            ),
            "evidence_numbers": (
                f"Consistent Fear pnl={consistent_fear['avg_pnl']:.0f}, "
                f"Inconsistent Fear pnl={inconsistent_fear['avg_pnl']:.0f}; "
                f"Consistent Fear avg_size={consistent_fear['avg_trade_size_usd']:.0f}, "
                f"Inconsistent Fear avg_size={inconsistent_fear['avg_trade_size_usd']:.0f}."
            ),
            "evidence_table": "outputs/tables/consistency_segment_summary.csv",
            "evidence_chart": "outputs/figures/top_accounts_cumulative_pnl.png",
        },
    ]

    return pd.DataFrame(insights)


def run_full_analysis(
    sentiment_path: str | Path | None = None,
    trades_path: str | Path | None = None,
    output_root: str | Path = "outputs",
) -> dict[str, object]:
    output_root = Path(output_root)
    figures_dir = ensure_directory(output_root / "figures")
    tables_dir = ensure_directory(output_root / "tables")
    dashboard_dir = ensure_directory(output_root / "dashboard")

    raw_sentiment = load_sentiment_raw(sentiment_path)
    raw_trades = load_trades_raw(trades_path)

    sentiment = clean_sentiment_data(raw_sentiment)
    trades = clean_trades_data(raw_trades)
    trades_with_features = add_trade_features(trades)
    daily_account = aggregate_daily_metrics(trades_with_features)
    daily_account = add_rolling_account_metrics(daily_account)
    daily_account, account_level = add_account_segments(daily_account)
    merged = merge_with_sentiment(daily_account, sentiment)

    quality_tables = build_quality_tables(sentiment, trades)
    sentiment_summary_5class = summarize_sentiment_performance(
        merged, binary_only=False
    )
    sentiment_summary_binary = summarize_sentiment_performance(merged, binary_only=True)
    size_segment_summary = summarize_segment_behavior(merged, "size_segment")
    frequency_segment_summary = summarize_segment_behavior(merged, "frequency_segment")
    consistency_segment_summary = summarize_segment_behavior(
        merged, "consistency_segment"
    )
    size_insight_frame = build_insight_frames(merged, "size_segment")
    key_insights = generate_key_insights(merged)
    strategy_rules = generate_strategy_recommendations(merged)
    model_metrics, model_predictions = run_profitability_baseline(merged)

    export_table(sentiment_summary_5class, tables_dir / "sentiment_summary_5class.csv")
    export_table(sentiment_summary_binary, tables_dir / "sentiment_summary_binary.csv")
    export_table(size_segment_summary, tables_dir / "size_segment_summary.csv")
    export_table(
        frequency_segment_summary, tables_dir / "frequency_segment_summary.csv"
    )
    export_table(
        consistency_segment_summary, tables_dir / "consistency_segment_summary.csv"
    )
    export_table(size_insight_frame, tables_dir / "size_segment_insights.csv")
    export_table(key_insights, tables_dir / "key_insights.csv")
    export_table(strategy_rules, tables_dir / "strategy_recommendations.csv")
    export_table(model_metrics, tables_dir / "model_metrics.csv")
    for name, table in quality_tables.items():
        export_table(table, tables_dir / f"{name}.csv")

    plot_sentiment_timeline(sentiment, figures_dir / "sentiment_timeline.png")
    plot_pnl_distribution(merged, figures_dir / "pnl_distribution_fear_vs_greed.png")
    plot_behavior_comparison(
        merged, figures_dir / "behavior_comparison_fear_vs_greed.png"
    )
    plot_segment_heatmap(
        size_segment_summary, "size_segment", figures_dir / "size_segment_heatmap.png"
    )
    plot_cumulative_pnl(
        merged, account_level, figures_dir / "top_accounts_cumulative_pnl.png"
    )

    dashboard_paths = save_dashboard_data(merged, account_level, dashboard_dir)
    model_paths = save_model_outputs(
        model_metrics, model_predictions, output_root / "model"
    )

    return {
        "sentiment": sentiment,
        "trades": trades,
        "daily_account": daily_account,
        "account_level": account_level,
        "merged": merged,
        "quality_tables": quality_tables,
        "sentiment_summary_5class": sentiment_summary_5class,
        "sentiment_summary_binary": sentiment_summary_binary,
        "size_segment_summary": size_segment_summary,
        "frequency_segment_summary": frequency_segment_summary,
        "consistency_segment_summary": consistency_segment_summary,
        "key_insights": key_insights,
        "strategy_rules": strategy_rules,
        "model_metrics": model_metrics,
        "model_predictions": model_predictions,
        "dashboard_paths": dashboard_paths,
        "model_paths": model_paths,
    }
