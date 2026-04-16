from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

MODEL_FEATURES = [
    "sentiment_binary",
    "trade_count",
    "avg_trade_size_usd",
    "win_rate",
    "long_ratio",
    "short_ratio",
    "drawdown_proxy",
    "pnl_volatility_proxy",
    "size_segment",
    "frequency_segment",
    "consistency_segment",
]


def prepare_model_data(merged: pd.DataFrame) -> pd.DataFrame:
    frame = merged.sort_values(["account", "date"]).copy()
    frame["next_day_positive_pnl"] = frame.groupby("account")["positive_pnl_day"].shift(
        -1
    )
    frame = frame[frame["sentiment_binary"].isin(["Fear", "Greed"])].copy()
    frame = frame.dropna(subset=["next_day_positive_pnl"])
    return frame


def run_profitability_baseline(
    merged: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = prepare_model_data(merged)
    X = frame[MODEL_FEATURES]
    y = frame["next_day_positive_pnl"].astype(int)

    numeric_features = [
        "trade_count",
        "avg_trade_size_usd",
        "win_rate",
        "long_ratio",
        "short_ratio",
        "drawdown_proxy",
        "pnl_volatility_proxy",
    ]
    categorical_features = [
        "sentiment_binary",
        "size_segment",
        "frequency_segment",
        "consistency_segment",
    ]

    split_date = frame["date"].quantile(0.8)
    train_mask = frame["date"] <= split_date
    test_mask = frame["date"] > split_date

    pipeline = Pipeline(
        steps=[
            (
                "preprocess",
                ColumnTransformer(
                    transformers=[
                        (
                            "numeric",
                            Pipeline(
                                [
                                    ("imputer", SimpleImputer(strategy="median")),
                                    ("scaler", StandardScaler()),
                                ]
                            ),
                            numeric_features,
                        ),
                        (
                            "categorical",
                            Pipeline(
                                [
                                    (
                                        "imputer",
                                        SimpleImputer(strategy="most_frequent"),
                                    ),
                                    ("encoder", OneHotEncoder(handle_unknown="ignore")),
                                ]
                            ),
                            categorical_features,
                        ),
                    ]
                ),
            ),
            ("model", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )

    pipeline.fit(X.loc[train_mask], y.loc[train_mask])
    test_scores = pipeline.predict_proba(X.loc[test_mask])[:, 1]
    test_pred = (test_scores >= 0.5).astype(int)

    metrics = pd.DataFrame(
        {
            "metric": [
                "train_rows",
                "test_rows",
                "test_accuracy",
                "test_roc_auc",
                "test_base_rate",
            ],
            "value": [
                int(train_mask.sum()),
                int(test_mask.sum()),
                float(accuracy_score(y.loc[test_mask], test_pred)),
                float(roc_auc_score(y.loc[test_mask], test_scores)),
                float(y.loc[test_mask].mean()),
            ],
        }
    )

    prediction_frame = frame.loc[
        test_mask, ["account", "date", "sentiment_binary"]
    ].copy()
    prediction_frame["actual_next_day_positive_pnl"] = y.loc[test_mask].to_numpy()
    prediction_frame["predicted_probability_positive_pnl"] = test_scores
    prediction_frame["predicted_class"] = test_pred
    return metrics, prediction_frame


def save_model_outputs(
    metrics: pd.DataFrame, predictions: pd.DataFrame, output_dir: str | Path
) -> dict[str, Path]:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    metrics_path = output_root / "model_metrics.csv"
    predictions_path = output_root / "model_predictions.csv"
    metrics.to_csv(metrics_path, index=False)
    predictions.to_csv(predictions_path, index=False)
    return {"model_metrics": metrics_path, "model_predictions": predictions_path}
