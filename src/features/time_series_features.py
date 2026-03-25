import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from typing import Dict, List, Tuple


def build_lag_features(series, max_lag: int = 3) -> Dict[str, float]:
    features: Dict[str, float] = {}

    for i in range(1, max_lag + 1):
        features[f"lag_{i}"] = series[-i] if len(series) >= i else np.nan

    return features


def build_stat_features(series) -> Dict[str, float]:
    return {
        "mean": float(np.mean(series)),
        "std": float(np.std(series)) if len(series) > 1 else 0.0,
        "min": float(np.min(series)),
        "max": float(np.max(series)),
    }


def build_trend_features(series) -> Dict[str, float]:
    if len(series) >= 2:
        x = np.arange(len(series)).reshape(-1, 1)
        y = np.array(series, dtype=float)
        model = LinearRegression().fit(x, y)
        trend = float(model.coef_[0])
        last_diff = float(series[-1] - series[-2])
    else:
        trend = 0.0
        last_diff = 0.0

    return {
        "trend": trend,
        "last_diff": last_diff,
        "length": len(series),
    }


def build_features(series, max_lag: int = 3) -> Dict[str, float]:
    features = {}
    features.update(build_lag_features(series, max_lag=max_lag))
    features.update(build_stat_features(series))
    features.update(build_trend_features(series))
    return features


def build_time_series_data(
    df: pd.DataFrame,
    user_col="user_id",
    time_col="week",
    value_col="score",
    max_lag: int = 3,
) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    df = df.copy()
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    df = df.dropna(subset=[user_col, time_col, value_col])
    df = df.sort_values([user_col, time_col], ascending=True)

    rows = []

    for _, user_df in df.groupby(user_col, sort=False):
        values = user_df[value_col].tolist()

        for idx in range(1, len(user_df)):
            history = values[:idx]
            target = values[idx]

            feature_row = build_features(history, max_lag=max_lag)
            feature_row["_target"] = target
            rows.append(feature_row)

    feature_cols = [f"lag_{i}" for i in range(1, max_lag + 1)] + [
        "mean",
        "std",
        "min",
        "max",
        "trend",
        "last_diff",
        "length",
    ]

    if not rows:
        return pd.DataFrame(columns=feature_cols), pd.Series(dtype=float), feature_cols

    feature_df = pd.DataFrame(rows)
    X = feature_df[feature_cols]
    y = feature_df["_target"]

    return X, y, feature_cols
