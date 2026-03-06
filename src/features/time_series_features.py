# src/features/time_series_features.py

import pandas as pd
import numpy as np


def build_lag_features(
    df: pd.DataFrame,
    user_col: str = "user_id",
    time_col: str = "week",
    value_col: str = "score",
    lags=(1, 2, 3),
):
    """
    构建时间序列 lag 特征

    example
    -------
    score_t-1
    score_t-2
    score_t-3
    """

    df = df.sort_values([user_col, time_col]).copy()

    for lag in lags:
        df[f"{value_col}_lag{lag}"] = df.groupby(user_col)[value_col].shift(lag)

    return df


def build_stat_features(
    df: pd.DataFrame,
    value_col: str = "score",
):
    """
    构建统计特征
    """

    lag_cols = [c for c in df.columns if "lag" in c]

    df["mean_last3"] = df[lag_cols].mean(axis=1)
    df["std_last3"] = df[lag_cols].std(axis=1)
    df["max_last3"] = df[lag_cols].max(axis=1)
    df["min_last3"] = df[lag_cols].min(axis=1)

    return df


def build_trend_features(
    df: pd.DataFrame,
    value_col: str = "score",
):
    """
    构建趋势特征
    """

    if f"{value_col}_lag1" in df.columns and f"{value_col}_lag3" in df.columns:
        df["trend_last3"] = df[f"{value_col}_lag1"] - df[f"{value_col}_lag3"]

    return df


def build_training_dataset(
    df: pd.DataFrame,
    user_col="user_id",
    time_col="week",
    value_col="score",
):
    """
    构建训练数据

    输出:
        X, y
    """

    df = build_lag_features(df, user_col, time_col, value_col)
    df = build_stat_features(df, value_col)
    df = build_trend_features(df, value_col)

    # 删除 lag 不完整的数据
    df = df.dropna()

    feature_cols = [
        c for c in df.columns if "lag" in c or "mean" in c or "std" in c or "trend" in c
    ]

    X = df[feature_cols]
    y = df[value_col]

    return X, y, feature_cols
