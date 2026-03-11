# src/features/time_series_features.py

import pandas as pd
import numpy as np


from typing import Tuple, List


def build_lag_features(
    df: pd.DataFrame,
    user_col: str = "user_id",
    time_col: str = "week",
    value_col: str = "score",
    lags: tuple = (1, 2, 3),
) -> Tuple[pd.DataFrame, List[str]]:
    """
    构建时间序列 lag 特征

    example
    -------
    score_lag1
    score_lag2
    score_lag3

    Returns
    -------
    df : pd.DataFrame
        添加 lag 特征后的 dataframe

    feature_cols : List[str]
        新生成的 lag 特征列名
    """

    df = df.sort_values([user_col, time_col], ascending=True).copy()

    new_cols: List[str] = []

    for lag in lags:
        col_name = f"{value_col}_lag{lag}"
        df[col_name] = df.groupby(user_col)[value_col].shift(lag)
        new_cols.append(col_name)

    return df, new_cols


def build_stat_features(
    df: pd.DataFrame,
    value_col: str = "score",
) -> Tuple[pd.DataFrame, List[str]]:
    """
    构建统计特征

    使用已有 lag 特征计算统计量：
    - mean_last3
    - std_last3
    - max_last3
    - min_last3

    Returns
    -------
    df : pd.DataFrame
        添加统计特征后的 dataframe

    feature_cols : List[str]
        新生成的统计特征列名
    """

    lag_cols = [c for c in df.columns if "lag" in c]

    new_cols: List[str] = []

    df["mean_last3"] = df[lag_cols].mean(axis=1)
    new_cols.append("mean_last3")

    df["std_last3"] = df[lag_cols].std(axis=1)
    new_cols.append("std_last3")

    df["max_last3"] = df[lag_cols].max(axis=1)
    new_cols.append("max_last3")

    df["min_last3"] = df[lag_cols].min(axis=1)
    new_cols.append("min_last3")

    return df, new_cols


def build_trend_features(
    df: pd.DataFrame,
    value_col: str = "score",
) -> Tuple[pd.DataFrame, List[str]]:
    """
    构建趋势特征

    trend_last3 = score_lag1 - score_lag3

    Returns
    -------
    df : pd.DataFrame
        添加趋势特征后的 dataframe

    feature_cols : List[str]
        新生成的趋势特征列名
    """

    new_cols: List[str] = []

    lag1 = f"{value_col}_lag1"
    lag3 = f"{value_col}_lag3"

    if lag1 in df.columns and lag3 in df.columns:
        df["trend_last3"] = df[lag1] - df[lag3]
        new_cols.append("trend_last3")

    return df, new_cols


def build_time_series_data(
    df: pd.DataFrame,
    user_col="user_id",
    time_col="week",
    value_col="score",
):

    df = df.copy()

    # 确保目标列为数值
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")

    df, lag_cols = build_lag_features(df, user_col, time_col, value_col)
    df, stat_cols = build_stat_features(df, value_col)
    df, trend_cols = build_trend_features(df, value_col)

    df = df.dropna()

    feature_cols = lag_cols + stat_cols + trend_cols

    X = df[feature_cols]
    y = df[value_col]

    return X, y, feature_cols
