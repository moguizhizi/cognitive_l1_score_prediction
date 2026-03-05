# src/data/dataset.py

import pandas as pd


def build_dataset(
    df: pd.DataFrame,
    feature_cols: list,
    target_col: str,
):
    """
    构建模型训练数据
    """

    X = df[feature_cols]
    y = df[target_col]

    return X, y