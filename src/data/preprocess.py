# src/data/preprocess.py

import pandas as pd

from utils.dataframe_utils import clean_dataframe
from utils.dataframe_utils import normalize_multilabel_series


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    数据清洗与基础预处理
    """

    df = clean_dataframe(df)

    # 示例：多标签字段
    multilabel_cols = ["tags", "disease"]

    for col in multilabel_cols:
        if col in df.columns:
            df[col] = normalize_multilabel_series(df[col])

    return df
