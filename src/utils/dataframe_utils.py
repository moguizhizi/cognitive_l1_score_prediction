# src/utils/dataframe_utils.py

import pandas as pd


def normalize_multilabel_series(series: pd.Series) -> pd.Series:
    """
    规范多标签字段：
    """

    return series.str.split("_").apply(
        lambda parts: "_".join(sorted({p.strip() for p in parts if p and p.strip()}))
    )


def clean_dataframe(
    df: pd.DataFrame, multi_label_keywords: list | None = None
) -> pd.DataFrame:
    """
    高性能清洗函数

    当 multi_label_keywords=None 时：
        -> 不执行多标签规范化
    """

    df.columns = (
        df.columns.str.replace(
            r"[\u200b\u200c\u200d\ufeff]", "", regex=True
        )  # 去隐形字符
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )

    df = (
        df.fillna("")
        .astype(str)
        .apply(lambda col: col.str.strip())
        .replace(r"\s*_\s*", "_", regex=True)
        .replace(r"\s+", " ", regex=True)
        .replace(r"[\u200b\u200c\u200d\ufeff]", "", regex=True)
    )

    # 关键改动
    if not multi_label_keywords:
        return df

    # 防止有人传字符串，例如 "颜色"
    if isinstance(multi_label_keywords, str):
        multi_label_keywords = [multi_label_keywords]

    target_cols = [
        col for col in df.columns if any(k == col for k in multi_label_keywords)
    ]

    for col in target_cols:
        try:
            df[col] = normalize_multilabel_series(df[col])
        except Exception:
            pass

    return df
