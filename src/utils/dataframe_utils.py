# src/utils/dataframe_utils.py

from typing import List, Optional

import pandas as pd

from utils.logger import get_logger

logger = get_logger(__name__)


def normalize_multilabel_series(series: pd.Series) -> pd.Series:
    """
    规范化多标签字段（多标签使用 "_" 分隔）。

    该函数用于统一多标签字符串的格式，使语义相同但格式不同的标签组合
    具有一致的表示形式，便于后续特征工程、统计分析或模型训练。

    处理步骤：
    1. 使用 "_" 将字符串拆分为多个标签
    2. 去除每个标签前后的空格
    3. 过滤空标签（如 "" 或仅包含空格）
    4. 对标签进行去重
    5. 按字母顺序排序，保证标签顺序一致
    6. 使用 "_" 重新拼接为字符串

    示例
    -------
    输入：
        "B_A"
        "A_A_B"
        " A _ B _ "
        "C_B_A"

    输出：
        "A_B"
        "A_B"
        "A_B"
        "A_B_C"

    作用：
    - 保证逻辑相同的标签集合有一致的字符串表示
    - 避免 "A_B" 与 "B_A" 被当成不同类别
    - 便于 groupby、统计、特征编码等操作
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

def parse_date_fields(
    df: pd.DataFrame,
    date_fields: List[str],
    date_format: Optional[str] = None,
) -> pd.DataFrame:
    """
    将日期字段统一转为 ISO 格式字符串
    """
    df = df.copy()

    for field in date_fields:
        if field not in df.columns:
            logger.warning(f"Date field not found, skip: {field}")
            continue

        logger.info(f"Parsing date field: {field}")
        df[field] = pd.to_datetime(
            df[field],
            format=date_format,
            errors="coerce",
        ).dt.strftime("%Y-%m-%d")

    return df
