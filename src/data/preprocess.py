# src/data/preprocess.py

from typing import Dict, List, Optional

import pandas as pd

from utils.dataframe_utils import clean_dataframe, drop_empty_rows, normalize_columns, parse_date_fields, parse_multivalue_columns, validate_schema
from utils.dataframe_utils import normalize_multilabel_series


def preprocess_dataframe(
    df: pd.DataFrame,
    column_mapping: Optional[Dict[str, str]] = None,
    date_fields: Optional[List[str]] = None,
    multi_value_fields: Optional[List[str]] = None,
    required_fields: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    DataFrame 数据预处理主流程

    包含：
    - 列名规范化
    - 删除空行
    - 缺失值填充
    - schema校验（可选）
    - 日期字段解析（可选）
    - 多值字段拆分（可选）
    """

    df = normalize_columns(df, column_mapping=column_mapping)

    df = drop_empty_rows(df)

    df = fill_na_values(df)

    if required_fields:
        validate_schema(df, required_fields)

    if date_fields:
        df = parse_date_fields(df, date_fields)

    if multi_value_fields:
        df = parse_multivalue_columns(df, multi_value_fields)

    return df
