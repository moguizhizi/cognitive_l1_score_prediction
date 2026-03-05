# src/data/loader.py

import pandas as pd
from pathlib import Path


def load_xlsx(path: str) -> pd.DataFrame:
    """读取 xlsx 原始数据"""
    return pd.read_excel(path)


def save_parquet(df: pd.DataFrame, path: str):
    """保存 parquet 文件"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def load_parquet(path: str) -> pd.DataFrame:
    """读取 parquet 文件"""
    return pd.read_parquet(path)


def convert_xlsx_to_parquet(xlsx_path: str, parquet_path: str) -> pd.DataFrame:
    """
    将 raw xlsx 转换为 parquet
    """

    df = load_xlsx(xlsx_path)

    save_parquet(df, parquet_path)

    return df
