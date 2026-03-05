# src/data/loader.py

from utils.xlsx_utils import xlsx_to_parquet_dataset


def convert_xlsx_to_parquet(
    xlsx_path: str,
    parquet_path: str,
    compression="zstd",
    overwrite=False,
    multi_label_keywords: list = None,
):
    """
    将 raw xlsx 转换为 parquet
    """

    xlsx_to_parquet_dataset(
        input_path=xlsx_path,
        output_dir=parquet_path,
        compression=compression,
        overwrite=overwrite,
        multi_label_keywords=multi_label_keywords,
    )
