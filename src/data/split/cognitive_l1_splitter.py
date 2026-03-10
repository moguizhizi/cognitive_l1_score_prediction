# src/data/split/cognitive_l1_splitter.py

"""
cognitive_l1 数据集划分模块

划分规则：

第一步：
筛选出训练最大周数 >= 3 的患者数据

第二步：
在第一步基础上构建验证集和测试集：

    1）筛选出训练最大周数 >= 10 的患者
    2）最近四周的数据作为测试集
    3）倒数第二周 ~ 倒数第五周的数据作为验证集（允许与测试集重叠）

第三步：
剩余数据作为训练集
"""

import pandas as pd
import json
from pathlib import Path

from src.core.raw_training_weekly_cognitive_ability_scores.constants import ColumnName
from src.utils.logger import get_logger, setup_logging

# 初始化日志系统
setup_logging()

logger = get_logger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent.parent
COGNITIVE_ABILITY_TRAINING = (
    BASE_DIR / "core" / "raw_training_weekly_cognitive_ability_scores"
)

with open(Path(COGNITIVE_ABILITY_TRAINING) / "column_mapping.json") as f:
    COLUMN_MAPPING = json.load(f)


def split_cognitive_l1_dataset(df: pd.DataFrame, column_mapping: dict):
    """
    按训练周数规则划分 train / val / test
    """

    patient_id = column_mapping[ColumnName.PATIENT_ID.value]
    week = column_mapping[ColumnName.TRAINING_WEEK.value]

    logger.info("Starting dataset split")

    df = df.dropna(subset=[week])

    # 计算每个患者最大训练周数
    patient_max_week = df.groupby(patient_id)[week].max()

    logger.info(f"Total patients: {patient_max_week.shape[0]}")

    # --------------------------------------------------
    # Step 1: 只保留训练周数 >= 3 的患者
    # --------------------------------------------------

    valid_patients = patient_max_week[patient_max_week >= 3].index
    df = df[df[patient_id].isin(valid_patients)]

    logger.info(f"Patients with >=3 weeks training: {len(valid_patients)}")
    logger.info(f"Remaining samples: {len(df)}")

    # --------------------------------------------------
    # Step 2: 构建 test / val
    # --------------------------------------------------

    test_list = []
    val_list = []

    long_training_patients = patient_max_week[patient_max_week >= 10]

    logger.info(f"Patients with >=10 weeks training: {len(long_training_patients)}")

    for pid, max_week in long_training_patients.items():

        patient_df = df[df[patient_id] == pid]

        # test: 最近四周
        test_start = max_week - 3
        test_df = patient_df[patient_df[week] >= test_start]

        # val: 倒数第二周 ~ 倒数第五周
        val_start = max_week - 4
        val_end = max_week - 1

        val_df = patient_df[
            (patient_df[week] >= val_start) & (patient_df[week] <= val_end)
        ]

        test_list.append(test_df)
        val_list.append(val_df)

    test_df = pd.concat(test_list) if test_list else pd.DataFrame()
    val_df = pd.concat(val_list) if val_list else pd.DataFrame()

    logger.info(f"Test samples: {len(test_df)}")
    logger.info(f"Validation samples: {len(val_df)}")

    # --------------------------------------------------
    # Step 3: 剩余作为 train
    # --------------------------------------------------

    used_index = set(test_df.index).union(set(val_df.index))
    train_df = df.loc[~df.index.isin(used_index)]

    logger.info(f"Train samples: {len(train_df)}")

    logger.info("Dataset split finished")

    return train_df, val_df, test_df


# =========================================================
# main
# =========================================================


def main():

    DATA_PATH = Path(
        "data/processed/raw_training_weekly_cognitive_ability_scores/processed.parquet"
    )

    OUTPUT_DIR = Path("data/splitter/raw_training_weekly_cognitive_ability_scores")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading dataset: {DATA_PATH}")

    df = pd.read_parquet(DATA_PATH)

    logger.info(f"Dataset shape: {df.shape}")

    train_df, val_df, test_df = split_cognitive_l1_dataset(df, COLUMN_MAPPING)

    # 保存 CSV（用于 head 查看）
    train_csv = OUTPUT_DIR / "train.csv"
    val_csv = OUTPUT_DIR / "val.csv"
    test_csv = OUTPUT_DIR / "test.csv"

    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)
    test_df.to_csv(test_csv, index=False)

    # 保存 Parquet
    train_parquet = OUTPUT_DIR / "train.parquet"
    val_parquet = OUTPUT_DIR / "val.parquet"
    test_parquet = OUTPUT_DIR / "test.parquet"

    train_df.to_parquet(train_parquet, index=False)
    val_df.to_parquet(val_parquet, index=False)
    test_df.to_parquet(test_parquet, index=False)

    logger.info(f"Train CSV saved: {train_csv}")
    logger.info(f"Val CSV saved: {val_csv}")
    logger.info(f"Test CSV saved: {test_csv}")

    logger.info(f"Train Parquet saved: {train_parquet}")
    logger.info(f"Val Parquet saved: {val_parquet}")
    logger.info(f"Test Parquet saved: {test_parquet}")


if __name__ == "__main__":
    main()
