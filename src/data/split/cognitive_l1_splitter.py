# src/data/split/cognitive_l1_splitter.py

"""
cognitive_l1 数据集划分模块

该脚本默认承接 `src/data/datasets/cognitive_l1_dataset.py` 生成的
processed parquet，并复用同一套 config 和列映射配置。

划分规则：

1. 先按每个患者最近配置周数尝试划分验证集
2. 仅当该患者 max_week >= recent_valid_weeks 时，最近配置周数的数据才进入验证集
3. 其余数据划入训练集
4. 再对训练集进行过滤，仅保留训练记录数 >= valid_patient_min_weeks 的患者
5. 不再生成测试集
"""

from pathlib import Path

import pandas as pd

from configs.loader import load_config
from src.core.brain_ability_values_by_training_week_20260324.constants import ColumnName
from src.core.constants import CognitiveL1DatasetName
from src.data.dataset_meta import load_column_mapping
from src.utils.logger import get_logger, setup_logging
from src.utils.path_utils import resolve_project_path

# 初始化日志系统
setup_logging()

logger = get_logger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
DATASET_KEY = CognitiveL1DatasetName.WEEKLY_BRAIN_ABILITY.value


def load_dataset_runtime():
    config = load_config("configs/config.yaml")
    column_mapping = load_column_mapping(config, DATASET_KEY, BASE_DIR)
    split_config = config["processed_to_splitter"][DATASET_KEY]
    processed_dir = resolve_project_path(split_config["processed"], BASE_DIR)
    splitter_dir = resolve_project_path(split_config["splitter"], BASE_DIR)

    return config, column_mapping, processed_dir, splitter_dir, split_config


def split_cognitive_l1_dataset(
    df: pd.DataFrame,
    column_mapping: dict,
    valid_patient_min_weeks: int,
    recent_valid_weeks: int,
):
    """
    只生成 train / val。

    规则：
    1. 先按每个患者最近 recent_valid_weeks 周尝试划分验证集
    2. 仅当该患者 max_week >= recent_valid_weeks 时，最近 recent_valid_weeks 周的数据才进入验证集
    3. 其余数据划入训练集
    4. 再对训练集进行过滤，仅保留训练记录数 >= valid_patient_min_weeks 的患者
    """

    patient_id = column_mapping[ColumnName.PATIENT_ID.value]
    week = column_mapping[ColumnName.TRAINING_WEEK.value]

    logger.info("Starting dataset split")

    df = df.dropna(subset=[week]).copy()
    patient_max_week = df.groupby(patient_id)[week].max()

    logger.info(f"Total patients: {patient_max_week.shape[0]}")

    train_list = []
    val_list = []
    val_patient_count = 0
    train_only_patient_count = 0

    for pid, max_week in patient_max_week.items():
        patient_df = df[df[patient_id] == pid].sort_values(week).copy()
        recent_valid_start = max(int(max_week - recent_valid_weeks + 1), 1)

        if max_week >= recent_valid_weeks:
            patient_val_df = patient_df[patient_df[week] >= recent_valid_start].copy()
            patient_train_df = patient_df[patient_df[week] < recent_valid_start].copy()
            val_list.append(patient_val_df)
            train_list.append(patient_train_df)
            val_patient_count += 1
        else:
            train_list.append(patient_df)
            train_only_patient_count += 1

    train_df = (
        pd.concat(train_list, ignore_index=False)
        if train_list
        else pd.DataFrame(columns=df.columns)
    )
    val_df = (
        pd.concat(val_list, ignore_index=False)
        if val_list
        else pd.DataFrame(columns=df.columns)
    )

    train_counts = (
        train_df.groupby(patient_id)[week].count()
        if not train_df.empty
        else pd.Series(dtype="int64")
    )
    eligible_train_patients = train_counts[train_counts >= valid_patient_min_weeks].index
    train_df = train_df[train_df[patient_id].isin(eligible_train_patients)].copy()

    filtered_train_counts = (
        train_df.groupby(patient_id)[week].count()
        if not train_df.empty
        else pd.Series(dtype="int64")
    )
    if not filtered_train_counts.empty and int(filtered_train_counts.min()) < valid_patient_min_weeks:
        raise ValueError(
            "Train split contains patients with fewer records than valid_patient_min_weeks."
        )

    logger.info(f"Patients assigned to validation: {val_patient_count}")
    logger.info(f"Patients kept in train only before filtering: {train_only_patient_count}")
    logger.info(f"Patients kept in train after filtering: {len(eligible_train_patients)}")
    logger.info(f"Train samples: {len(train_df)}")
    logger.info(f"Validation samples: {len(val_df)}")
    logger.info("Dataset split finished")

    return train_df, val_df


# =========================================================
# main
# =========================================================


def main():
    _, column_mapping, processed_dir, splitter_dir, split_config = load_dataset_runtime()
    valid_patient_min_weeks = split_config.get("valid_patient_min_weeks", 5)
    recent_valid_weeks = split_config.get("recent_valid_weeks", 24)
    data_path = processed_dir / "processed.parquet"

    splitter_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading dataset: {data_path}")

    df = pd.read_parquet(data_path)

    logger.info(f"Dataset shape: {df.shape}")
    logger.info(
        "Split config: "
        f"valid_patient_min_weeks={valid_patient_min_weeks}, "
        f"recent_valid_weeks={recent_valid_weeks}"
    )

    train_df, val_df = split_cognitive_l1_dataset(
        df,
        column_mapping,
        valid_patient_min_weeks=valid_patient_min_weeks,
        recent_valid_weeks=recent_valid_weeks,
    )

    # 保存 CSV（用于 head 查看）
    train_csv = splitter_dir / "train.csv"
    val_csv = splitter_dir / "val.csv"

    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)

    # 保存 Parquet
    train_parquet = splitter_dir / "train.parquet"
    val_parquet = splitter_dir / "val.parquet"

    train_df.to_parquet(train_parquet, index=False)
    val_df.to_parquet(val_parquet, index=False)

    logger.info(f"Train CSV saved: {train_csv}")
    logger.info(f"Val CSV saved: {val_csv}")

    logger.info(f"Train Parquet saved: {train_parquet}")
    logger.info(f"Val Parquet saved: {val_parquet}")


if __name__ == "__main__":
    main()
