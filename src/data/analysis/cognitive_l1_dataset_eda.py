"""
EDA 脚本：cognitive_l1_dataset 数据探索分析

功能：
1. 读取 processed.parquet 数据
2. 统计数据集基本信息
3. 统计患者信息
4. 统计训练周数分布
5. 统计认知能力分数分布
6. 统计缺失值情况
7. 将分析结果保存为 JSON 和 CSV 文件

运行方式：
python -m src.data.analysis.cognitive_l1_dataset_eda
"""

import json
from pathlib import Path

import pandas as pd

from configs.loader import load_config
from src.core.brain_ability_values_by_training_week_20260324.constants import ColumnName
from src.data.dataset_meta import build_column_accessor, load_column_mapping
from src.utils.logger import get_logger, setup_logging
from src.utils.path_utils import resolve_project_path


# 初始化日志系统
setup_logging()
logger = get_logger(__name__)


# 项目根目录
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
DEFAULT_ENABLED_REPORTS = {
    "dataset_overview",
    "patient_statistics",
    "training_week_statistics",
    "cognitive_score_statistics",
    "missing_values",
}


def load_dataset_runtime():
    config = load_config("configs/config.yaml")
    eda_config = config["eda"]
    dataset_key = eda_config["dataset_key"]
    column_mapping = load_column_mapping(config, dataset_key, BASE_DIR)
    processed_dir = resolve_project_path(
        config["raw_to_processed"][dataset_key]["processed"], BASE_DIR
    )
    output_dir = resolve_project_path(eda_config["output_dir"], BASE_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    return config, eda_config, column_mapping, processed_dir, output_dir


def resolve_cognitive_score_columns(cols, eda_config: dict) -> list[str]:
    columns = []

    for column_name in eda_config["cognitive_score_columns"]:
        columns.append(getattr(cols, column_name.lower()))

    return columns


def dataset_overview(df: pd.DataFrame):
    """
    数据集整体概览

    包括：
    - 数据行数
    - 数据列数
    - 列名列表
    """

    info = {
        "num_rows": int(df.shape[0]),
        "num_columns": int(df.shape[1]),
        "columns": df.columns.tolist(),
    }

    return info


def patient_statistics(df: pd.DataFrame, cols):
    """
    患者统计信息

    包括：
    - 患者总数量
    - 性别分布
    """

    patient_id = cols.patient_id
    gender = cols.gender

    patient_df = df[[patient_id, gender]].drop_duplicates(subset=[patient_id])

    stats = {
        "num_patients": int(patient_df[patient_id].nunique()),
        "gender_distribution": patient_df[gender].value_counts().to_dict(),
    }

    return stats


def training_week_statistics(df: pd.DataFrame, cols, thresholds: list[int]):
    """
    训练周数统计

    统计每个患者训练的最大周数，并计算整体统计指标
    同时统计不同训练时长的患者占比
    """

    patient_id = cols.patient_id
    week = cols.training_week

    df = df.dropna(subset=[week])
    patient_weeks = df.groupby(patient_id)[week].max()
    desc = patient_weeks.describe()

    total_patients = int(desc["count"])
    max_week_distribution = patient_weeks.value_counts().sort_values(ascending=False)

    stats = {
        "num_patients": total_patients,
        "mean_training_weeks": round(float(desc["mean"]), 2),
        "std_training_weeks": round(float(desc["std"]), 2),
        "min_training_weeks": int(desc["min"]),
        "max_training_weeks": int(desc["max"]),
        "max_training_week_distribution": {
            int(week): {
                "ratio": round(float(count / total_patients), 4),
                "count": int(count),
            }
            for week, count in max_week_distribution.items()
        },
    }

    for threshold in thresholds:
        matched_count = int((patient_weeks >= threshold).sum())
        stats[f"week_ge_{threshold}_ratio"] = round(matched_count / total_patients, 2)
        stats[f"week_ge_{threshold}_count"] = matched_count

    return stats, patient_weeks


def cognitive_score_statistics(df: pd.DataFrame, columns: list[str]):
    """
    认知能力得分统计
    """

    stats = df[columns].describe().round(2).to_dict()

    return stats


def missing_value_statistics(df: pd.DataFrame):
    """
    缺失值统计

    返回：
    每一列缺失值数量
    """

    missing = df.isnull().sum()
    missing = missing[missing > 0]

    return missing.to_dict()


def run_eda():
    """
    执行完整 EDA 分析流程
    """

    logger.info("========== EDA START ==========")

    _, eda_config, column_mapping, processed_dir, output_dir = load_dataset_runtime()
    cols = build_column_accessor(column_mapping, ColumnName)
    data_path = processed_dir / "processed.parquet"
    enabled_reports = set(eda_config.get("enabled_reports", DEFAULT_ENABLED_REPORTS))
    thresholds = eda_config.get("training_week_thresholds", [3, 5, 8, 10])
    score_columns = resolve_cognitive_score_columns(cols, eda_config)

    logger.info(f"Loading dataset: {data_path}")

    df = pd.read_parquet(data_path)

    logger.info(f"Dataset shape: {df.shape}")

    report = {}
    patient_weeks = None

    if "dataset_overview" in enabled_reports:
        logger.info("Running dataset overview analysis")
        report["dataset_overview"] = dataset_overview(df)

    if "patient_statistics" in enabled_reports:
        logger.info("Running patient statistics analysis")
        report["patient_statistics"] = patient_statistics(df, cols)

    if "training_week_statistics" in enabled_reports:
        logger.info("Running training week statistics analysis")
        week_stats, patient_weeks = training_week_statistics(df, cols, thresholds)
        report["training_week_statistics"] = week_stats

    if "cognitive_score_statistics" in enabled_reports:
        logger.info("Running cognitive ability statistics analysis")
        report["cognitive_score_statistics"] = cognitive_score_statistics(df, score_columns)

    if "missing_values" in enabled_reports:
        logger.info("Running missing value analysis")
        report["missing_values"] = missing_value_statistics(df)

    report_path = output_dir / "eda_report.json"

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4, ensure_ascii=False)

    logger.info(f"EDA report saved to: {report_path}")

    if patient_weeks is not None:
        weeks_path = output_dir / "patient_training_weeks.csv"
        patient_weeks_df = patient_weeks.reset_index()
        patient_weeks_df.columns = ["patient_id", "max_training_week"]
        patient_weeks_df.to_csv(weeks_path, index=False)
        logger.info(f"Patient training weeks saved to: {weeks_path}")


if __name__ == "__main__":
    run_eda()
