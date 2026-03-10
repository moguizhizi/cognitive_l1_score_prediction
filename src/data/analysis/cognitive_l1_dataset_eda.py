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

from src.utils.logger import get_logger, setup_logging
from src.core.raw_training_weekly_cognitive_ability_scores.constants import ColumnName


# 初始化日志系统
setup_logging()
logger = get_logger(__name__)


# 项目根目录
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent

# 处理后的数据路径
DATA_PATH = (
    BASE_DIR
    / "data/processed/raw_training_weekly_cognitive_ability_scores/processed.parquet"
)

# EDA结果输出目录
OUTPUT_DIR = BASE_DIR / "experiments/eda/raw_training_weekly_cognitive_ability_scores"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

COGNITIVE_ABILITY_TRAINING = (
    BASE_DIR / "src" / "core" / "raw_training_weekly_cognitive_ability_scores"
)

with open(Path(COGNITIVE_ABILITY_TRAINING) / "column_mapping.json") as f:
    COLUMN_MAPPING = json.load(f)


def dataset_overview(df: pd.DataFrame):
    """
    数据集整体概览

    包括：
    - 数据行数
    - 数据列数
    - 列名列表
    """

    info = {
        "num_rows": int(df.shape[0]),  # 数据行数
        "num_columns": int(df.shape[1]),  # 数据列数
        "columns": df.columns.tolist(),  # 所有列名
    }

    return info


def patient_statistics(df: pd.DataFrame):
    """
    患者统计信息

    包括：
    - 患者总数量
    - 性别分布
    """

    patient_id = COLUMN_MAPPING[ColumnName.PATIENT_ID.value]
    gender = COLUMN_MAPPING[ColumnName.GENDER.value]

    # 每个患者只保留一条记录
    patient_df = df[[patient_id, gender]].drop_duplicates(subset=[patient_id])

    stats = {
        "num_patients": int(patient_df[patient_id].nunique()),
        "gender_distribution": patient_df[gender].value_counts().to_dict(),
    }

    return stats


def training_week_statistics(df: pd.DataFrame):
    """
    训练周数统计

    统计每个患者训练的最大周数，并计算整体统计指标
    同时统计不同训练时长的患者占比
    """

    patient_id = COLUMN_MAPPING[ColumnName.PATIENT_ID.value]
    week = COLUMN_MAPPING[ColumnName.TRAINING_WEEK.value]

    # 删除非法week
    df = df.dropna(subset=[week])

    # 每个患者的训练最大周数
    patient_weeks = df.groupby(patient_id)[week].max()

    # 统计信息
    desc = patient_weeks.describe()

    total_patients = int(desc["count"])

    stats = {
        "num_patients": total_patients,
        "mean_training_weeks": round(float(desc["mean"]), 2),
        "std_training_weeks": round(float(desc["std"]), 2),
        "min_training_weeks": int(desc["min"]),
        "max_training_weeks": int(desc["max"]),
        # 训练周数达标占比
        "week_ge_3_ratio": round((patient_weeks >= 3).sum() / total_patients, 2),
        "week_ge_5_ratio": round((patient_weeks >= 5).sum() / total_patients, 2),
        "week_ge_8_ratio": round((patient_weeks >= 8).sum() / total_patients, 2),
        "week_ge_10_ratio": round((patient_weeks >= 10).sum() / total_patients, 2),
    }

    return stats, patient_weeks


def cognitive_score_statistics(df: pd.DataFrame):
    """
    认知能力得分统计

    包括：
    - 感知觉
    - 注意力
    - 记忆力
    - 执行功能
    """

    columns = [
        COLUMN_MAPPING[ColumnName.PERCEPTION.value],
        COLUMN_MAPPING[ColumnName.ATTENTION.value],
        COLUMN_MAPPING[ColumnName.MEMORY.value],
        COLUMN_MAPPING[ColumnName.EXECUTIVE_FUNCTION.value],
    ]

    # describe统计
    stats = df[columns].describe().round(2).to_dict()

    return stats


def missing_value_statistics(df: pd.DataFrame):
    """
    缺失值统计

    返回：
    每一列缺失值数量
    """

    missing = df.isnull().sum()

    # 只保留存在缺失值的列
    missing = missing[missing > 0]

    return missing.to_dict()


def run_eda():
    """
    执行完整 EDA 分析流程
    """

    logger.info("========== EDA START ==========")

    # 加载数据
    logger.info(f"Loading dataset: {DATA_PATH}")

    df = pd.read_parquet(DATA_PATH)

    logger.info(f"Dataset shape: {df.shape}")

    # 保存分析结果
    report = {}

    # 数据集概览
    logger.info("Running dataset overview analysis")
    report["dataset_overview"] = dataset_overview(df)

    # 患者统计
    logger.info("Running patient statistics analysis")
    report["patient_statistics"] = patient_statistics(df)

    # 训练周数统计
    logger.info("Running training week statistics analysis")
    week_stats, patient_weeks = training_week_statistics(df)

    report["training_week_statistics"] = week_stats

    # 认知能力统计
    logger.info("Running cognitive ability statistics analysis")
    report["cognitive_score_statistics"] = cognitive_score_statistics(df)

    # 缺失值统计
    logger.info("Running missing value analysis")
    report["missing_values"] = missing_value_statistics(df)

    # 保存 EDA JSON 报告
    report_path = OUTPUT_DIR / "eda_report.json"

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4, ensure_ascii=False)

    logger.info(f"EDA report saved to: {report_path}")

    # 保存每个患者训练周数（CSV格式）
    weeks_path = OUTPUT_DIR / "patient_training_weeks.csv"

    patient_weeks_df = patient_weeks.reset_index()
    patient_weeks_df.columns = ["patient_id", "max_training_week"]

    patient_weeks_df.to_csv(weeks_path, index=False)

    logger.info(f"Patient training weeks saved to: {weeks_path}")


if __name__ == "__main__":
    run_eda()
