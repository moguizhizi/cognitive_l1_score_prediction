import pandas as pd
import json
from pathlib import Path
import pickle

from src.pipelines.train_pipleline.cognitive_l1 import train_pipeline
from src.utils.logger import get_logger, setup_logging
from src.core.raw_training_weekly_cognitive_ability_scores.constants import ColumnName

# 初始化日志系统
setup_logging()
logger = get_logger(__name__)

# 项目根目录
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent

COGNITIVE_ABILITY_TRAINING = (
    BASE_DIR / "src" / "core" / "raw_training_weekly_cognitive_ability_scores"
)

with open(Path(COGNITIVE_ABILITY_TRAINING) / "column_mapping.json") as f:
    COLUMN_MAPPING = json.load(f)


TARGET_COLS = [
    COLUMN_MAPPING[ColumnName.PERCEPTION.value],
    COLUMN_MAPPING[ColumnName.ATTENTION.value],
    COLUMN_MAPPING[ColumnName.MEMORY.value],
    COLUMN_MAPPING[ColumnName.EXECUTIVE_FUNCTION.value],
]


def train_all_models(train_df: pd.DataFrame, val_df: pd.DataFrame):

    logger.info("Training all cognitive models")

    models = {}
    feature_dict = {}

    for target in TARGET_COLS:

        logger.info(f"Training target: {target}")

        model, feature_cols = train_pipeline(
            train_df,
            val_df,
            user_col=COLUMN_MAPPING[ColumnName.PATIENT_ID.value],
            time_col=COLUMN_MAPPING[ColumnName.TRAINING_WEEK.value],
            target=target,
        )

        models[target] = model
        feature_dict[target] = feature_cols

    logger.info("All models trained successfully")

    return models, feature_dict


def main():

    logger.info("Start cognitive ability model training")

    # ------------------------------------------------
    # 1 读取数据
    # ------------------------------------------------

    data_dir = (
        BASE_DIR / "data" / "splitter" / "raw_training_weekly_cognitive_ability_scores"
    )

    train_path = data_dir / "train.parquet"
    val_path = data_dir / "val.parquet"

    logger.info(f"Loading train data: {train_path}")
    train_df = pd.read_parquet(train_path)

    logger.info(f"Loading val data: {val_path}")
    val_df = pd.read_parquet(val_path)

    logger.info(f"Train shape: {train_df.shape}")
    logger.info(f"Val shape: {val_df.shape}")

    # ------------------------------------------------
    # 2 训练模型
    # ------------------------------------------------

    models, feature_dict = train_all_models(train_df, val_df)

    # ------------------------------------------------
    # 3 保存模型
    # ------------------------------------------------

    model_dir = BASE_DIR / "checkpoints" / "cognitive_l1"
    model_dir.mkdir(parents=True, exist_ok=True)

    for target, model in models.items():

        model_path = model_dir / f"{target}_lightgbm.txt"

        model.save(model_path)

        logger.info(f"Model saved: {model_path}")

    # ------------------------------------------------
    # 4 保存特征列表
    # ------------------------------------------------

    feature_path = model_dir / "feature_columns.json"

    with open(feature_path, "w") as f:
        json.dump(feature_dict, f, indent=4)

    logger.info(f"Feature columns saved: {feature_path}")

    logger.info("Training finished successfully")


if __name__ == "__main__":
    main()
