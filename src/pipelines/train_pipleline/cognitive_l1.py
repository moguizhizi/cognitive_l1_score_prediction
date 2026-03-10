# src/pipelines/train_pipeline/cognitive_l1.py

import pandas as pd

from src.features.time_series_features import build_time_series_data
from src.models.lightgbm_model import LightGBMModel
from src.training.trainer import Trainer
from src.utils.logger import get_logger

logger = get_logger(__name__)


def train_pipeline(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    user_col: str,
    time_col: str,
    target: str,
):

    logger.info(f"Start training pipeline for target: {target}")

    # ------------------------------------------------
    # 1 构建训练特征
    # ------------------------------------------------

    logger.info("Building training features...")

    X_train, y_train, feature_cols = build_time_series_data(
        train_df,
        user_col=user_col,
        time_col=time_col,
        value_col=target,
    )

    X_val, y_val, _ = build_time_series_data(
        val_df,
        user_col=user_col,
        time_col=time_col,
        value_col=target,
    )

    logger.info(f"Training samples: {len(X_train)}")
    logger.info(f"Validation samples: {len(X_val)}")
    logger.info(f"Feature count: {len(feature_cols)}")

    # ------------------------------------------------
    # 2 训练模型
    # ------------------------------------------------

    logger.info(f"Training LightGBM model for {target}")

    model = LightGBMModel()

    trainer = Trainer(model)

    trainer.fit(
        X_train,
        y_train,
        X_val,
        y_val,
    )

    logger.info(f"{target} model training finished")

    return model, feature_cols
