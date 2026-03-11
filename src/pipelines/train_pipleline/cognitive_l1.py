# src/pipelines/train_pipeline/cognitive_l1.py

import pandas as pd

from configs.loader import load_config
from src.features.time_series_features import build_time_series_data
from src.models.lightgbm_model import LightGBMModel
from src.training.trainer import Trainer
from src.utils.logger import get_logger

logger = get_logger(__name__)


from src.utils.logger import get_logger
from src.models.model_factory import build_model
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

logger = get_logger(__name__)


def train_pipeline(
    train_df,
    val_df,
    user_col,
    time_col,
    target,
):

    logger.info(f"Start training pipeline for target: {target}")

    # ------------------------------------------------
    # 1 读取模型配置
    # ------------------------------------------------

    config = load_config("configs/train.yaml")

    model_name = config["model_name"]
    model_params = config.get("model_params", {})

    logger.info(f"Model name: {model_name}")
    logger.info(f"Model params: {model_params}")

    # ------------------------------------------------
    # 2 构建训练特征
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
    # 3 构建模型
    # ------------------------------------------------

    logger.info("Building model...")

    model = build_model(model_name=model_name, params=model_params)

    trainer = Trainer(model)

    # ------------------------------------------------
    # 4 训练模型
    # ------------------------------------------------

    val_pred = trainer.fit(
        X_train,
        y_train,
        X_val,
        y_val,
    )

    # ------------------------------------------------
    # 5 验证集评估
    # ------------------------------------------------

    if val_pred is not None:

        rmse = np.sqrt(mean_squared_error(y_val, val_pred))
        mae = mean_absolute_error(y_val, val_pred)

        logger.info("Validation Result")
        logger.info(f"RMSE: {rmse:.4f}")
        logger.info(f"MAE : {mae:.4f}")

    logger.info(f"{target} model training finished")

    return model, feature_cols
