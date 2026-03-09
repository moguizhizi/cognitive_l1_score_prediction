# src/pipelines/train_pipeline.py

import pandas as pd

from src.features.time_series_features import build_training_dataset
from src.models.lightgbm_model import LightGBMModel

from utils.logger import get_logger

logger = get_logger(__name__)


def train_pipeline(df: pd.DataFrame):

    logger.info("Start training pipeline")

    logger.info(f"Input dataframe shape: {df.shape}")

    logger.info("Building training features...")
    X, y, feature_cols = build_training_dataset(df)

    logger.info(f"Feature building completed")
    logger.info(f"Training samples: {len(X)}")
    logger.info(f"Number of features: {len(feature_cols)}")
    logger.debug(f"Feature columns: {feature_cols}")

    logger.info("Training LightGBM model...")

    model = LightGBMModel()
    model.fit(X, y)

    logger.info("Model training finished")

    return model, feature_cols
