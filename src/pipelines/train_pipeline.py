# src/pipelines/train_pipeline.py

# import pandas as pd

# from src.features.time_series_features import build_training_dataset
# from src.models.lightgbm_model import LightGBMModel

# from utils.logger import get_logger

# logger = get_logger(__name__)


# def train_pipeline(df: pd.DataFrame):

#     logger.info("Start training pipeline")

#     logger.info(f"Input dataframe shape: {df.shape}")

#     logger.info("Building training features...")
#     X, y, feature_cols = build_training_dataset(df)

#     logger.info(f"Feature building completed")
#     logger.info(f"Training samples: {len(X)}")
#     logger.info(f"Number of features: {len(feature_cols)}")
#     logger.debug(f"Feature columns: {feature_cols}")

#     logger.info("Training LightGBM model...")

#     model = LightGBMModel()
#     model.fit(X, y)

#     logger.info("Model training finished")

#     return model, feature_cols


import pandas as pd

from src.models.model_factory import build_model
from src.training.trainer import Trainer


def train_pipeline():

    train_df = pd.read_parquet("data/splitter/train.parquet")
    val_df = pd.read_parquet("data/splitter/val.parquet")

    target = "score"

    X_train = train_df.drop(columns=[target])
    y_train = train_df[target]

    X_val = val_df.drop(columns=[target])
    y_val = val_df[target]

    # 构建模型
    model = build_model("lightgbm")

    # Trainer
    trainer = Trainer(model)

    preds = trainer.fit(X_train, y_train, X_val, y_val)

    print(preds[:10])
