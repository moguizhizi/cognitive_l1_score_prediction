# src/pipelines/train_pipeline.py

import pandas as pd

from src.features.time_series_features import build_training_dataset
from src.models.lightgbm_model import LightGBMModel


def train_pipeline(df: pd.DataFrame):

    print("Building features...")

    X, y, feature_cols = build_training_dataset(df)

    print("Training LightGBM...")

    model = LightGBMModel()

    model.fit(X, y)

    print("Training finished")

    return model, feature_cols
