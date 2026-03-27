# src/pipelines/train_pipeline/cognitive_l1.py

import random

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from configs.loader import load_config
from src.models.model_factory import build_model
from src.training.trainer import Trainer
from src.utils.logger import get_logger

logger = get_logger(__name__)


def build_features(history: list[float]) -> dict:
    arr = np.array(history, dtype=float)
    feats = {}

    for i in range(1, min(len(arr), 12) + 1):
        feats[f'lag_{i}'] = arr[-i]

    feats['mean_4'] = arr[-4:].mean() if len(arr) >= 4 else arr.mean()
    feats['mean_12'] = arr[-12:].mean() if len(arr) >= 12 else arr.mean()
    feats['std_12'] = arr[-12:].std() if len(arr) >= 12 else arr.std()
    feats['min'] = arr.min()
    feats['max'] = arr.max()

    if len(arr) >= 2:
        x = np.arange(len(arr))
        feats['trend'] = np.polyfit(x, arr, 1)[0]
    else:
        feats['trend'] = 0.0

    feats['growth_4'] = arr[-1] - arr[-4] if len(arr) >= 4 else 0.0
    feats['growth_12'] = arr[-1] - arr[-12] if len(arr) >= 12 else 0.0
    feats['last'] = arr[-1]

    return feats


def build_training_data(
    df: pd.DataFrame,
    user_col: str,
    time_col: str,
    value_col: str,
    min_history_len: int,
    max_history_len: int,
    noise_prob: float,
    noise_std: float,
    randomize_history_len: bool,
    add_noise: bool,
):
    df = df.sort_values([user_col, time_col])
    rows = []

    for _, user_df in df.groupby(user_col):
        values = user_df[value_col].values.astype(float)

        if len(values) < min_history_len + 1:
            continue

        for i in range(min_history_len, len(values) - 1):
            max_available_history = min(max_history_len, i)
            if randomize_history_len:
                hist_len = random.randint(min_history_len, max_available_history)
            else:
                hist_len = max_available_history

            history = values[i - hist_len:i].copy()

            if add_noise:
                for j in range(len(history)):
                    if random.random() < noise_prob:
                        history[j] += np.random.normal(0, noise_std)

            current = values[i]
            target = values[i + 1] - current

            feats = build_features(history)
            feats['hist_len'] = hist_len
            feats['current'] = current
            feats['_target'] = target
            rows.append(feats)

    if not rows:
        return pd.DataFrame(), pd.Series(dtype='float64'), []

    feature_df = pd.DataFrame(rows)
    y = feature_df.pop('_target')

    return feature_df, y, feature_df.columns.tolist()


def build_validation_data(
    df: pd.DataFrame,
    user_col: str,
    time_col: str,
    value_col: str,
    min_history_len: int,
    max_history_len: int,
    validation_horizon_weeks: int,
):
    """
    为验证集构建样本。

    每个样本包含三段：
    1. history
    2. current
    3. 后续 validation_horizon_weeks 周数据

    y_val 取这段 validation_horizon_weeks 之后的真实数值。
    """

    df = df.sort_values([user_col, time_col])
    rows = []

    for _, user_df in df.groupby(user_col):
        values = user_df[value_col].values.astype(float)

        min_required_len = min_history_len + validation_horizon_weeks + 1
        if len(values) < min_required_len:
            continue

        for i in range(min_history_len, len(values) - validation_horizon_weeks):
            hist_len = min(max_history_len, i)
            history = values[i - hist_len:i].copy()
            current = values[i]
            target_value = values[i + validation_horizon_weeks]

            feats = build_features(history)
            feats['hist_len'] = hist_len
            feats['current'] = current
            feats['_target'] = target_value
            rows.append(feats)

    if not rows:
        return pd.DataFrame(), pd.Series(dtype='float64'), []

    feature_df = pd.DataFrame(rows)
    y = feature_df.pop('_target')

    return feature_df, y, feature_df.columns.tolist()


def train_pipeline(
    train_df,
    val_df,
    user_col,
    time_col,
    target,
):
    logger.info(f'Start training pipeline for target: {target}')

    config = load_config('configs/train.yaml')

    model_name = config['model_name']
    model_params = config.get('model_params', {})
    training_data_params = config.get('training_data_params', {})

    min_history_len = training_data_params.get('min_history_len', 3)
    max_history_len = training_data_params.get('max_history_len', 12)
    noise_prob = training_data_params.get('noise_prob', 0.3)
    noise_std = training_data_params.get('noise_std', 0.5)
    validation_horizon_weeks = training_data_params.get('validation_horizon_weeks', 12)

    logger.info(f'Model name: {model_name}')
    logger.info(f'Model params: {model_params}')
    logger.info(f'Training data params: {training_data_params}')

    logger.info('Building training features...')

    X_train, y_train, feature_cols = build_training_data(
        train_df,
        user_col=user_col,
        time_col=time_col,
        value_col=target,
        min_history_len=min_history_len,
        max_history_len=max_history_len,
        noise_prob=noise_prob,
        noise_std=noise_std,
        randomize_history_len=True,
        add_noise=True,
    )

    X_val, y_val, _ = build_validation_data(
        val_df,
        user_col=user_col,
        time_col=time_col,
        value_col=target,
        min_history_len=min_history_len,
        max_history_len=max_history_len,
        validation_horizon_weeks=validation_horizon_weeks,
    )

    logger.info(f'Training samples: {len(X_train)}')
    logger.info(f'Validation samples: {len(X_val)}')
    logger.info(f'Feature count: {len(feature_cols)}')

    if X_train.empty or y_train.empty:
        raise ValueError(f'No training samples generated for target: {target}')

    if X_val.empty or y_val.empty:
        logger.warning(f'No validation samples generated for target: {target}')
        X_val = None
        y_val = None
    elif feature_cols:
        X_val = X_val.reindex(columns=feature_cols, fill_value=0.0)

    logger.info('Building model...')

    model = build_model(model_name=model_name, params=model_params)
    trainer = Trainer(model)

    val_pred = trainer.fit(
        X_train,
        y_train,
        X_val,
        y_val,
    )

    if val_pred is not None and y_val is not None:
        rmse = np.sqrt(mean_squared_error(y_val, val_pred))
        mae = mean_absolute_error(y_val, val_pred)

        logger.info('Validation Result')
        logger.info(f'RMSE: {rmse:.4f}')
        logger.info(f'MAE : {mae:.4f}')

    logger.info(f'{target} model training finished')

    return model, feature_cols
