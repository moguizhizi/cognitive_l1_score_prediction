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
            hist_len = (
                random.randint(min_history_len, max_available_history)
                if randomize_history_len
                else max_available_history
            )

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


def recursive_forecast(
    model,
    history: list[float],
    current: float,
    steps: int,
    clip_min: float,
    clip_max: float,
    max_history_len: int,
) -> list[float]:
    history = history.copy()
    preds = []
    current_value = float(current)

    for _ in range(steps):
        effective_history = history[-max_history_len:]

        feats = build_features(effective_history)
        feats['hist_len'] = len(effective_history)
        feats['current'] = current_value

        X = pd.DataFrame([feats])
        delta = model.predict(X)[0]
        delta = np.clip(delta, clip_min, clip_max)

        next_value = current_value + delta
        preds.append(float(next_value))

        history.append(float(current_value))
        current_value = float(next_value)

    return preds


def evaluate_recursive_validation(
    model,
    df: pd.DataFrame,
    user_col: str,
    time_col: str,
    value_col: str,
    min_history_len: int,
    max_history_len: int,
    validation_horizon_weeks: int,
    clip_min: float,
    clip_max: float,
):
    df = df.sort_values([user_col, time_col])
    predictions = []
    targets = []
    horizon_predictions = []
    horizon_targets = []
    sequence_count = 0

    for _, user_df in df.groupby(user_col):
        values = user_df[value_col].values.astype(float)
        if len(values) < min_history_len + validation_horizon_weeks + 1:
            continue

        for i in range(min_history_len, len(values) - validation_horizon_weeks):
            hist_len = min(max_history_len, i)
            history = values[i - hist_len:i].tolist()
            current = float(values[i])
            actual_future = values[i + 1 : i + 1 + validation_horizon_weeks].tolist()

            pred_future = recursive_forecast(
                model=model,
                history=history,
                current=current,
                steps=validation_horizon_weeks,
                clip_min=clip_min,
                clip_max=clip_max,
                max_history_len=max_history_len,
            )

            predictions.extend(pred_future)
            targets.extend(actual_future)
            horizon_predictions.append(pred_future[-1])
            horizon_targets.append(actual_future[-1])
            sequence_count += 1

    if not predictions:
        return None

    rmse = np.sqrt(mean_squared_error(targets, predictions))
    mae = mean_absolute_error(targets, predictions)
    horizon_rmse = np.sqrt(mean_squared_error(horizon_targets, horizon_predictions))
    horizon_mae = mean_absolute_error(horizon_targets, horizon_predictions)
    horizon_relative_error = np.mean(
        np.abs(np.array(horizon_predictions) - np.array(horizon_targets))
        / np.maximum(np.abs(np.array(horizon_targets)), 1e-8)
    )
    horizon_accuracy = max(0.0, 1.0 - float(horizon_relative_error))

    return {
        'sequence_count': sequence_count,
        'point_count': len(predictions),
        'RMSE': rmse,
        'MAE': mae,
        'horizon_rmse': horizon_rmse,
        'horizon_mae': horizon_mae,
        'horizon_accuracy': horizon_accuracy,
    }


def train_pipeline(
    train_df,
    val_df,
    user_col,
    time_col,
    target,
):
    logger.info(f'Start training pipeline for target: {target}')

    app_config = load_config('configs/config.yaml')
    config = app_config.get('train', {})

    model_name = config['model_name']
    model_params = config.get('model_params', {})
    training_data_params = config.get('training_data_params', {})

    min_history_len = training_data_params.get('min_history_len', 3)
    max_history_len = training_data_params.get('max_history_len', 12)
    noise_prob = training_data_params.get('noise_prob', 0.3)
    noise_std = training_data_params.get('noise_std', 0.5)
    validation_horizon_weeks = training_data_params.get('validation_horizon_weeks', 12)
    recursive_clip_min = training_data_params.get('recursive_clip_min', -5)
    recursive_clip_max = training_data_params.get('recursive_clip_max', 10)

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

    logger.info(f'Training samples: {len(X_train)}')
    logger.info(f'Feature count: {len(feature_cols)}')

    if X_train.empty or y_train.empty:
        raise ValueError(f'No training samples generated for target: {target}')

    logger.info('Building model...')

    model = build_model(model_name=model_name, params=model_params)
    trainer = Trainer(model)

    trainer.fit(X_train, y_train)

    recursive_metrics = evaluate_recursive_validation(
        model=model,
        df=val_df,
        user_col=user_col,
        time_col=time_col,
        value_col=target,
        min_history_len=min_history_len,
        max_history_len=max_history_len,
        validation_horizon_weeks=validation_horizon_weeks,
        clip_min=recursive_clip_min,
        clip_max=recursive_clip_max,
    )

    if recursive_metrics is None:
        logger.warning(f'No recursive validation samples generated for target: {target}')
    else:
        logger.info('Recursive Validation Result')
        logger.info(f"Sequences: {recursive_metrics['sequence_count']}")
        logger.info(f"Points: {recursive_metrics['point_count']}")
        logger.info(f"RMSE: {recursive_metrics['RMSE']:.4f}")
        logger.info(f"MAE : {recursive_metrics['MAE']:.4f}")
        logger.info(f"Horizon RMSE: {recursive_metrics['horizon_rmse']:.4f}")
        logger.info(f"Horizon MAE : {recursive_metrics['horizon_mae']:.4f}")
        logger.info(f"Horizon Accuracy: {recursive_metrics['horizon_accuracy']:.4f}")

    logger.info(f'{target} model training finished')

    return model, feature_cols
