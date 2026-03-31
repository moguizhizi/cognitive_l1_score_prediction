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


def build_features(history: list[float], max_history_len: int) -> dict:
    arr = np.array(history, dtype=float)
    feats = {}

    for i in range(1, max_history_len + 1):
        feats[f'lag_{i}'] = arr[-i] if len(arr) >= i else np.nan

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
    feats['diff_1'] = arr[-1] - arr[-2] if len(arr) >= 2 else 0.0
    feats['diff_2'] = arr[-2] - arr[-3] if len(arr) >= 3 else 0.0
    feats['diff_last_vs_mean_4'] = arr[-1] - (arr[-4:].mean() if len(arr) >= 4 else arr.mean())
    feats['diff_mean_4_12'] = (arr[-4:].mean() - arr[-12:].mean()) if len(arr) >= 12 else 0.0
    feats['range_4'] = (arr[-4:].max() - arr[-4:].min()) if len(arr) >= 4 else (arr.max() - arr.min())
    feats['range_12'] = (arr[-12:].max() - arr[-12:].min()) if len(arr) >= 12 else (arr.max() - arr.min())
    feats['std_ratio_4_12'] = ((arr[-4:].std() + 1e-6) / (arr[-12:].std() + 1e-6)) if len(arr) >= 12 else 1.0
    feats['trend_4'] = np.polyfit(np.arange(4), arr[-4:], 1)[0] if len(arr) >= 4 else feats['trend']
    feats['trend_8'] = np.polyfit(np.arange(8), arr[-8:], 1)[0] if len(arr) >= 8 else feats['trend']
    feats['last_vs_min'] = arr[-1] - arr.min()
    feats['last_vs_max'] = arr[-1] - arr.max()

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
    target_horizon_weeks: int = 1,
):
    df = df.sort_values([user_col, time_col])
    rows = []

    for _, user_df in df.groupby(user_col):
        values = user_df[value_col].values.astype(float)

        if len(values) < min_history_len + target_horizon_weeks + 1:
            continue

        for i in range(min_history_len, len(values) - target_horizon_weeks):
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
            future_value = values[i + target_horizon_weeks]
            target = future_value

            feats = build_features(history, max_history_len)
            feats['hist_len'] = hist_len
            feats['current'] = current
            feats['_target'] = target
            rows.append(feats)

    if not rows:
        return pd.DataFrame(), pd.Series(dtype='float64'), []

    feature_df = pd.DataFrame(rows)
    y = feature_df.pop('_target')

    return feature_df, y, feature_df.columns.tolist()


def direct_horizon_forecast(
    model,
    history: list[float],
    current: float,
    max_history_len: int,
    feature_cols: list[str],
) -> float:
    effective_history = history[-max_history_len:]

    feats = build_features(effective_history, max_history_len)
    feats['hist_len'] = len(effective_history)
    feats['current'] = current

    X = pd.DataFrame([feats]).reindex(columns=feature_cols, fill_value=np.nan)
    return float(model.predict(X)[0])


def evaluate_direct_validation(
    model,
    df: pd.DataFrame,
    user_col: str,
    time_col: str,
    value_col: str,
    min_history_len: int,
    max_history_len: int,
    target_horizon_weeks: int,
    feature_cols: list[str],
):
    df = df.sort_values([user_col, time_col])
    predictions = []
    targets = []
    sequence_count = 0

    for _, user_df in df.groupby(user_col):
        values = user_df[value_col].values.astype(float)
        if len(values) < min_history_len + target_horizon_weeks + 1:
            continue

        for i in range(min_history_len, len(values) - target_horizon_weeks):
            hist_len = min(max_history_len, i)
            history = values[i - hist_len:i].tolist()
            current = float(values[i])
            actual_target = float(values[i + target_horizon_weeks])

            pred_target = direct_horizon_forecast(
                model=model,
                history=history,
                current=current,
                max_history_len=max_history_len,
                feature_cols=feature_cols,
            )

            predictions.append(pred_target)
            targets.append(actual_target)
            sequence_count += 1

    if not predictions:
        return None

    rmse = np.sqrt(mean_squared_error(targets, predictions))
    mae = mean_absolute_error(targets, predictions)
    relative_error = np.mean(
        np.abs(np.array(predictions) - np.array(targets))
        / np.maximum(np.abs(np.array(targets)), 1e-8)
    )
    accuracy = max(0.0, 1.0 - float(relative_error))

    return {
        'sequence_count': sequence_count,
        'point_count': len(predictions),
        'RMSE': rmse,
        'MAE': mae,
        'horizon_rmse': rmse,
        'horizon_mae': mae,
        'horizon_accuracy': accuracy,
        'evaluation_strategy': 'direct_horizon',
        'target_horizon_weeks': target_horizon_weeks,
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
    model_params = dict(config.get('model_params', {}))
    training_data_params = dict(config.get('training_data_params', {}))
    target_overrides = config.get('target_overrides', {}).get(target, {})
    model_params.update(target_overrides.get('model_params', {}))
    training_data_params.update(target_overrides.get('training_data_params', {}))

    min_history_len = training_data_params.get('min_history_len', 3)
    max_history_len = training_data_params.get('max_history_len', 12)
    noise_prob = training_data_params.get('noise_prob', 0.3)
    noise_std = training_data_params.get('noise_std', 0.5)
    target_horizon_weeks = training_data_params.get('target_horizon_weeks', 12)
    evaluation_strategy = 'direct_horizon'

    logger.info(f'Model name: {model_name}')
    logger.info(f'Model params: {model_params}')
    logger.info(f'Training data params: {training_data_params}')
    logger.info(f'Target horizon weeks: {target_horizon_weeks}')
    logger.info(f'Evaluation strategy: {evaluation_strategy}')
    if target_overrides:
        logger.info(f'Target overrides applied: {target_overrides}')

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
        add_noise=bool(noise_prob and noise_std),
        target_horizon_weeks=target_horizon_weeks,
    )

    logger.info(f'Training samples: {len(X_train)}')
    logger.info(f'Feature count: {len(feature_cols)}')

    if X_train.empty or y_train.empty:
        raise ValueError(f'No training samples generated for target: {target}')

    logger.info('Building model...')

    model = build_model(model_name=model_name, params=model_params)
    trainer = Trainer(model)

    trainer.fit(X_train, y_train)

    recursive_metrics = evaluate_direct_validation(
        model=model,
        df=val_df,
        user_col=user_col,
        time_col=time_col,
        value_col=target,
        min_history_len=min_history_len,
        max_history_len=max_history_len,
        target_horizon_weeks=target_horizon_weeks,
        feature_cols=feature_cols,
    )

    if recursive_metrics is None:
        logger.warning(f'No validation samples generated for target: {target}')
    else:
        logger.info('Validation Result')
        logger.info(f"Sequences: {recursive_metrics['sequence_count']}")
        logger.info(f"Points: {recursive_metrics['point_count']}")
        logger.info(f"RMSE: {recursive_metrics['RMSE']:.4f}")
        logger.info(f"MAE : {recursive_metrics['MAE']:.4f}")
        logger.info(f"Horizon RMSE: {recursive_metrics['horizon_rmse']:.4f}")
        logger.info(f"Horizon MAE : {recursive_metrics['horizon_mae']:.4f}")
        logger.info(f"Horizon Accuracy: {recursive_metrics['horizon_accuracy']:.4f}")

    logger.info(f'{target} model training finished')

    return model, feature_cols, recursive_metrics
