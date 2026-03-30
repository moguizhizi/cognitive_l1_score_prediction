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


def fit_linear_correction(predictions, targets) -> dict:
    preds = np.asarray(predictions, dtype=float).reshape(-1)
    y_true = np.asarray(targets, dtype=float).reshape(-1)

    if preds.size == 0 or y_true.size == 0:
        return {
            'enabled': False,
            'slope': 1.0,
            'intercept': 0.0,
            'sample_count': 0,
        }

    design = np.column_stack([preds, np.ones(len(preds))])
    slope, intercept = np.linalg.lstsq(design, y_true, rcond=None)[0]

    return {
        'enabled': True,
        'slope': float(slope),
        'intercept': float(intercept),
        'sample_count': int(len(preds)),
    }


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
    target_mode: str = 'delta',
    target_horizon_weeks: int = 1,
):
    df = df.sort_values([user_col, time_col])
    rows = []

    for _, user_df in df.groupby(user_col):
        values = user_df[value_col].values.astype(float)

        if len(values) < min_history_len + target_horizon_weeks:
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
            if target_mode == 'next_value':
                target = future_value
            else:
                target = future_value - current

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


def recursive_forecast(
    model,
    history: list[float],
    current: float,
    steps: int,
    clip_min: float,
    clip_max: float,
    max_history_len: int,
    feature_cols: list[str],
    target_mode: str = 'delta',
) -> list[float]:
    history = history.copy()
    preds = []
    current_value = float(current)

    for _ in range(steps):
        effective_history = history[-max_history_len:]

        feats = build_features(effective_history, max_history_len)
        feats['hist_len'] = len(effective_history)
        feats['current'] = current_value

        X = pd.DataFrame([feats]).reindex(columns=feature_cols, fill_value=np.nan)
        raw_pred = model.predict(X)[0]
        if target_mode == 'next_value':
            next_value = float(raw_pred)
        else:
            delta = np.clip(raw_pred, clip_min, clip_max)
            next_value = current_value + delta
        preds.append(float(next_value))

        history.append(float(current_value))
        current_value = float(next_value)

    return preds


def direct_horizon_forecast(
    model,
    history: list[float],
    current: float,
    clip_min: float,
    clip_max: float,
    max_history_len: int,
    feature_cols: list[str],
    target_mode: str = 'delta',
) -> float:
    effective_history = history[-max_history_len:]

    feats = build_features(effective_history, max_history_len)
    feats['hist_len'] = len(effective_history)
    feats['current'] = current

    X = pd.DataFrame([feats]).reindex(columns=feature_cols, fill_value=np.nan)
    raw_pred = model.predict(X)[0]
    if target_mode == 'next_value':
        return float(raw_pred)

    delta = np.clip(raw_pred, clip_min, clip_max)
    return float(current + delta)


def evaluate_direct_validation(
    model,
    df: pd.DataFrame,
    user_col: str,
    time_col: str,
    value_col: str,
    min_history_len: int,
    max_history_len: int,
    target_horizon_weeks: int,
    clip_min: float,
    clip_max: float,
    feature_cols: list[str],
    target_mode: str = 'delta',
):
    df = df.sort_values([user_col, time_col])
    predictions = []
    targets = []
    sequence_count = 0

    for _, user_df in df.groupby(user_col):
        values = user_df[value_col].values.astype(float)
        if len(values) < min_history_len + target_horizon_weeks:
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
                clip_min=clip_min,
                clip_max=clip_max,
                max_history_len=max_history_len,
                feature_cols=feature_cols,
                target_mode=target_mode,
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
        'evaluation_strategy': 'direct',
    }


def build_calibration_dataset(
    df: pd.DataFrame,
    user_col: str,
    time_col: str,
    value_col: str,
    min_history_len: int,
    max_history_len: int,
    target_mode: str,
    target_horizon_weeks: int,
):
    return build_training_data(
        df=df,
        user_col=user_col,
        time_col=time_col,
        value_col=value_col,
        min_history_len=min_history_len,
        max_history_len=max_history_len,
        noise_prob=0.0,
        noise_std=0.0,
        randomize_history_len=False,
        add_noise=False,
        target_mode=target_mode,
        target_horizon_weeks=target_horizon_weeks,
    )


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
    feature_cols: list[str],
    target_mode: str = 'delta',
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
                feature_cols=feature_cols,
                target_mode=target_mode,
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
    target_mode_override=None,
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
    validation_horizon_weeks = training_data_params.get('validation_horizon_weeks', 12)
    target_horizon_weeks = training_data_params.get('target_horizon_weeks', 1)
    recursive_clip_min = training_data_params.get('recursive_clip_min', -5)
    recursive_clip_max = training_data_params.get('recursive_clip_max', 10)
    random_seed = training_data_params.get('random_seed', 42)
    target_mode = target_mode_override or training_data_params.get('target_mode', 'delta')
    evaluation_strategy = 'direct' if target_horizon_weeks > 1 else 'recursive'

    random.seed(random_seed)
    np.random.seed(random_seed)
    model_params.setdefault('random_state', random_seed)

    logger.info(f'Model name: {model_name}')
    logger.info(f'Model params: {model_params}')
    logger.info(f'Training data params: {training_data_params}')
    logger.info(f'Random seed: {random_seed}')
    logger.info(f'Target mode: {target_mode}')
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
        target_mode=target_mode,
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

    X_calibration, y_calibration, _ = build_calibration_dataset(
        df=val_df,
        user_col=user_col,
        time_col=time_col,
        value_col=target,
        min_history_len=min_history_len,
        max_history_len=max_history_len,
        target_mode=target_mode,
        target_horizon_weeks=target_horizon_weeks,
    )

    linear_correction = {
        'enabled': False,
        'slope': 1.0,
        'intercept': 0.0,
        'sample_count': 0,
    }
    if not X_calibration.empty and not y_calibration.empty:
        calibration_predictions = model.predict(X_calibration)
        linear_correction = fit_linear_correction(calibration_predictions, y_calibration)
        model.set_linear_correction(
            slope=linear_correction['slope'],
            intercept=linear_correction['intercept'],
            enabled=linear_correction['enabled'],
        )
        logger.info(
            'Linear correction fitted: '
            f"slope={linear_correction['slope']:.6f}, "
            f"intercept={linear_correction['intercept']:.6f}, "
            f"samples={linear_correction['sample_count']}"
        )
    else:
        logger.warning(f'No calibration samples generated for target: {target}')

    if evaluation_strategy == 'direct':
        recursive_metrics = evaluate_direct_validation(
            model=model,
            df=val_df,
            user_col=user_col,
            time_col=time_col,
            value_col=target,
            min_history_len=min_history_len,
            max_history_len=max_history_len,
            target_horizon_weeks=target_horizon_weeks,
            clip_min=recursive_clip_min,
            clip_max=recursive_clip_max,
            feature_cols=feature_cols,
            target_mode=target_mode,
        )
    else:
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
            feature_cols=feature_cols,
            target_mode=target_mode,
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
        logger.info(
            'Linear Correction: '
            f"enabled={linear_correction['enabled']}, "
            f"slope={linear_correction['slope']:.6f}, "
            f"intercept={linear_correction['intercept']:.6f}"
        )

    logger.info(f'{target} model training finished')

    return model, feature_cols, recursive_metrics, linear_correction
