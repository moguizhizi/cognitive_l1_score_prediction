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

def compute_alpha(current, c=150, s=10):
    """
    计算权重 k ∈ (0,1)

    参数：
    - current: 当前值
    - c: 拐点（越小越保守）
    - s: 平滑程度（越小下降越快）

    性质：
    - current ↑ → k ↓
    - current → 160 → k → 0
    """
    return 1 / (1 + np.exp((current - c) / s))


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
    alpha_c: float = 150,
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
            alpha = compute_alpha(current, c=alpha_c)

            target = future_value + alpha * max(0, current - future_value)

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



def compute_M(N, current, range_val, alpha_c=150):
    """
    计算最终修正值 M

    约束：
    - M > current
    - M < 160
    - current 越大 → 增长越保守
    """

    # --- Step 0: 修正 range_val（关键） ---
    range_val = max(range_val, 0.0)

    # --- Step 1: 计算 k ---
    k = compute_alpha(current, c=alpha_c)

    # --- Step 2: 防止预测下降 ---
    # 至少增长一个极小值 or range_val
    min_increase = max(1e-6, range_val)
    N = min(N, current + min_increase)

    # --- Step 3: 上界控制 ---
    max_cap = 160 - 1  # 你这里留了 buffer（很好）

    # 可增长空间
    delta = min(N - current, max_cap - current)

    # --- Step 4: 插值 ---
    M = current + k * delta

    # --- Step 5: 下界保护 ---
    M = max(M, current + 1e-6)

    return M


def direct_horizon_forecast(
    model,
    history: list[float],
    current: float,
    max_history_len: int,
    feature_cols: list[str],
    alpha_c: float,
) -> float:
    effective_history = history[-max_history_len:]

    feats = build_features(effective_history, max_history_len)
    feats['hist_len'] = len(effective_history)
    feats['current'] = current

    # --- 计算 range ---
    if len(effective_history) > 0:
        range_val = max(effective_history) - min(effective_history)
    else:
        range_val = 0.0

    X = pd.DataFrame([feats]).reindex(columns=feature_cols, fill_value=np.nan)
    pred = float(model.predict(X)[0])
    pred = compute_M(pred, current, range_val, alpha_c=alpha_c)
    return pred


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
    alpha_c: float,
):
    df = df.sort_values([user_col, time_col])
    predictions = []
    targets = []
    currents = []   # 👈 新增
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
                alpha_c=alpha_c,
            )

            predictions.append(pred_target)
            targets.append(actual_target)
            currents.append(current)   # 👈 记录 current
            sequence_count += 1

    if not predictions:
        return None

    # 转 numpy
    predictions = np.array(predictions)
    targets = np.array(targets)
    currents = np.array(currents)

    # =====================
    # 全量指标
    # =====================
    rmse = np.sqrt(mean_squared_error(targets, predictions))
    mae = mean_absolute_error(targets, predictions)

    relative_error = np.mean(
        np.abs(predictions - targets)
        / np.maximum(np.abs(targets), 1e-8)
    )
    accuracy = max(0.0, 1.0 - float(relative_error))

    # =====================
    # 上升子集指标（关键）
    # =====================
    mask_up = targets >= currents

    if mask_up.sum() > 0:
        rmse_up = np.sqrt(mean_squared_error(
            targets[mask_up],
            predictions[mask_up]
        ))
        mae_up = mean_absolute_error(
            targets[mask_up],
            predictions[mask_up]
        )
    else:
        rmse_up = None
        mae_up = None

    # =====================
    # （可选）下降子集
    # =====================
    mask_down = targets < currents

    if mask_down.sum() > 0:
        rmse_down = np.sqrt(mean_squared_error(
            targets[mask_down],
            predictions[mask_down]
        ))
    else:
        rmse_down = None

    return {
        # 基础信息
        'sequence_count': sequence_count,
        'point_count': len(predictions),

        # 全量
        'RMSE': rmse,
        'MAE': mae,

        # 上升子集（重点看这个）
        'RMSE_up': rmse_up,
        'MAE_up': mae_up,
        'up_ratio': float(mask_up.mean()),  # 上升样本比例

        # （可选）下降子集
        'RMSE_down': rmse_down,

        # 其他
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
    alpha_c = training_data_params.get('alpha_c', 150)
    evaluation_strategy = 'direct_horizon'

    logger.info(f'Model name: {model_name}')
    logger.info(f'Model params: {model_params}')
    logger.info(f'Training data params: {training_data_params}')
    logger.info(f'Target horizon weeks: {target_horizon_weeks}')
    logger.info(f'Alpha c: {alpha_c}')
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
        alpha_c=alpha_c,
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
        alpha_c=alpha_c,
    )

    if recursive_metrics is None:
        logger.warning(f'No validation samples generated for target: {target}')
    else:
        logger.info('====== Validation Result ======')

        logger.info(f"Sequences: {recursive_metrics['sequence_count']}")
        logger.info(f"Points   : {recursive_metrics['point_count']}")

        # =====================
        # 全量指标
        # =====================
        logger.info(f"[ALL] RMSE: {recursive_metrics['RMSE']:.4f}")
        logger.info(f"[ALL] MAE : {recursive_metrics['MAE']:.4f}")

        # =====================
        # 上升子集（重点）
        # =====================
        if recursive_metrics['RMSE_up'] is not None:
            logger.info(f"[UP ] RMSE: {recursive_metrics['RMSE_up']:.4f}")
            logger.info(f"[UP ] MAE : {recursive_metrics['MAE_up']:.4f}")
        else:
            logger.info("[UP ] RMSE: None (no upward samples)")
            logger.info("[UP ] MAE : None (no upward samples)")

        logger.info(f"[UP ] Ratio: {recursive_metrics['up_ratio']:.4f}")

        # =====================
        # 下降子集（辅助）
        # =====================
        if recursive_metrics['RMSE_down'] is not None:
            logger.info(f"[DOWN] RMSE: {recursive_metrics['RMSE_down']:.4f}")
        else:
            logger.info("[DOWN] RMSE: None (no downward samples)")

        # =====================
        # Horizon（保留）
        # =====================
        logger.info(f"Horizon RMSE: {recursive_metrics['horizon_rmse']:.4f}")
        logger.info(f"Horizon MAE : {recursive_metrics['horizon_mae']:.4f}")
        logger.info(f"Horizon Accuracy: {recursive_metrics['horizon_accuracy']:.4f}")

        logger.info('================================')

    logger.info(f'{target} model training finished')

    return model, feature_cols, recursive_metrics
