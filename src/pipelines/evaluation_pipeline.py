import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    root_mean_squared_error,
)

from utils.logger import get_logger
from src.pipelines.infer_pipeline import predict_next_week

logger = get_logger(__name__)


def evaluate_model(
    model,
    df: pd.DataFrame,
    feature_cols: list[str],
):
    """
    评估模型效果

    评估逻辑：
    - 每个用户按 week 排序
    - 使用前 3 周数据预测第 4 周
    - 采用 rolling evaluation（滑动窗口）

    Example:
        week1,2,3 -> 预测 week4
        week2,3,4 -> 预测 week5
    """

    logger.info("Start model evaluation")

    predictions = []
    targets = []

    user_ids = df["user_id"].unique()

    logger.info(f"Total users for evaluation: {len(user_ids)}")

    evaluation_samples = 0

    for user_id in user_ids:

        df_user = df[df["user_id"] == user_id].sort_values("week")

        # 至少需要4周数据
        if len(df_user) < 4:
            continue

        for i in range(3, len(df_user)):

            history = df_user.iloc[i - 3 : i]
            target = df_user.iloc[i]["score"]

            try:

                pred = predict_next_week(
                    model=model,
                    df_user=history,
                    feature_cols=feature_cols,
                )

                predictions.append(pred)
                targets.append(target)

                evaluation_samples += 1

            except Exception as e:

                logger.warning(f"Prediction failed for user {user_id}, index {i}: {e}")

    if not predictions:
        logger.error("No predictions generated during evaluation")
        return None

    # ===== metrics =====
    mae = mean_absolute_error(targets, predictions)
    mse = mean_squared_error(targets, predictions)
    rmse = root_mean_squared_error(targets, predictions)
    r2 = r2_score(targets, predictions)

    logger.info("Evaluation finished")
    logger.info(f"Evaluation samples: {evaluation_samples}")
    logger.info(f"MAE: {mae:.4f}")
    logger.info(f"MSE: {mse:.4f}")
    logger.info(f"RMSE: {rmse:.4f}")
    logger.info(f"R2: {r2:.4f}")

    return {
        "samples": evaluation_samples,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2,
    }
