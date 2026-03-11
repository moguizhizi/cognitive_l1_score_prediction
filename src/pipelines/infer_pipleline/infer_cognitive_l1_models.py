# src/pipelines/infer_pipeline.py

import pandas as pd
import json
import joblib
from datetime import datetime
from src.core.raw_training_weekly_cognitive_ability_scores.constants import ColumnName
from src.utils.logger import get_logger, setup_logging
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    root_mean_squared_error,
)
from pathlib import Path

# 初始化日志系统
setup_logging()
logger = get_logger(__name__)

# 项目根目录
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent

COGNITIVE_ABILITY_TRAINING = (
    BASE_DIR / "src" / "core" / "raw_training_weekly_cognitive_ability_scores"
)

with open(Path(COGNITIVE_ABILITY_TRAINING) / "column_mapping.json") as f:
    COLUMN_MAPPING = json.load(f)

TARGET_COLS = [
    COLUMN_MAPPING[ColumnName.PERCEPTION.value],
    COLUMN_MAPPING[ColumnName.ATTENTION.value],
    COLUMN_MAPPING[ColumnName.MEMORY.value],
    COLUMN_MAPPING[ColumnName.EXECUTIVE_FUNCTION.value],
]


def predict_next_week(model, df_user: pd.DataFrame, score_col: str):
    """
    用用户最近几周数据预测下一周
    """

    logger.info("Start next-week prediction")

    logger.debug(f"Input user dataframe shape: {df_user.shape}")

    df_user = df_user.sort_values(
        COLUMN_MAPPING[ColumnName.TRAINING_WEEK.value], ascending=True
    )

    last = df_user.tail(3)

    if len(last) < 3:
        logger.warning(
            f"Insufficient history for prediction: only {len(last)} weeks available"
        )

    features = {}

    scores = last[score_col].tolist()

    logger.debug(f"Last scores used for feature construction: {scores}")

    features[f"{score_col}_lag1"] = scores[-1]
    features[f"{score_col}_lag2"] = scores[-2]
    features[f"{score_col}_lag3"] = scores[-3]

    features["mean_last3"] = sum(scores) / 3
    features["std_last3"] = pd.Series(scores).std()
    features["max_last3"] = max(scores)
    features["min_last3"] = min(scores)

    features["trend_last3"] = scores[-1] - scores[-3]

    logger.debug(f"Constructed features: {features}")

    X = pd.DataFrame([features])

    logger.debug(f"Feature dataframe for prediction: {X.to_dict(orient='records')}")

    pred = model.predict(X)[0]

    logger.info(f"Prediction result: {pred}")

    return pred


def evaluate_model(
    model,
    df: pd.DataFrame,
    target_col: str,
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

    user_ids = df[COLUMN_MAPPING[ColumnName.PATIENT_ID.value]].unique()

    logger.info(f"Total users for evaluation: {len(user_ids)}")

    evaluation_samples = 0

    for user_id in user_ids:

        df_user = df[
            df[COLUMN_MAPPING[ColumnName.PATIENT_ID.value]] == user_id
        ].sort_values(COLUMN_MAPPING[ColumnName.TRAINING_WEEK.value], ascending=True)

        # 至少需要4周数据
        if len(df_user) < 4:
            continue

        for i in range(3, len(df_user)):

            history = df_user.iloc[i - 3 : i]
            target = df_user.iloc[i][target_col]

            try:

                pred = predict_next_week(
                    model=model,
                    df_user=history,
                    score_col=target_col,
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


def main():

    logger.info("Start evaluation pipeline")

    # ==========================
    # 1 读取 parquet 数据
    # ==========================

    data_path = (
        BASE_DIR
        / "data"
        / "splitter"
        / "raw_training_weekly_cognitive_ability_scores"
        / "test.parquet"
    )

    logger.info(f"Loading evaluation dataset: {data_path}")

    df = pd.read_parquet(data_path)

    logger.info(f"Dataset shape: {df.shape}")

    # ==========================
    # 2 模型路径
    # ==========================

    checkpoint_dir = BASE_DIR / "checkpoints"

    model_paths = {
        COLUMN_MAPPING[ColumnName.PERCEPTION.value]: checkpoint_dir
        / "cognitive_l1"
        / f"{COLUMN_MAPPING[ColumnName.PERCEPTION.value]}_lightgbm.pkl",
        COLUMN_MAPPING[ColumnName.ATTENTION.value]: checkpoint_dir
        / "cognitive_l1"
        / f"{COLUMN_MAPPING[ColumnName.ATTENTION.value]}_lightgbm.pkl",
        COLUMN_MAPPING[ColumnName.MEMORY.value]: checkpoint_dir
        / "cognitive_l1"
        / f"{COLUMN_MAPPING[ColumnName.MEMORY.value]}_lightgbm.pkl",
        COLUMN_MAPPING[ColumnName.EXECUTIVE_FUNCTION.value]: checkpoint_dir
        / "cognitive_l1"
        / f"{COLUMN_MAPPING[ColumnName.EXECUTIVE_FUNCTION.value]}_lightgbm.pkl",
    }

    results = {}

    # ==========================
    # 3 逐模型评估
    # ==========================

    for target_name, model_path in model_paths.items():

        logger.info(f"Loading model: {model_path}")

        model = joblib.load(model_path)

        metrics = evaluate_model(
            model=model,
            df=df,
            target_col=target_name,
        )

        results[target_name] = metrics

    # ==========================
    # 4 保存评估结果
    # ==========================

    result_dir = BASE_DIR / "experiments" / "evaluation"
    result_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    result_path = result_dir / f"evaluation_{timestamp}.json"

    with open(result_path, "w") as f:
        json.dump(results, f, indent=4)

    logger.info(f"Evaluation results saved to: {result_path}")


if __name__ == "__main__":
    main()
