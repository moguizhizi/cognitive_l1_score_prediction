# src/pipelines/infer_pipeline.py

import pandas as pd
from utils.logger import get_logger

logger = get_logger(__name__)


def predict_next_week(
    model,
    df_user: pd.DataFrame,
    feature_cols,
):
    """
    用用户最近几周数据预测下一周
    """

    logger.info("Start next-week prediction")

    logger.debug(f"Input user dataframe shape: {df_user.shape}")

    df_user = df_user.sort_values("week")

    last = df_user.tail(3)

    if len(last) < 3:
        logger.warning(
            f"Insufficient history for prediction: only {len(last)} weeks available"
        )

    features = {}

    scores = last["score"].tolist()

    logger.debug(f"Last scores used for feature construction: {scores}")

    features["score_lag1"] = scores[-1]
    features["score_lag2"] = scores[-2]
    features["score_lag3"] = scores[-3]

    features["mean_last3"] = sum(scores) / 3
    features["std_last3"] = pd.Series(scores).std()
    features["max_last3"] = max(scores)
    features["min_last3"] = min(scores)

    features["trend_last3"] = scores[-1] - scores[-3]

    logger.debug(f"Constructed features: {features}")

    X = pd.DataFrame([features])[feature_cols]

    logger.debug(f"Feature dataframe for prediction: {X.to_dict(orient='records')}")

    pred = model.predict(X)[0]

    logger.info(f"Prediction result: {pred}")

    return pred