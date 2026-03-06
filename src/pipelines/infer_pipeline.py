# src/pipelines/infer_pipeline.py

import pandas as pd


def predict_next_week(
    model,
    df_user: pd.DataFrame,
    feature_cols,
):
    """
    用用户最近几周数据预测下一周
    """

    df_user = df_user.sort_values("week")

    last = df_user.tail(3)

    features = {}

    scores = last["score"].tolist()

    features["score_lag1"] = scores[-1]
    features["score_lag2"] = scores[-2]
    features["score_lag3"] = scores[-3]

    features["mean_last3"] = sum(scores) / 3
    features["std_last3"] = pd.Series(scores).std()
    features["max_last3"] = max(scores)
    features["min_last3"] = min(scores)

    features["trend_last3"] = scores[-1] - scores[-3]

    X = pd.DataFrame([features])[feature_cols]

    pred = model.predict(X)[0]

    return pred
