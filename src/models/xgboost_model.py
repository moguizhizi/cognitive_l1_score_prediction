import joblib

import numpy as np
import xgboost as xgb

from src.models.base_model import BaseModel


class XGBoostModel(BaseModel):
    """
    XGBoost 回归模型

    继承 BaseModel，统一 fit / predict / save / load 接口
    """

    def __init__(self, params: dict | None = None):
        default_params = {
            "n_estimators": 500,
            "learning_rate": 0.05,
            "max_depth": 6,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "objective": "reg:squarederror",
            "n_jobs": -1,
            "random_state": 42,
        }

        if params:
            default_params.update(params)

        self.params = default_params
        self.model = xgb.XGBRegressor(**self.params)

    def fit(self, X, y, sample_weight=None):
        """
        训练模型
        """

        if sample_weight is None:
            self.model.fit(X, y)
        else:
            self.model.fit(X, y, sample_weight=sample_weight)

        return self

    def predict(self, X) -> np.ndarray:
        """
        预测
        """

        return self.model.predict(X)

    def save(self, path):
        """
        保存模型
        """

        joblib.dump(self.model, path)

    def load(self, path):
        """
        加载模型
        """

        self.model = joblib.load(path)

        return self
