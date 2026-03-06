import lightgbm as lgb
from .base_model import BaseModel


class LightGBMModel(BaseModel):

    def __init__(self, params=None):
        self.params = params or {}
        self.model = None

    def fit(self, X, y):
        self.model = lgb.LGBMRegressor(**self.params)
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)
