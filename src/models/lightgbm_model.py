import lightgbm as lgb

from .base_model import BaseModel


class LightGBMModel(BaseModel):

    def __init__(self, params=None):
        self.params = params or {}
        self.model = None

    def fit(self, X, y, sample_weight=None):

        self.model = lgb.LGBMRegressor(**self.params)
        self.model.fit(X, y, sample_weight=sample_weight)

    def predict(self, X):

        if isinstance(self.model, lgb.Booster):
            return self.model.predict(X)

        return self.model.predict(X)

    def save(self, path):

        booster = self.model.booster_
        booster.save_model(path)

    def load(self, path):

        self.model = lgb.Booster(model_file=path)
