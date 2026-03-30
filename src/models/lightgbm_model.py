import lightgbm as lgb

from .base_model import BaseModel


class LightGBMModel(BaseModel):

    def __init__(self, params=None):
        super().__init__()
        self.params = params or {}
        self.model = None

    def fit(self, X, y, sample_weight=None):

        self.model = lgb.LGBMRegressor(**self.params)
        self.model.fit(X, y, sample_weight=sample_weight)

    def predict(self, X):

        if isinstance(self.model, lgb.Booster):
            raw_preds = self.model.predict(X)
            return self.apply_linear_correction(raw_preds)

        raw_preds = self.model.predict(X)
        return self.apply_linear_correction(raw_preds)

    def save(self, path):

        booster = self.model.booster_
        booster.save_model(path)
        self.save_linear_correction(path)

    def load(self, path):

        self.model = lgb.Booster(model_file=path)
        self.load_linear_correction(path)
