import numpy as np
from .base_model import BaseModel


class LeastSquareModel(BaseModel):

    def __init__(self):
        self.coef = None

    def fit(self, X, y):
        # 最小二乘求解
        self.coef = np.linalg.pinv(X) @ y

    def predict(self, X):
        return X @ self.coef
