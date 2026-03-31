from abc import ABC, abstractmethod


class BaseModel(ABC):

    @abstractmethod
    def fit(self, X, y, sample_weight=None):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    @abstractmethod
    def save(self, path):
        pass

    @abstractmethod
    def load(self, path):
        pass
