from abc import ABC, abstractmethod
import json
from pathlib import Path

import numpy as np


class BaseModel(ABC):

    def __init__(self):
        self.linear_correction = {
            'enabled': False,
            'slope': 1.0,
            'intercept': 0.0,
        }

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

    def set_linear_correction(self, slope: float, intercept: float, enabled: bool = True):
        self.linear_correction = {
            'enabled': bool(enabled),
            'slope': float(slope),
            'intercept': float(intercept),
        }

    def clear_linear_correction(self):
        self.set_linear_correction(slope=1.0, intercept=0.0, enabled=False)

    def apply_linear_correction(self, predictions):
        preds = np.asarray(predictions, dtype=float)
        if not self.linear_correction.get('enabled', False):
            return preds

        slope = float(self.linear_correction.get('slope', 1.0))
        intercept = float(self.linear_correction.get('intercept', 0.0))
        return slope * preds + intercept

    def get_linear_correction(self) -> dict:
        return {
            'enabled': bool(self.linear_correction.get('enabled', False)),
            'slope': float(self.linear_correction.get('slope', 1.0)),
            'intercept': float(self.linear_correction.get('intercept', 0.0)),
        }

    def _calibration_sidecar_path(self, path) -> Path:
        model_path = Path(path)
        return model_path.with_name(f'{model_path.name}.calibration.json')

    def save_linear_correction(self, path):
        sidecar_path = self._calibration_sidecar_path(path)
        with open(sidecar_path, 'w', encoding='utf-8') as f:
            json.dump(self.get_linear_correction(), f, indent=4, ensure_ascii=False)

    def load_linear_correction(self, path):
        sidecar_path = self._calibration_sidecar_path(path)
        if not sidecar_path.exists():
            self.clear_linear_correction()
            return

        with open(sidecar_path, 'r', encoding='utf-8') as f:
            payload = json.load(f)

        self.set_linear_correction(
            slope=payload.get('slope', 1.0),
            intercept=payload.get('intercept', 0.0),
            enabled=payload.get('enabled', False),
        )
