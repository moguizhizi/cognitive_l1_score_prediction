from .lightgbm_model import LightGBMModel
from .least_square_model import LeastSquareModel

def create_model(model_name, config):

    if model_name == "lightgbm":
        return LightGBMModel(config)

    elif model_name == "least_square":
        return LeastSquareModel()

    else:
        raise ValueError(f"Unknown model: {model_name}")