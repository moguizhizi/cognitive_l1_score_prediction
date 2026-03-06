from src.data.loader import load_data
from src.features.feature_builder import build_features
from src.models.model_factory import create_model


def train_pipeline(config):

    # 1 读取数据
    df = load_data(config["data_path"])

    # 2 特征工程
    X, y = build_features(df)

    # 3 创建模型
    model = create_model(config["model_name"], config.get("model_params", {}))

    # 4 训练
    model.fit(X, y)

    # 5 预测
    preds = model.predict(X)

    return model, preds
