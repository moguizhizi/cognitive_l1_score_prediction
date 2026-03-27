import json
from pathlib import Path

import pandas as pd

from configs.loader import load_config
from src.core.constants import CognitiveL1DatasetName
from src.data.dataset_meta import load_column_mapping
from src.pipelines.train_pipleline.cognitive_l1 import train_pipeline
from src.utils.logger import get_logger, setup_logging
from src.utils.path_utils import resolve_project_path
from src.core.brain_ability_values_by_training_week_20260324.constants import ColumnName

# 初始化日志系统
setup_logging()
logger = get_logger(__name__)

# 项目根目录
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
DEFAULT_DATASET_KEY = CognitiveL1DatasetName.WEEKLY_BRAIN_ABILITY.value


def load_training_runtime():
    app_config = load_config('configs/config.yaml')
    train_config = load_config('configs/train.yaml')

    dataset_key = train_config.get('dataset_key', DEFAULT_DATASET_KEY)
    column_mapping = load_column_mapping(app_config, dataset_key, BASE_DIR)
    data_dir = resolve_project_path(train_config['split_dir'], BASE_DIR)
    model_dir = resolve_project_path(train_config['checkpoint_dir'], BASE_DIR)
    feature_columns_filename = train_config.get('feature_columns_filename', 'feature_columns.json')

    return train_config, column_mapping, data_dir, model_dir, feature_columns_filename


def resolve_target_columns(column_mapping: dict) -> list[str]:
    return [
        column_mapping[ColumnName.PERCEPTION.value],
        column_mapping[ColumnName.ATTENTION.value],
        column_mapping[ColumnName.MEMORY.value],
        column_mapping[ColumnName.EXECUTIVE_FUNCTION.value],
    ]


def train_all_models(train_df: pd.DataFrame, val_df: pd.DataFrame, column_mapping: dict):
    logger.info('Training all cognitive models')

    models = {}
    feature_dict = {}
    target_cols = resolve_target_columns(column_mapping)
    user_col = column_mapping[ColumnName.PATIENT_ID.value]
    time_col = column_mapping[ColumnName.TRAINING_WEEK.value]

    for target in target_cols:
        logger.info(f'Training target: {target}')

        model, feature_cols = train_pipeline(
            train_df,
            val_df,
            user_col=user_col,
            time_col=time_col,
            target=target,
        )

        models[target] = model
        feature_dict[target] = feature_cols

    logger.info('All models trained successfully')

    return models, feature_dict


def main():
    logger.info('Start cognitive ability model training')

    _, column_mapping, data_dir, model_dir, feature_columns_filename = load_training_runtime()

    train_path = data_dir / 'train.parquet'
    val_path = data_dir / 'val.parquet'

    logger.info(f'Loading train data: {train_path}')
    train_df = pd.read_parquet(train_path)

    logger.info(f'Loading val data: {val_path}')
    val_df = pd.read_parquet(val_path)

    logger.info(f'Train shape: {train_df.shape}')
    logger.info(f'Val shape: {val_df.shape}')

    models, feature_dict = train_all_models(train_df, val_df, column_mapping)

    model_dir.mkdir(parents=True, exist_ok=True)

    for target, model in models.items():
        model_path = model_dir / f'{target}_lightgbm.txt'
        model.save(model_path)
        logger.info(f'Model saved: {model_path}')

    feature_path = model_dir / feature_columns_filename
    with open(feature_path, 'w', encoding='utf-8') as f:
        json.dump(feature_dict, f, indent=4, ensure_ascii=False)

    logger.info(f'Feature columns saved: {feature_path}')
    logger.info('Training finished successfully')


if __name__ == '__main__':
    main()
