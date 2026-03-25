from pathlib import Path

from configs.loader import load_config
from src.core.brain_ability_values_by_training_week_20260324.fields import (
    DATE_COLUMNS,
    NUMERIC_COLUMNS,
    REQUIRED_COLUMNS,
)
from src.core.constants import CognitiveL1DatasetName
from src.data.dataset_meta import load_column_mapping, map_column_names
from src.data.loader import convert_xlsx_to_parquet, load_parquet_as_dataframe
from src.data.preprocess import preprocess_dataframe
from src.utils.logger import get_logger, setup_logging
from src.utils.path_utils import resolve_project_path


setup_logging()
logger = get_logger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
DATASET_KEY = CognitiveL1DatasetName.WEEKLY_BRAIN_ABILITY.value


def main():
    logger.info("========== Dataset Build Started ==========")

    config = load_config("configs/config.yaml")

    raw_file_config = next(
        item
        for item in config["xlsx_to_parquet"]["raw_files"]
        if Path(item["xlsx"]).stem == DATASET_KEY
    )
    raw_to_processed = config["raw_to_processed"][DATASET_KEY]
    expected_columns = config["columns"][DATASET_KEY]
    column_mapping = load_column_mapping(config, DATASET_KEY, BASE_DIR)

    xlsx_path = resolve_project_path(raw_file_config["xlsx"], BASE_DIR)
    raw_parquet_dir = resolve_project_path(raw_file_config["parquet"], BASE_DIR)
    processed_parquet_dir = resolve_project_path(
        raw_to_processed["processed"], BASE_DIR
    )

    logger.info("Step 1: Converting XLSX to Parquet")
    paths = convert_xlsx_to_parquet(
        xlsx_path=str(xlsx_path),
        parquet_path=str(raw_parquet_dir),
        overwrite=True,
    )
    logger.info(f"Generated parquet files: {paths}")

    raw_parquet_path = raw_parquet_dir / "raw.parquet"

    logger.info("Step 2: Loading parquet dataset")
    df = load_parquet_as_dataframe(
        parquet_path=str(raw_parquet_path),
        columns=expected_columns,
    )
    logger.info(f"Raw dataset shape: {df.shape}")
    logger.info(f"Raw dataset columns: {list(df.columns)}")

    logger.info("Step 3: Preparing preprocessing fields")
    date_fields = map_column_names(column_mapping, DATE_COLUMNS)
    required_fields = map_column_names(column_mapping, REQUIRED_COLUMNS)
    numeric_fields = map_column_names(column_mapping, NUMERIC_COLUMNS)

    logger.info(f"Date fields: {date_fields}")
    logger.info(f"Required fields: {required_fields}")
    logger.info(f"Numeric fields: {numeric_fields}")

    logger.info("Step 4: Preprocessing dataset")
    df = preprocess_dataframe(
        df=df,
        column_mapping=column_mapping,
        date_fields=date_fields,
        required_fields=required_fields,
        numeric_fields=numeric_fields,
    )
    logger.info(f"Processed dataset shape: {df.shape}")

    logger.info(f"Step 5: Saving processed dataset -> {processed_parquet_dir}")
    processed_parquet_dir.mkdir(parents=True, exist_ok=True)
    output_path = processed_parquet_dir / "processed.parquet"
    df.to_parquet(output_path, index=False)

    logger.info(f"Dataset successfully saved: {output_path}")
    logger.info("========== Dataset Build Finished ==========")


if __name__ == "__main__":
    main()
