import pandas as pd
import json
from pathlib import Path

from src.data.loader import convert_xlsx_to_parquet, load_parquet_as_dataframe
from src.data.preprocess import preprocess_dataframe
from src.utils.logger import get_logger, setup_logging
from src.core.raw_training_weekly_cognitive_ability_scores.constants import ColumnName


# 初始化日志系统
setup_logging()

logger = get_logger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent.parent
HOME_BASED_USER_TRAINING = (
    BASE_DIR / "core" / "raw_training_weekly_cognitive_ability_scores"
)


def main():

    logger.info("========== Dataset Build Started ==========")

    # 1. xlsx -> parquet
    logger.info("Step 1: Converting XLSX to Parquet")

    paths = convert_xlsx_to_parquet(
        xlsx_path="data/raw/raw_training_weekly_cognitive_ability_scores/raw_training_weekly_cognitive_ability_scores_v2_20251218.xlsx",
        parquet_path="data/raw/raw_training_weekly_cognitive_ability_scores",
        overwrite=True,
    )

    logger.info(f"Generated parquet files: {paths}")

    # 2. load parquet
    logger.info("Step 2: Loading parquet dataset")

    df = load_parquet_as_dataframe(
        parquet_path="data/raw/raw_training_weekly_cognitive_ability_scores/raw.parquet"
    )

    logger.info(f"Raw dataset shape: {df.shape}")
    logger.info(f"Raw dataset columns: {list(df.columns)}")

    # 3. load column mapping
    logger.info("Step 3: Loading column mapping")

    with open(Path(HOME_BASED_USER_TRAINING) / "column_mapping.json") as f:
        COLUMN_MAPPING = json.load(f)

    logger.info(f"Column mapping size: {len(COLUMN_MAPPING)}")

    # 4. define fields
    logger.info("Step 4: Preparing preprocessing fields")

    date_fields = [
        COLUMN_MAPPING[ColumnName.BIRTH_DATE.value],
    ]

    required_fields = [
        COLUMN_MAPPING[ColumnName.PATIENT_NAME.value],
        COLUMN_MAPPING[ColumnName.PATIENT_ID.value],
        COLUMN_MAPPING[ColumnName.TRAINING_WEEK.value],
        COLUMN_MAPPING[ColumnName.PERCEPTION.value],
        COLUMN_MAPPING[ColumnName.ATTENTION.value],
        COLUMN_MAPPING[ColumnName.MEMORY.value],
        COLUMN_MAPPING[ColumnName.EXECUTIVE_FUNCTION.value],
    ]

    logger.info(f"Date fields: {date_fields}")
    logger.info(f"Required fields: {required_fields}")

    # 5. preprocess
    logger.info("Step 5: Preprocessing dataset")

    df = preprocess_dataframe(
        df=df,
        column_mapping=COLUMN_MAPPING,
        date_fields=date_fields,
        required_fields=required_fields,
    )

    logger.info(f"Processed dataset shape: {df.shape}")

    # 6. save processed dataset
    output_path = (
        "data/processed/raw_training_weekly_cognitive_ability_scores/processed.parquet"
    )

    logger.info(f"Step 6: Saving processed dataset -> {output_path}")

    df.to_parquet(output_path)

    logger.info("Dataset successfully saved")

    logger.info("========== Dataset Build Finished ==========")


if __name__ == "__main__":
    main()
