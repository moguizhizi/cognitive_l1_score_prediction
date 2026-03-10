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

    paths = convert_xlsx_to_parquet(
        xlsx_path="data/raw/raw_training_weekly_cognitive_ability_scores/raw_training_weekly_cognitive_ability_scores_v2_20251218.xlsx",
        parquet_path="data/raw/raw_training_weekly_cognitive_ability_scores",
        overwrite=True,
    )

    logger.info(paths)

    df = load_parquet_as_dataframe(
        parquet_path="data/raw/raw_training_weekly_cognitive_ability_scores/raw.parquet"
    )

    with open(Path(HOME_BASED_USER_TRAINING) / "column_mapping.json") as f:
        COLUMN_MAPPING = json.load(f)

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

    df = preprocess_dataframe(
        df=df,
        column_mapping=COLUMN_MAPPING,
        date_fields=date_fields,
        required_fields=required_fields,
    )


if __name__ == "__main__":
    main()
