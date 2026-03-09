import pandas as pd
from pathlib import Path

from src.data.loader import convert_xlsx_to_parquet
from src.utils.logger import get_logger, setup_logging


# 初始化日志系统
setup_logging()

logger = get_logger(__name__)


def main():

    paths = convert_xlsx_to_parquet(
        xlsx_path="data/raw/raw_training_weekly_cognitive_ability_scores/raw_training_weekly_cognitive_ability_scores_v2_20251218.xlsx",
        parquet_path="data/raw/raw_training_weekly_cognitive_ability_scores",
        overwrite=True,
    )

    logger.info(paths)


if __name__ == "__main__":
    main()
