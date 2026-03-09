import pandas as pd
from pathlib import Path

from src.data.loader import convert_xlsx_to_parquet

def main():

    convert_xlsx_to_parquet(
        xlsx_path="data/raw/raw_training_weekly_cognitive_ability_scores/raw_training_weekly_cognitive_ability_scores_v2_20251218.xlsx",
        parquet_path="data/raw/raw_training_weekly_cognitive_ability_scores",
        overwrite=True,
    )


if __name__ == "__main__":
    main()
