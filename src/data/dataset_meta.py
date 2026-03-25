import json
from pathlib import Path
from typing import Iterable

from src.utils.path_utils import resolve_project_path


def load_column_mapping(config: dict, dataset_key: str, base_dir: Path) -> dict:
    mapping_path = resolve_project_path(config["column_mapping"][dataset_key], base_dir)

    with open(mapping_path, "r", encoding="utf-8") as f:
        return json.load(f)


def map_column_names(column_mapping: dict, column_names: Iterable) -> list[str]:
    return [column_mapping[column_name.value] for column_name in column_names]
