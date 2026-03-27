import json
from pathlib import Path
from typing import Iterable

from src.utils.path_utils import resolve_project_path


class ColumnAccessor:
    """
    DataFrame column accessor based on Enum + column mapping.
    """

    def __init__(self, mapping: dict, enum_cls):
        self._cols = {member.name.lower(): mapping[member.value] for member in enum_cls}

    def __getattr__(self, item: str) -> str:
        if item not in self._cols:
            raise AttributeError(f"Column '{item}' not defined")
        return self._cols[item]


def load_column_mapping(config: dict, dataset_key: str, base_dir: Path) -> dict:
    mapping_path = resolve_project_path(config["column_mapping"][dataset_key], base_dir)

    with open(mapping_path, "r", encoding="utf-8") as f:
        return json.load(f)


def map_column_names(column_mapping: dict, column_names: Iterable) -> list[str]:
    return [column_mapping[column_name.value] for column_name in column_names]


def build_column_accessor(column_mapping: dict, enum_cls) -> ColumnAccessor:
    return ColumnAccessor(column_mapping, enum_cls)
