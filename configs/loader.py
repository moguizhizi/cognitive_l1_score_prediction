import yaml
from pathlib import Path


def load_config(path: str | None = None) -> dict:
    config_dir = Path(__file__).resolve().parent
    project_root = config_dir.parent

    if path is None:
        resolved_path = config_dir / "config.yaml"
    else:
        candidate = Path(path)

        if candidate.is_absolute():
            resolved_path = candidate
        else:
            candidates = [
                candidate,
                project_root / candidate,
                config_dir / candidate,
                config_dir / candidate.name,
            ]

            resolved_path = next((p for p in candidates if p.exists()), candidates[1])

    if resolved_path.suffix.lower() not in {".yaml", ".yml"}:
        raise ValueError(f"Config file must be a yaml file: {resolved_path}")

    if not resolved_path.exists():
        raise FileNotFoundError(f"Config file not found: {resolved_path}")

    with open(resolved_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}
