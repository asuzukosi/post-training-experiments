"""yaml config loading for prepare stages."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from prepare.paths import CONFIG_DIR


def load_config(name: str, config_dir: Path = CONFIG_DIR) -> dict[str, Any]:
    import yaml

    path = config_dir / f"{name}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"missing config: {path}")
    with path.open() as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"config must be a mapping: {path}")
    return cfg
