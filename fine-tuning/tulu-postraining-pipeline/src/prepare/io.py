"""read/write processed datasets under data/processed/."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from prepare.paths import resolve_path


def save_rows(rows: list[dict[str, Any]], path: str | Path) -> Path:
    from datasets import Dataset

    out = resolve_path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = [dict(r) for r in rows]
    Dataset.from_list(payload).save_to_disk(str(out))
    print(f"wrote {len(payload)} rows -> {out}")
    return out


def load_processed_rows(path: str | Path) -> list[dict[str, Any]]:
    from datasets import load_from_disk

    p = resolve_path(path)
    ds = load_from_disk(str(p))
    return [dict(ds[i]) for i in range(len(ds))]


def load_ultrafeedback(cfg: dict[str, Any], *, default_split: str):
    """load HuggingFaceH4/ultrafeedback_binarized (or cfg override)."""
    from datasets import load_dataset

    dataset_id = cfg["dataset"]
    split = cfg.get("split", default_split)
    print(f"loading ultrafeedback: {dataset_id} split={split}")
    ds = load_dataset(dataset_id, split=split)
    print(f"ultrafeedback {split} rows: {len(ds)}")
    return ds
