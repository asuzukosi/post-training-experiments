"""shared helpers for the train smokes.

each smoke defaults to a handful of synthetic rows so it runs offline, but can be
pointed at real data instead — which is what you want on a gpu box, where synthetic
rows exercise the api but not the tokenizer, the chat template, or realistic
sequence lengths.

    SFT_SMOKE_DATASET=data/processed/sft_25k                  # saved to disk
    SFT_SMOKE_DATASET=allenai/tulu-3-sft-mixture              # hub id
    SFT_SMOKE_DATASET=allenai/tulu-3-sft-mixture:train        # hub id + split
    SFT_SMOKE_DATASET=HuggingFaceH4/ultrafeedback_binarized::train_prefs
    SFT_SMOKE_DATASET=/abs/path/rows.jsonl                    # local jsonl/json

`<STAGE>_SMOKE_N` caps rows (default 8) so a smoke stays a smoke.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Sequence

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_N = 8

# overrides every smoke wants: two steps, no accumulation, fp32, no saving mid-run
BASE_OVERRIDES: dict[str, Any] = {
    "max_steps": 2,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 1,
    "max_length": 256,
    "bf16": False,
    "fp16": False,
    "logging_steps": 1,
    "save_strategy": "no",
    # smokes must not phone home; trainers read use_wandb via cfg_use_wandb
    "use_wandb": False,
    "warmup_ratio": 0.0,
}


def _messages(n: int) -> list[dict]:
    return [
        {
            "id": f"smoke-{i}",
            "source": "smoke",
            "messages": [
                {"role": "user", "content": f"question {i}: what is {i}+{i}?"},
                {"role": "assistant", "content": f"answer {i}: {i}+{i} equals {i + i}."},
            ],
        }
        for i in range(n)
    ]


def _pairs(n: int) -> list[dict]:
    rows = []
    for i in range(n):
        prompt = f"question {i}: what is {i}+{i}?"
        rows.append(
            {
                "prompt": prompt,
                "prompt_id": f"smoke-{i}",
                "chosen": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": f"correct: {i + i}"},
                ],
                "rejected": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": "i do not know"},
                ],
                "score_chosen": 8.0,
                "score_rejected": 3.0,
            }
        )
    return rows


def _prompts(n: int) -> list[dict]:
    return [
        {"prompt": f"say hello number {i}", "prompt_id": f"smoke-{i}"} for i in range(n)
    ]


PREF_COLS = ("prompt", "chosen", "rejected")

# stage -> (env prefix, required columns, synthetic fallback)
STAGES: dict[str, tuple[str, Sequence[str], Callable[[int], list[dict]]]] = {
    "sft": ("SFT", ("messages",), _messages),
    "rm": ("RM", PREF_COLS, _pairs),
    "dpo": ("DPO", PREF_COLS, _pairs),
    "ppo": ("PPO", ("prompt",), _prompts),
}


@pytest.fixture
def require_env() -> Callable[..., tuple[str, ...]]:
    """return env values, skipping the test if any is unset."""

    def _require(*names: str) -> tuple[str, ...]:
        values = tuple(os.environ.get(n, "") for n in names)
        if not all(values):
            pytest.skip(f"set {' and '.join(names)}")
        return values

    return _require


@pytest.fixture
def assert_saved_model() -> Callable[[Path], None]:
    """assert the trainer wrote a loadable checkpoint."""

    def _assert(out: Path) -> None:
        assert out.is_dir()
        assert (out / "config.json").is_file()
        assert (
            any(out.glob("*.safetensors"))
            or (out / "model.safetensors.index.json").is_file()
        )

    return _assert


def _parse_spec(spec: str) -> tuple[str, str | None, str | None]:
    """`id`, `id:split`, or `id:config:split` -> (id, config, split)."""
    if "::" in spec:
        name, split = spec.split("::", 1)
        return name, None, split or None
    parts = spec.split(":")
    if len(parts) == 1:
        return parts[0], None, None
    if len(parts) == 2:
        return parts[0], None, parts[1] or None
    return parts[0], parts[1] or None, parts[2] or None


def _load_spec(spec: str):
    from datasets import load_dataset, load_from_disk

    p = Path(spec)
    if p.is_dir():
        return load_from_disk(str(p))
    if p.exists():
        return load_dataset("json", data_files=str(p), split="train")
    name, config, split = _parse_spec(spec)
    return load_dataset(name, config, split=split or "train")


@pytest.fixture
def smoke_cfg() -> Callable[..., dict]:
    """load `configs/<stage>.yaml` with smoke overrides applied."""

    def _cfg(stage: str, **extra: Any) -> dict:
        with (REPO_ROOT / "configs" / f"{stage}.yaml").open() as f:
            cfg = yaml.safe_load(f)
        cfg.update(BASE_OVERRIDES)
        cfg.update(extra)
        return cfg

    return _cfg


@pytest.fixture
def smoke_dataset() -> Callable[[str], Any]:
    """real dataset from `<STAGE>_SMOKE_DATASET`, else synthetic rows.

    required columns are checked here so pointing a smoke at the wrong dataset
    fails readably, rather than as a shape error deep inside the trainer.
    """

    def _load(stage: str):
        from datasets import Dataset

        prefix, required, fallback = STAGES[stage]
        n = int(os.environ.get(f"{prefix}_SMOKE_N", DEFAULT_N))
        spec = os.environ.get(f"{prefix}_SMOKE_DATASET")
        if not spec:
            return Dataset.from_list(fallback(n))

        ds = _load_spec(spec)
        if hasattr(ds, "keys"):  # DatasetDict -> first split
            ds = ds[next(iter(ds.keys()))]
        missing = [c for c in required if c not in ds.column_names]
        if missing:
            raise ValueError(
                f"{prefix}_SMOKE_DATASET={spec!r} missing column(s) {missing}; "
                f"dataset has {sorted(ds.column_names)}"
            )
        if len(ds) > n:
            ds = ds.select(range(n))
        print(f"[smoke] {stage}: {spec} -> {len(ds)} rows")
        return ds

    return _load
