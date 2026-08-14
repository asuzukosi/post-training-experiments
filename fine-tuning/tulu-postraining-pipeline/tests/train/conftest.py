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

import json
import os
from pathlib import Path
from typing import Any, Callable, Sequence

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_N = 8

# overrides every smoke wants: two steps, no accumulation, fp32, no saving mid-run.
#
# deliberately does NOT override max_length or any other data-shape setting — the
# smoke uses each stage's production value from configs/<stage>.yaml. a 256-token
# cap made the smoke pass while training NOTHING: real tulu rows are median 554
# tokens and 11 of 32 have their assistant span starting past token 256, so
# truncation removed every assistant token, the mask was all-zero, and loss/grad
# were 0.0 while the test still asserted a checkpoint existed. keep the smoke short
# via max_steps, never by reshaping the data.
BASE_OVERRIDES: dict[str, Any] = {
    "max_steps": 2,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 1,
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


def _values(history: list[dict], *keys: str) -> list[float]:
    """values for the first of `keys` that appears anywhere in the log history."""
    for key in keys:
        found = [e[key] for e in history if key in e]
        if found:
            return found
    return []


def _assert_supervised_trained(history: list[dict], state: Path) -> None:
    losses = _values(history, "loss")
    assert losses, f"no loss logged in {state}"
    assert any(l > 0 for l in losses), (
        f"every logged loss is 0.0 ({losses}) — nothing was learned. usually means the "
        "label mask is empty: truncation removed the assistant span, or assistant_only_loss "
        "is on with a template that yields no assistant tokens."
    )


def _assert_ppo_trained(history: list[dict], state: Path) -> None:
    """ppo's null-run signal is not a zero loss — see `assert_trained`."""
    rewards = _values(history, "objective/rlhf_reward", "objective/scores")
    assert rewards, (
        f"no objective/rlhf_reward or objective/scores logged in {state} — the reward "
        "model never scored a completion, so there was no learning signal at all."
    )
    entropy = _values(history, "policy/entropy_avg", "objective/entropy")
    assert not entropy or any(e != 0 for e in entropy), (
        f"policy entropy is 0.0 at every step ({entropy}) — the policy emitted nothing to "
        "learn from. usually means response_length collapsed or generation was fully masked."
    )


@pytest.fixture
def assert_trained() -> Callable[[Path], None]:
    """assert the run actually learned something, not just wrote a checkpoint.

    `assert_saved_model` cannot tell a trained checkpoint from a null one: a run whose
    labels were all masked or truncated away writes a byte-identical-looking directory
    while loss and grad_norm stay 0.0. that exact failure passed a smoke earlier — real
    tulu rows are median 554 tokens, so a 256-token cap removed every assistant token.

    the check is per-shape, because the two trainer families fail differently:

    - supervised (sft/rm/dpo) log a scalar `loss`; an all-zero loss means an empty mask.
    - ppo never logs a plain `loss` — trl logs loss/policy_avg, loss/value_avg and
      objective/*. its policy loss is -min(ratio*adv, clip(ratio)*adv), which is
      legitimately negative or ~0, so "loss > 0" would be a meaningless assertion.
      the real ppo null run is one that generated or scored nothing, so assert on the
      reward signal and on non-degenerate entropy instead.

    kept separate from `assert_saved_model` on purpose, so tests that legitimately do not
    train can still assert the artifacts alone.
    """

    def _assert(out: Path) -> None:
        state = out / "trainer_state.json"
        assert state.is_file(), f"no trainer_state.json in {out}; did run_* call save_state()?"
        history = json.loads(state.read_text(encoding="utf-8")).get("log_history", [])
        assert history, f"empty log_history in {state} — no step was ever logged"
        is_ppo = any(k.startswith("objective/") for e in history for k in e)
        (_assert_ppo_trained if is_ppo else _assert_supervised_trained)(history, state)

    return _assert


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
