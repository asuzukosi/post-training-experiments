"""sft smoke on a tiny conversational subset."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from trainers.sft import build_sft_config, run_sft

SFT_OVERRIDES = {"num_train_epochs": 1, "packing": False, "assistant_only_loss": True}


def _cfg(smoke_cfg) -> dict:
    cfg = smoke_cfg("sft", **SFT_OVERRIDES)
    cfg["base_model"] = os.environ.get("SFT_SMOKE_MODEL", cfg["base_model"])
    return cfg



@pytest.mark.skipif(
    os.environ.get("RUN_SFT_SMOKE") != "1",
    reason="set RUN_SFT_SMOKE=1 to run the gpu/model smoke",
)
def test_sft_smoke_train_tiny_subset(
    tmp_path: Path, smoke_cfg, smoke_dataset, assert_saved_model, assert_trained
) -> None:
    out = run_sft(
        _cfg(smoke_cfg),
        dataset=smoke_dataset("sft"),
        run_name="sft_smoke",
        output_dir=tmp_path / "sft_smoke",
        push_to_hub=False,
    )
    assert_saved_model(out)
    assert_trained(out)


@pytest.mark.skipif(
    os.environ.get("RUN_TRIPWIRE_SMOKE") != "1",
    reason="set RUN_TRIPWIRE_SMOKE=1 to run the gpu tripwire smoke (needs lm-eval)",
)
def test_training_survives_the_tripwire(
    tmp_path: Path, smoke_cfg, smoke_dataset, assert_saved_model, assert_trained
) -> None:
    """the tripwire must measure and hand the run back, not break it.

    the only test that runs the callback against a real model and a real lm-eval. it
    proves three things at once: HFLM accepts an in-memory model, the measurement does
    not exhaust the card training is already using, and training continues afterwards
    and still writes a trained checkpoint.
    """
    cfg = _cfg(smoke_cfg)
    cfg.update({"max_steps": 4, "tripwire_evals": 1, "tripwire_questions": 1})

    out = run_sft(
        cfg,
        dataset=smoke_dataset("sft"),
        run_name="tripwire_smoke",
        output_dir=tmp_path / "tripwire_smoke",
        push_to_hub=False,
    )
    assert_saved_model(out)
    assert_trained(out)
