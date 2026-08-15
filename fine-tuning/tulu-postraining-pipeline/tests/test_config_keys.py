"""every key the trainers require must exist in the config they read.

this is the check that a gpu-gated smoke cannot give you locally: rewriting a config
renamed `base_model` to `model`, nothing read it, and the failure only appeared on a
rented box minutes into a run.
"""
from __future__ import annotations

import pathlib
import re

import pytest
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]

# stage -> the config it loads
STAGE_CONFIG = {
    "sft": "sft.yaml",
    "rm": "rm.yaml",
    "dpo": "dpo.yaml",
    "ppo": "ppo.yaml",
    "rs_sft": "rs_sft.yaml",
}


def _required_keys(module: pathlib.Path) -> set[str]:
    """keys read as cfg["x"] — a missing one is a KeyError at run time."""
    return set(re.findall(r'cfg\[["\'](\w+)["\']\]', module.read_text()))


@pytest.mark.parametrize(
    ("stage", "modules"),
    [
        ("sft", ["src/trainers/sft.py"]),
        ("rm", ["src/trainers/rm.py"]),
        ("dpo", ["src/trainers/dpo.py"]),
        ("ppo", ["src/trainers/ppo/trainer.py", "src/trainers/ppo/models.py"]),
    ],
)
def test_config_has_every_key_its_trainer_requires(stage: str, modules: list[str]) -> None:
    cfg = yaml.safe_load((ROOT / "configs" / STAGE_CONFIG[stage]).read_text())
    required: set[str] = set()
    for m in modules:
        required |= _required_keys(ROOT / m)

    # keys supplied at call time rather than from the file
    supplied_elsewhere = {
        "betas",          # one beta per invocation, passed in
        "max_steps",      # smoke override
        "total_episodes", # ppo sets it, but smokes override
        "missing_eos_penalty",
        "tripwire_baseline_mmlu",
        "num_pairs", "num_samples", "num_prompts",  # prep-time keys
    }
    missing = sorted(required - set(cfg) - supplied_elsewhere)
    assert not missing, f"{STAGE_CONFIG[stage]} is missing {missing}"
