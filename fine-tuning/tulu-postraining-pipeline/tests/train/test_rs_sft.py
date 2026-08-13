"""rs-sft config + cli wiring (reuses trainers.sft; no gpu)."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import yaml

from trainers.sft import build_sft_config, resolve_sft_checkpoint

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_rs_sft_config_is_separate_from_sft() -> None:
    with (REPO_ROOT / "configs" / "rs_sft.yaml").open() as f:
        rs = yaml.safe_load(f)
    with (REPO_ROOT / "configs" / "sft.yaml").open() as f:
        sft = yaml.safe_load(f)
    assert rs["task"] == "rs_sft"
    assert rs["processed_path"] == "data/processed/rs_sft"
    assert rs["assistant_only_loss"] is True
    assert rs.get("sft_checkpoint") is None
    assert "base_model" not in rs
    assert rs["processed_path"] != sft["processed_path"]
    assert rs["task"] != sft["task"]


def test_rs_sft_config_builds_sft_args(tmp_path: Path) -> None:
    with (REPO_ROOT / "configs" / "rs_sft.yaml").open() as f:
        cfg = yaml.safe_load(f)
    args = build_sft_config(
        cfg,
        run_name="rs_sft_cfg",
        output_dir=tmp_path / "out",
        push_to_hub=False,
    )
    assert args.assistant_only_loss is True
    assert args.packing is True
    assert args.push_to_hub is False


def test_rs_sft_script_defaults_to_rs_config() -> None:
    path = REPO_ROOT / "scripts" / "train" / "rs_sft.py"
    spec = importlib.util.spec_from_file_location("train_rs_sft", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.DEFAULT_CONFIG == REPO_ROOT / "configs" / "rs_sft.yaml"
    ns = mod.parse_args([])
    assert ns.config == mod.DEFAULT_CONFIG
    assert ns.sft_checkpoint is None


def test_resolve_sft_checkpoint_required(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="sft_checkpoint is required"):
        resolve_sft_checkpoint({"sft_checkpoint": None}, None)
    ckpt = tmp_path / "sft_run"
    ckpt.mkdir()
    assert resolve_sft_checkpoint({}, str(ckpt)) == ckpt
    assert resolve_sft_checkpoint({"sft_checkpoint": str(ckpt)}, None) == ckpt
