"""unit tests for wandb_run context manager (no network)."""
from __future__ import annotations

from pathlib import Path

import pytest

from resume import cfg_use_wandb, trainer_report_to, wandb_run


def test_wandb_run_finishes_on_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []

    class _fake_wandb:
        @staticmethod
        def init(**_kwargs):
            calls.append("init")
            return object()

        @staticmethod
        def finish():
            calls.append("finish")

        class util:
            @staticmethod
            def generate_id():
                return "testid"

    import sys

    monkeypatch.setitem(sys.modules, "wandb", _fake_wandb)
    with pytest.raises(RuntimeError, match="boom"):
        with wandb_run(
            use_wandb=True,
            output_dir=tmp_path,
            project="proj",
            name="run-b",
        ):
            raise RuntimeError("boom")
    assert calls == ["init", "finish"]


def test_smoke_overrides_disable_wandb(tmp_path, smoke_cfg) -> None:
    """a smoke must never phone home to w&b.

    trainers read `use_wandb` via cfg_use_wandb and derive hf report_to from it; the
    smoke overrides set it false.
    """
    from trainers.rm import build_rm_config

    args = build_rm_config(
        smoke_cfg("rm"), run_name="wandb_off", output_dir=tmp_path, push_to_hub=False
    )
    assert args.report_to in ([], "none", ["none"])
