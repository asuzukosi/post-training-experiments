"""unit tests for wandb_run context manager (no network)."""
from __future__ import annotations

from pathlib import Path

import pytest

from resume import cfg_use_wandb, trainer_report_to, wandb_run


def test_cfg_use_wandb_and_report_to() -> None:
    assert cfg_use_wandb({}) is True
    assert cfg_use_wandb({"use_wandb": True}) is True
    assert cfg_use_wandb({"use_wandb": False}) is False
    assert trainer_report_to(True) == "wandb"
    assert trainer_report_to(False) == "none"


def test_wandb_run_noop_when_disabled(tmp_path: Path) -> None:
    with wandb_run(
        use_wandb=False,
        output_dir=tmp_path,
        project="p",
        name="n",
    ) as run:
        assert run is None


def test_wandb_run_inits_and_finishes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    config_updates: list[dict] = []

    class _fake_wandb:
        @staticmethod
        def init(**kwargs):
            calls.append(f"init:{kwargs.get('name')}")
            return {"id": kwargs.get("id")}

        class config:
            @staticmethod
            def update(payload, allow_val_change=False):
                config_updates.append(dict(payload))

        @staticmethod
        def finish():
            calls.append("finish")

        class util:
            @staticmethod
            def generate_id():
                return "testid"

    import sys

    monkeypatch.setitem(sys.modules, "wandb", _fake_wandb)
    with wandb_run(
        use_wandb=True,
        output_dir=tmp_path,
        project="proj",
        name="run-a",
        config={"beta": 0.1},
    ) as run:
        assert run is not None
        assert calls == ["init:run-a"]
    assert calls == ["init:run-a", "finish"]
    assert config_updates == [{"beta": 0.1}]


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
