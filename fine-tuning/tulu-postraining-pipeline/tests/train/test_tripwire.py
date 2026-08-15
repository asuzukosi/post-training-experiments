"""the mid-training mmlu tripwire."""
from __future__ import annotations

import sys
import types

import pytest

from trainers.tripwire import MMLUTripwire, eval_steps


class _State:
    def __init__(self, max_steps: int, step: int = 0) -> None:
        self.max_steps = max_steps
        self.global_step = step


class _Model:
    training = True

    def eval(self) -> None:
        self.training = False

    def train(self) -> None:
        self.training = True


@pytest.fixture
def fake_lm_eval(monkeypatch: pytest.MonkeyPatch):
    """stub lm-eval; return a handle that sets the accuracy each call returns."""
    scores: list[float] = []

    def simple_evaluate(**_kw):
        return {"results": {"mmlu": {"acc,none": scores.pop(0)}}}

    root = types.ModuleType("lm_eval")
    root.simple_evaluate = simple_evaluate
    hf = types.ModuleType("lm_eval.models.huggingface")
    hf.HFLM = lambda **kw: object()
    models = types.ModuleType("lm_eval.models")
    monkeypatch.setitem(sys.modules, "lm_eval", root)
    monkeypatch.setitem(sys.modules, "lm_eval.models", models)
    monkeypatch.setitem(sys.modules, "lm_eval.models.huggingface", hf)
    return scores


def test_four_checks_evenly_spaced_whatever_the_run_length() -> None:
    """a fixed fraction, not a fixed step count — sft and ppo differ by ~30x."""
    assert eval_steps(1500) == [300, 600, 900, 1200]
    assert eval_steps(46) == [9, 18, 27, 36]
    # the end of the run is excluded; that is the final evaluation's job
    assert all(s < 1500 for s in eval_steps(1500))


def test_short_runs_get_no_checks_rather_than_a_crash() -> None:
    assert eval_steps(1) == []
    assert eval_steps(0) == []


def test_first_reading_becomes_the_baseline(fake_lm_eval) -> None:
    fake_lm_eval.extend([0.60])
    tw = MMLUTripwire(tokenizer=object())
    tw.on_train_begin(None, _State(50), None)
    tw.on_step_end(None, _State(50, step=10), None, model=_Model())
    assert tw.baseline == pytest.approx(0.60)
    assert len(tw.readings) == 1
    assert tw.readings[0]["step"] == 10 and tw.readings[0]["mmlu_diff"] is None


def test_a_small_drop_is_allowed_through(fake_lm_eval) -> None:
    fake_lm_eval.extend([0.60, 0.58])  # 2 points, under the limit
    tw = MMLUTripwire(tokenizer=object())
    tw.on_train_begin(None, _State(50), None)
    tw.on_step_end(None, _State(50, step=10), None, model=_Model())
    tw.on_step_end(None, _State(50, step=20), None, model=_Model())
    assert len(tw.readings) == 2


def test_readings_are_flushed_to_disk_as_they_happen(fake_lm_eval, tmp_path) -> None:
    """an aborted run must still leave the evidence of why it stopped."""
    import json

    fake_lm_eval.extend([0.60, 0.40])
    tw = MMLUTripwire(tokenizer=object(), output_dir=tmp_path)
    tw.on_train_begin(None, _State(50), None)
    tw.on_step_end(None, _State(50, step=10), None, model=_Model())
    with pytest.raises(RuntimeError):
        tw.on_step_end(None, _State(50, step=20), None, model=_Model())

    lines = (tmp_path / "tripwire_mmlu.jsonl").read_text().strip().splitlines()
    rows = [json.loads(line) for line in lines]
    assert [r["step"] for r in rows] == [10, 20]
    assert rows[1]["mmlu_diff"] == pytest.approx(-20.0)
    # each line stands alone: baseline and threshold travel with the reading
    assert rows[1]["baseline_mmlu"] == pytest.approx(0.60)
    assert rows[1]["backend"] == "hf"


def test_a_collapse_aborts_the_run(fake_lm_eval) -> None:
    """it stops rather than warns — a warning is a slower way to find out at the end."""
    fake_lm_eval.extend([0.60, 0.40])  # 20 points
    tw = MMLUTripwire(tokenizer=object())
    tw.on_train_begin(None, _State(50), None)
    tw.on_step_end(None, _State(50, step=10), None, model=_Model())
    with pytest.raises(RuntimeError, match="tripwire"):
        tw.on_step_end(None, _State(50, step=20), None, model=_Model())


def test_a_supplied_baseline_catches_a_drop_on_the_first_check(fake_lm_eval) -> None:
    """with the pre-training score passed in, check one is already a comparison."""
    fake_lm_eval.extend([0.40])
    tw = MMLUTripwire(tokenizer=object(), baseline=0.60)
    tw.on_train_begin(None, _State(50), None)
    with pytest.raises(RuntimeError, match=r"-20\.00 points"):
        tw.on_step_end(None, _State(50, step=10), None, model=_Model())


def test_it_does_not_fire_on_other_steps(fake_lm_eval) -> None:
    tw = MMLUTripwire(tokenizer=object())
    tw.on_train_begin(None, _State(50), None)
    tw.on_step_end(None, _State(50, step=7), None, model=_Model())
    assert tw.readings == []  # no lm-eval call; the stub would have raised IndexError


def test_training_mode_is_restored_after_measuring(fake_lm_eval) -> None:
    """leaving the model in eval mode would silently disable dropout for the rest."""
    fake_lm_eval.extend([0.60])
    model = _Model()
    tw = MMLUTripwire(tokenizer=object())
    tw.on_train_begin(None, _State(50), None)
    tw.on_step_end(None, _State(50, step=10), None, model=model)
    assert model.training is True


def test_smoke_overrides_disable_the_tripwire(smoke_cfg) -> None:
    """a 2-step smoke must not stop to run 285 mmlu questions.

    the production configs enable the tripwire, and smokes inherit them, so this is the
    guard that a smoke stays a smoke.
    """
    for stage in ("sft", "dpo", "ppo"):
        assert smoke_cfg(stage).get("tripwire_evals") == 0, stage


def test_attach_is_opt_in(monkeypatch) -> None:
    """never attach unless the config asks; a run should not pay for it by accident."""
    from trainers.tripwire import attach_mmlu_tripwire

    added: list = []
    trainer = type("T", (), {"add_callback": lambda self, cb: added.append(cb)})()

    attach_mmlu_tripwire(trainer, {}, tokenizer=object())
    assert added == []

    attach_mmlu_tripwire(trainer, {"tripwire_evals": 0}, tokenizer=object())
    assert added == []

    attach_mmlu_tripwire(trainer, {"tripwire_evals": 2, "tripwire_questions": 3}, tokenizer=object())
    assert len(added) == 1 and added[0].evals == 2 and added[0].limit == 3


def test_it_survives_every_event_the_real_handler_fires() -> None:
    """the handler dispatches with getattr(callback, event) and fires fifteen events.

    a duck-typed object with only the two methods we need raises AttributeError on the
    first one it lacks — before training starts. this drives the real CallbackHandler
    rather than calling our methods directly, which is what a unit test misses.
    """
    from transformers import TrainerControl, TrainerState
    from transformers.trainer_callback import CallbackHandler

    tw = MMLUTripwire(tokenizer=object())
    handler = CallbackHandler([tw], model=None, processing_class=None, optimizer=None, lr_scheduler=None)
    state, control = TrainerState(), TrainerControl()
    state.max_steps, state.global_step = 50, 0

    # the real Trainer supplies these for the events that take them
    extras = {"metrics": {}, "logs": {}}
    for event in (e for e in dir(tw) if e.startswith("on_")):
        handler.call_event(event, args=None, state=state, control=control, **extras)


def test_wandb_is_optional_and_never_starts_a_run(monkeypatch, fake_lm_eval) -> None:
    """log to the trainer's run if one is open; do not create one."""
    import sys
    import types

    logged: list[dict] = []
    fake = types.ModuleType("wandb")
    fake.run = None  # no run open
    fake.log = lambda payload, step=None: logged.append(payload)
    monkeypatch.setitem(sys.modules, "wandb", fake)

    fake_lm_eval.extend([0.60, 0.58])
    tw = MMLUTripwire(tokenizer=object())
    tw.on_train_begin(None, _State(50), None)
    tw.on_step_end(None, _State(50, step=10), None, model=_Model())
    assert logged == []  # nothing logged while no run is open

    fake.run = object()  # the trainer opened one
    tw.on_step_end(None, _State(50, step=20), None, model=_Model())
    assert logged and "tripwire/mmlu" in logged[0]
