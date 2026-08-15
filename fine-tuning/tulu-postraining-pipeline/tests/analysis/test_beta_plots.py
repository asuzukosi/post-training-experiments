"""T2 — O2's preference-displacement detector and the beta-arm loaders.

displacement is the specific claim that DPO pushed BOTH chosen and rejected log-probs
down — i.e. the policy moved away from the whole preference pair rather than learning to
prefer one side. the detector must not fire on the ordinary case (rejected falls, chosen
holds), because a false positive here reports a training pathology that did not happen.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from analysis.beta_plots import (
    BetaArmPoint,
    DisplacementSeries,
    beta_arms_from_json,
    detect_displacement,
    displacement_series_from_json,
    plot_beta_vs_kl_winrate,
    plot_displacement,
    plot_displacement_arms,
)


def _series(chosen: list[float], rejected: list[float], beta: float = 0.05):
    return DisplacementSeries(
        beta=beta,
        steps=list(range(len(chosen))),
        chosen_logps=chosen,
        rejected_logps=rejected,
    )


def test_displacement_fires_only_when_both_logps_fall() -> None:
    out = detect_displacement(_series(chosen=[-1.0, -2.0], rejected=[-1.5, -3.0]))
    assert out["chosen_fell"] and out["rejected_fell"]
    assert out["displacement"] is True
    assert out["chosen_delta"] == pytest.approx(-1.0)
    assert out["rejected_delta"] == pytest.approx(-1.5)


def test_ordinary_dpo_is_not_displacement() -> None:
    """rejected falls, chosen rises — this is DPO working, not a pathology.

    the most damaging false positive: it would report preference displacement on a
    perfectly healthy run and put a training failure in the write-up that never happened.
    """
    out = detect_displacement(_series(chosen=[-2.0, -1.0], rejected=[-1.5, -3.0]))
    assert out["rejected_fell"] is True
    assert out["chosen_fell"] is False
    assert out["displacement"] is False


def test_chosen_falling_alone_is_not_displacement() -> None:
    out = detect_displacement(_series(chosen=[-1.0, -2.0], rejected=[-3.0, -1.5]))
    assert out["chosen_fell"] is True
    assert out["rejected_fell"] is False
    assert out["displacement"] is False


def test_flat_series_does_not_fire() -> None:
    """a delta of exactly 0 is not a fall; `< 0` must be strict."""
    out = detect_displacement(_series(chosen=[-1.0, -1.0], rejected=[-1.5, -1.5]))
    assert out["displacement"] is False
    assert out["chosen_delta"] == 0.0


def test_detector_is_end_vs_start_not_monotonic() -> None:
    """the docstring says end-vs-start; a dip that recovers must NOT count as a fall."""
    out = detect_displacement(
        _series(chosen=[-1.0, -5.0, -0.5], rejected=[-1.0, -5.0, -0.5])
    )
    assert out["chosen_delta"] == pytest.approx(0.5)
    assert out["displacement"] is False


def test_single_point_series_cannot_claim_displacement() -> None:
    """one logged step has no trend; reporting False beats inventing one."""
    out = detect_displacement(_series(chosen=[-1.0], rejected=[-1.5]))
    assert out["displacement"] is False
    assert out["chosen_delta"] == 0.0


def test_series_rejects_ragged_input() -> None:
    """mismatched lengths would silently pair the wrong steps together."""
    with pytest.raises(ValueError, match="length mismatch"):
        DisplacementSeries(
            beta=0.05, steps=[0, 1, 2], chosen_logps=[-1.0, -2.0], rejected_logps=[-1.0]
        )


def test_loaders_round_trip_and_reject_malformed(tmp_path: Path) -> None:
    arms_p = tmp_path / "arms.json"
    arms_p.write_text(
        json.dumps([{"beta": 0.05, "kl": 12.0, "win_rate_vs_sft": 0.6}]), encoding="utf-8"
    )
    arms = beta_arms_from_json(arms_p)
    assert arms[0].beta == pytest.approx(0.05) and arms[0].kl == pytest.approx(12.0)

    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps([{"kl": 1.0}]), encoding="utf-8")  # no beta
    with pytest.raises(ValueError, match="missing beta"):
        beta_arms_from_json(bad)

    ser_p = tmp_path / "series.json"
    ser_p.write_text(
        json.dumps(
            [{"beta": 0.1, "steps": [0, 1], "chosen_logps": [-1, -2],
              "rejected_logps": [-1, -3]}]
        ),
        encoding="utf-8",
    )
    loaded = displacement_series_from_json(ser_p)
    assert detect_displacement(loaded[0])["displacement"] is True


def test_plots_write_files_without_raising(tmp_path: Path) -> None:
    """smoke only — these are matplotlib wrappers, but a crash here loses O2's figures."""
    import matplotlib

    matplotlib.use("Agg")

    a = plot_beta_vs_kl_winrate(
        [BetaArmPoint(beta=0.05, kl=10.0, win_rate_vs_sft=0.6),
         BetaArmPoint(beta=0.1, kl=4.0, win_rate_vs_sft=0.55)],
        tmp_path / "beta.png",
    )
    b = plot_displacement(_series([-1.0, -2.0], [-1.5, -3.0]), tmp_path / "disp.png")
    c = plot_displacement_arms(
        [_series([-1.0, -2.0], [-1.5, -3.0], beta=0.05),
         _series([-1.0, -1.2], [-1.5, -2.0], beta=0.1)],
        tmp_path / "arms.png",
    )
    for p in (a, b, c):
        assert Path(p).is_file() and Path(p).stat().st_size > 0
