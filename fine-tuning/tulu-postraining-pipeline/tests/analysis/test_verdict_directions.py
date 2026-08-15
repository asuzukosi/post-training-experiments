"""T1 — the verdict maths that PRODUCES the headline numbers (O1, O4, O5).

these are pure computation over jsonl with no external library to crash, so a wrong
one emits plausible numbers and nothing downstream catches it. the assertions here are
deliberately about DIRECTION and SIGN: a flipped comparison is the likeliest bug and the
only one that would survive review, because every other symptom (crash, nan, empty
output) is loud.
"""
from __future__ import annotations

import math

import pytest

from analysis import (
    AttributionRow,
    compare_metric,
    compare_rs_against_dpo,
    compute_stage_deltas,
    summarize_runs,
    winner_against_chance,
)
from analysis.verdict.stats import t_crit_95


# --------------------------------------------------------------------------- O1


def test_stage_deltas_are_current_minus_previous() -> None:
    """sft_vs_base must be sft - base, not base - sft.

    the whole attribution table reads as "what did this stage ADD", so a flipped
    subtraction reports every improvement as a regression and vice versa.
    """
    rows = [
        AttributionRow(stage="base", format_ifeval=0.30, skills_mmlu=0.50),
        AttributionRow(stage="sft", format_ifeval=0.45, skills_mmlu=0.48),
    ]
    d = compute_stage_deltas(rows)["sft_vs_base"]
    assert d["format_ifeval"] == pytest.approx(0.15)   # sft improved format
    assert d["skills_mmlu"] == pytest.approx(-0.02)    # and cost a little mmlu


def test_preference_stage_deltas_are_measured_against_sft_not_base() -> None:
    """dpo/ppo deltas are vs SFT — measuring them vs base would double-count SFT's gain."""
    rows = [
        AttributionRow(stage="base", format_ifeval=0.30, skills_mmlu=0.50),
        AttributionRow(stage="sft", format_ifeval=0.45, skills_mmlu=0.48),
        AttributionRow(
            stage="ppo",
            format_ifeval=0.55,
            skills_mmlu=0.44,
            style_mean_chars=1200.0,
            style_markdown_rate=0.8,
        ),
    ]
    out = compute_stage_deltas(rows)
    d = out["ppo_vs_sft"]
    assert d["format_ifeval"] == pytest.approx(0.10)   # 0.55 - 0.45, NOT 0.55 - 0.30
    assert d["skills_mmlu"] == pytest.approx(-0.04)    # 0.44 - 0.48


def test_stage_deltas_tolerate_missing_stages_and_missing_metrics() -> None:
    """a stage that has not run yet must be skipped, not crash or report 0.0.

    reporting 0.0 for a missing metric would read as "this stage changed nothing",
    which is a different and wrong claim.
    """
    assert compute_stage_deltas([]) == {}
    # base present, sft absent -> nothing is comparable
    assert compute_stage_deltas([AttributionRow(stage="base", format_ifeval=0.3)]) == {}
    # sft present but the metric is None on one side -> None, not 0.0
    rows = [
        AttributionRow(stage="base", format_ifeval=None, skills_mmlu=0.50),
        AttributionRow(stage="sft", format_ifeval=0.45, skills_mmlu=0.48),
    ]
    d = compute_stage_deltas(rows)["sft_vs_base"]
    assert d["format_ifeval"] is None
    assert d["skills_mmlu"] == pytest.approx(-0.02)


# --------------------------------------------------------------------------- O4


def test_compare_metric_delta_is_ppo_minus_dpo() -> None:
    """the field is named delta_ppo_minus_dpo; the sign must match the name."""
    v = compare_metric(
        metric="win_rate",
        dpo_values=[0.50, 0.51, 0.49],
        ppo_values=[0.60, 0.61, 0.59],
    )
    assert v.delta_ppo_minus_dpo == pytest.approx(0.10)
    assert v.winner == "ppo"


def test_compare_metric_declares_dpo_when_dpo_is_ahead() -> None:
    v = compare_metric(
        metric="win_rate",
        dpo_values=[0.70, 0.71, 0.69],
        ppo_values=[0.55, 0.56, 0.54],
    )
    assert v.delta_ppo_minus_dpo == pytest.approx(-0.15)
    assert v.winner == "dpo"


def test_compare_metric_calls_a_tie_when_the_ci_includes_zero() -> None:
    """overlapping noisy runs must not produce a winner.

    this is the guard against reporting a DPO-vs-PPO verdict that is really run-to-run
    variance — the exact failure the spec asks for CIs to prevent.
    """
    v = compare_metric(
        metric="win_rate",
        dpo_values=[0.50, 0.60, 0.40],
        ppo_values=[0.52, 0.61, 0.39],
    )
    assert v.winner == "tie"
    assert v.delta_ci95_low < 0 < v.delta_ci95_high


def test_compare_metric_single_run_cannot_declare_a_winner() -> None:
    """n=1 has no spread, so the ci is a point and any difference looks significant.

    with n_eff < 2 the half-width is forced to 0, which would make lo == hi == delta and
    declare a winner off one run each. assert that does not happen.
    """
    v = compare_metric(metric="win_rate", dpo_values=[0.50], ppo_values=[0.90])
    assert v.delta_ppo_minus_dpo == pytest.approx(0.40)  # the delta is still reported
    assert v.winner == "tie", "a single run per arm must never decide dpo-vs-ppo"
    assert "no spread estimate" in v.note
    # the ci must be unbounded, not a zero-width point around the delta
    assert math.isinf(v.delta_ci95_low) and math.isinf(v.delta_ci95_high)


# --------------------------------------------------------------------------- O5


def test_winner_against_chance_uses_the_ci_not_the_mean() -> None:
    """rs wins only if the whole ci clears 0.5; a mean above 0.5 is not enough."""
    ahead = summarize_runs([0.70, 0.72, 0.71])
    assert winner_against_chance(ahead)[0] == "rs_sft"

    behind = summarize_runs([0.30, 0.28, 0.29])
    assert winner_against_chance(behind)[0] == "dpo"

    # mean 0.55 but wide spread -> ci straddles 0.5 -> tie, not "rs_sft"
    noisy = summarize_runs([0.20, 0.90, 0.55])
    assert noisy.mean > 0.5
    assert winner_against_chance(noisy)[0] == "tie"


def test_compare_rs_against_dpo_is_a_win_rate_against_0_5() -> None:
    """rs_values are head-to-head win rates for RS, so chance is 0.5 by construction."""
    m = compare_rs_against_dpo(metric="raw", rs_values=[0.65, 0.66, 0.64])
    assert m.winner == "rs_sft"
    assert m.rs.mean == pytest.approx(0.65)


# ------------------------------------------------------------------------- stats


def test_t_crit_95_is_infinite_below_two_runs_and_shrinks_with_n() -> None:
    """returning a finite t for n<2 would silently license a verdict off one run."""
    assert math.isinf(t_crit_95(0))
    assert math.isinf(t_crit_95(1))
    assert t_crit_95(2) > t_crit_95(5) > t_crit_95(10) >= 1.96
