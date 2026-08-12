"""beta vs kl/win-rate and preference-displacement plots."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pipeline.analysis.io import DEFAULT_PLOTS_DIR, resolve_output_path


@dataclass
class BetaArmPoint:
    """one dpo beta arm's summary metrics for the beta trade-off plot."""

    beta: float
    kl: float | None = None
    win_rate_vs_sft: float | None = None
    win_rate_vs_sft_lc: float | None = None


@dataclass
class DisplacementSeries:
    """per-step chosen/rejected log-probs for one beta arm (displacement signal)."""

    beta: float
    steps: list[int]
    chosen_logps: list[float]
    rejected_logps: list[float]

    def __post_init__(self) -> None:
        n = len(self.steps)
        if len(self.chosen_logps) != n or len(self.rejected_logps) != n:
            raise ValueError(
                "displacement series length mismatch: "
                f"steps={n} chosen={len(self.chosen_logps)} "
                f"rejected={len(self.rejected_logps)}"
            )


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _nan_or(values: Sequence[float | None]) -> list[float]:
    return [float("nan") if v is None else float(v) for v in values]


def _save_fig(fig: Any, out: Path, *, msg: str) -> Path:
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    _pyplot().close(fig)
    print(msg)
    return out


def plot_beta_vs_kl_winrate(
    arms: Sequence[BetaArmPoint],
    path: str | Path | None = None,
    *,
    title: str = "dpo beta vs kl / win-rate",
) -> Path:
    """plot beta on x; kl and win-rate vs sft on twin y-axes."""
    if not arms:
        raise ValueError("arms must be non-empty")

    plt = _pyplot()
    out = resolve_output_path(path, default=DEFAULT_PLOTS_DIR / "beta_vs_kl_winrate.png")

    betas = [a.beta for a in arms]
    fig, ax_kl = plt.subplots(figsize=(7, 4.5))
    ax_win = ax_kl.twinx()

    if any(a.kl is not None for a in arms):
        ax_kl.plot(betas, _nan_or([a.kl for a in arms]), marker="o", color="C0", label="kl")
    if any(a.win_rate_vs_sft is not None for a in arms):
        ax_win.plot(
            betas,
            _nan_or([a.win_rate_vs_sft for a in arms]),
            marker="s",
            color="C1",
            label="win-rate vs sft (raw)",
        )
    if any(a.win_rate_vs_sft_lc is not None for a in arms):
        ax_win.plot(
            betas,
            _nan_or([a.win_rate_vs_sft_lc for a in arms]),
            marker="^",
            color="C2",
            linestyle="--",
            label="win-rate vs sft (length-controlled)",
        )

    ax_kl.set_xlabel("beta")
    ax_kl.set_ylabel("kl")
    ax_win.set_ylabel("win-rate vs sft")
    ax_kl.set_title(title)
    h1, l1 = ax_kl.get_legend_handles_labels()
    h2, l2 = ax_win.get_legend_handles_labels()
    ax_kl.legend(h1 + h2, l1 + l2, loc="best")
    return _save_fig(fig, out, msg=f"wrote beta vs kl/win-rate plot -> {out}")


def plot_displacement(
    series: DisplacementSeries,
    path: str | Path | None = None,
    *,
    title: str | None = None,
) -> Path:
    """plot chosen + rejected log-probs over steps (both falling => displacement)."""
    plt = _pyplot()
    beta_tag = f"{series.beta:g}"
    out = resolve_output_path(
        path,
        default=DEFAULT_PLOTS_DIR / f"displacement_b{beta_tag}.png",
    )

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(series.steps, series.chosen_logps, label="logps/chosen", color="C0")
    ax.plot(series.steps, series.rejected_logps, label="logps/rejected", color="C1")
    ax.set_xlabel("step")
    ax.set_ylabel("log-prob")
    ax.set_title(title or f"preference displacement (beta={beta_tag})")
    ax.legend(loc="best")
    return _save_fig(fig, out, msg=f"wrote displacement plot -> {out}")


def plot_displacement_arms(
    series_list: Sequence[DisplacementSeries],
    path: str | Path | None = None,
    *,
    title: str = "preference displacement by beta",
) -> Path:
    """overlay chosen/rejected log-prob curves for multiple beta arms."""
    if not series_list:
        raise ValueError("series_list must be non-empty")

    plt = _pyplot()
    out = resolve_output_path(
        path,
        default=DEFAULT_PLOTS_DIR / "displacement_by_beta.png",
    )

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, series in enumerate(series_list):
        tag = f"b{series.beta:g}"
        color = f"C{i}"
        ax.plot(series.steps, series.chosen_logps, color=color, label=f"{tag} chosen")
        ax.plot(
            series.steps,
            series.rejected_logps,
            color=color,
            linestyle="--",
            label=f"{tag} rejected",
        )
    ax.set_xlabel("step")
    ax.set_ylabel("log-prob")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    return _save_fig(fig, out, msg=f"wrote multi-beta displacement plot -> {out}")


def detect_displacement(series: DisplacementSeries) -> dict[str, Any]:
    """end-vs-start check: both chosen and rejected logps fell => displacement."""
    if len(series.steps) < 2:
        return {
            "beta": series.beta,
            "chosen_fell": False,
            "rejected_fell": False,
            "displacement": False,
            "chosen_delta": 0.0,
            "rejected_delta": 0.0,
        }

    chosen_delta = series.chosen_logps[-1] - series.chosen_logps[0]
    rejected_delta = series.rejected_logps[-1] - series.rejected_logps[0]
    chosen_fell = chosen_delta < 0
    rejected_fell = rejected_delta < 0
    return {
        "beta": series.beta,
        "chosen_fell": chosen_fell,
        "rejected_fell": rejected_fell,
        "displacement": chosen_fell and rejected_fell,
        "chosen_delta": chosen_delta,
        "rejected_delta": rejected_delta,
    }
