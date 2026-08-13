"""chattiness plots: raw vs length-controlled win-rate + length/markdown by stage."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from analysis.io import (
    DEFAULT_PLOTS_DIR,
    as_float,
    load_json_mapping,
    resolve_output_path,
)


@dataclass
class ChattinessPoint:
    """one stage/arm's style + judged win-rates (usually vs sft)."""

    stage: str
    mean_chars: float | None = None
    markdown_rate: float | None = None
    win_rate_raw: float | None = None
    win_rate_lc: float | None = None
    mean_char_delta_vs_ref: float | None = None

    @property
    def win_rate_drop(self) -> float | None:
        """raw - length-controlled; how much of the win shrinks under length control."""
        if self.win_rate_raw is None or self.win_rate_lc is None:
            return None
        return self.win_rate_raw - self.win_rate_lc


def chattiness_point_from_style_report(
    stage: str,
    report: str | Path | Mapping[str, Any],
) -> ChattinessPoint:
    """build a point from a head-to-head style report (stage as model b)."""
    payload = load_json_mapping(report)
    style_b = payload.get("style_b") or {}
    raw = payload.get("raw") or {}
    lc = payload.get("length_controlled") or {}
    return ChattinessPoint(
        stage=stage,
        mean_chars=as_float(style_b.get("mean_chars")),
        markdown_rate=as_float(style_b.get("markdown_rate")),
        win_rate_raw=as_float(raw.get("win_rate_b")),
        win_rate_lc=as_float(lc.get("win_rate_b")),
        mean_char_delta_vs_ref=as_float(payload.get("mean_char_delta_b_minus_a")),
    )


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _save_fig(fig: Any, out: Path, *, msg: str) -> Path:
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    _pyplot().close(fig)
    print(msg)
    return out


def _nan_or(values: Sequence[float | None]) -> list[float]:
    return [float("nan") if v is None else float(v) for v in values]


def plot_raw_vs_length_controlled(
    points: Sequence[ChattinessPoint],
    path: str | Path | None = None,
    *,
    title: str = "chattiness: raw vs length-controlled win-rate",
) -> Path:
    """grouped bars: raw win-rate vs length-controlled win-rate per stage."""
    if not points:
        raise ValueError("points must be non-empty")

    plt = _pyplot()
    out = resolve_output_path(
        path,
        default=DEFAULT_PLOTS_DIR / "chattiness_raw_vs_lc.png",
    )

    stages = [p.stage for p in points]
    raw = _nan_or([p.win_rate_raw for p in points])
    lc = _nan_or([p.win_rate_lc for p in points])
    x = list(range(len(stages)))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.bar([i - width / 2 for i in x], raw, width, label="raw", color="C0")
    ax.bar([i + width / 2 for i in x], lc, width, label="length-controlled", color="C1")
    ax.set_xticks(x)
    ax.set_xticklabels(stages, rotation=20, ha="right")
    ax.set_ylabel("win-rate vs ref")
    ax.set_ylim(0.0, 1.0)
    ax.set_title(title)
    ax.legend(loc="best")
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=1)
    return _save_fig(fig, out, msg=f"wrote chattiness raw-vs-lc plot -> {out}")


def plot_length_markdown(
    points: Sequence[ChattinessPoint],
    path: str | Path | None = None,
    *,
    title: str = "chattiness: mean length and markdown rate",
) -> Path:
    """twin-axis plot: mean chars + markdown rate per stage."""
    if not points:
        raise ValueError("points must be non-empty")

    plt = _pyplot()
    out = resolve_output_path(
        path,
        default=DEFAULT_PLOTS_DIR / "chattiness_length_markdown.png",
    )

    stages = [p.stage for p in points]
    chars = _nan_or([p.mean_chars for p in points])
    md = _nan_or([p.markdown_rate for p in points])
    x = list(range(len(stages)))

    fig, ax_len = plt.subplots(figsize=(7.5, 4.5))
    ax_md = ax_len.twinx()
    ax_len.plot(x, chars, marker="o", color="C0", label="mean chars")
    ax_md.plot(x, md, marker="s", color="C1", label="markdown rate")
    ax_len.set_xticks(x)
    ax_len.set_xticklabels(stages, rotation=20, ha="right")
    ax_len.set_ylabel("mean chars")
    ax_md.set_ylabel("markdown rate")
    ax_md.set_ylim(0.0, 1.0)
    ax_len.set_title(title)
    h1, l1 = ax_len.get_legend_handles_labels()
    h2, l2 = ax_md.get_legend_handles_labels()
    ax_len.legend(h1 + h2, l1 + l2, loc="best")
    return _save_fig(fig, out, msg=f"wrote chattiness length/markdown plot -> {out}")


def summarize_chattiness(points: Sequence[ChattinessPoint]) -> list[dict[str, Any]]:
    """small table of win-rate drops for reports / json."""
    rows: list[dict[str, Any]] = []
    for p in points:
        rows.append(
            {
                "stage": p.stage,
                "mean_chars": p.mean_chars,
                "markdown_rate": p.markdown_rate,
                "win_rate_raw": p.win_rate_raw,
                "win_rate_lc": p.win_rate_lc,
                "win_rate_drop": p.win_rate_drop,
                "mean_char_delta_vs_ref": p.mean_char_delta_vs_ref,
            }
        )
    return rows
