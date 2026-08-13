"""kl-frontier utility: gold-vs-kl curves, peak detection, reusable kl bound.

the bound is the kl at the first gold peak (lc preferred). other bets
(ppo early-stop, mitigations) adopt that number as 'how far to push'.
a plateau at the last point is reported as a plateau, not a decline.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from analysis.io import (
    DEFAULT_METRICS_DIR,
    DEFAULT_PLOTS_DIR,
    as_float,
    load_json_mapping,
    resolve_output_path,
    write_json,
)
from prepare.paths import resolve_path

DEFAULT_FRONTIER_PATH = DEFAULT_METRICS_DIR / "kl_frontier.json"
DEFAULT_FRONTIER_PLOT = DEFAULT_PLOTS_DIR / "kl_frontier_gold_vs_kl.png"
DEFAULT_INVERTED_U_PLOT = DEFAULT_PLOTS_DIR / "inverted_u_proxy_gold.png"
PEAK_EPS = 1e-9
Shape = Literal["decline", "plateau", "peak_at_max_kl"]


@dataclass
class FrontierPoint:
    """one kl coordinate on the over-opt curve."""

    kl: float
    gold: float | None = None
    gold_lc: float | None = None
    proxy: float | None = None
    n: int | None = None
    source: str = "bon"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GoldPeak:
    kl: float
    gold: float
    metric: str
    n: int | None = None
    shape: Shape = "peak_at_max_kl"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class KlBound:
    """kl budget other bets should not push past (lc peak if present)."""

    kl: float
    gold: float
    metric: str
    shape: Shape
    n: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class KlFrontier:
    points: list[FrontierPoint]
    peak_raw: GoldPeak | None
    peak_lc: GoldPeak | None
    bound: KlBound | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "points": [p.to_dict() for p in self.points],
            "peak_raw": None if self.peak_raw is None else self.peak_raw.to_dict(),
            "peak_lc": None if self.peak_lc is None else self.peak_lc.to_dict(),
            "bound": None if self.bound is None else self.bound.to_dict(),
        }


def points_from_sweep(source: str | Path | Mapping[str, Any]) -> list[FrontierPoint]:
    """load frontier points from a bon sweep.json."""
    payload = load_json_mapping(source)
    rows = payload.get("points") or []
    if not isinstance(rows, list) or not rows:
        raise ValueError("sweep json missing points")
    points: list[FrontierPoint] = []
    for row in rows:
        if not isinstance(row, Mapping) or row.get("kl") is None:
            raise ValueError(f"sweep point needs kl: {row!r}")
        n_raw = row.get("n")
        points.append(
            FrontierPoint(
                kl=float(row["kl"]),
                gold=as_float(row.get("gold_win_rate")),
                gold_lc=as_float(row.get("gold_win_rate_lc")),
                proxy=as_float(row.get("mean_proxy")),
                n=None if n_raw is None else int(n_raw),
                source="bon",
            )
        )
    return sorted(points, key=lambda p: p.kl)


def _gold_of(point: FrontierPoint, metric: str) -> float | None:
    return point.gold_lc if metric == "gold_lc" else point.gold


def _shape_after_peak(
    points: Sequence[FrontierPoint],
    peak: FrontierPoint,
    metric: str,
    peak_gold: float,
) -> Shape:
    later = [
        p
        for p in points
        if p.kl > peak.kl and _gold_of(p, metric) is not None
    ]
    if not later:
        return "peak_at_max_kl"
    if any((_gold_of(p, metric) or 0.0) < peak_gold - PEAK_EPS for p in later):
        return "decline"
    return "plateau"


def detect_gold_peak(
    points: Sequence[FrontierPoint],
    *,
    metric: str = "gold",
) -> GoldPeak | None:
    """first (lowest-kl) point at the max gold for `metric`."""
    if metric not in ("gold", "gold_lc"):
        raise ValueError(f"metric must be gold or gold_lc, got {metric!r}")
    scored: list[tuple[FrontierPoint, float]] = []
    for point in points:
        value = _gold_of(point, metric)
        if value is not None:
            scored.append((point, value))
    if not scored:
        return None
    best = max(v for _, v in scored)
    ordered = sorted(scored, key=lambda item: item[0].kl)
    peak_pt = next(p for p, v in ordered if v >= best - PEAK_EPS)
    return GoldPeak(
        kl=peak_pt.kl,
        gold=best,
        metric=metric,
        n=peak_pt.n,
        shape=_shape_after_peak(points, peak_pt, metric, best),
    )


def build_kl_frontier(points: Sequence[FrontierPoint]) -> KlFrontier:
    """peak + bound from a kl-ordered series (usually bon sweep points)."""
    ordered = sorted(points, key=lambda p: p.kl)
    if not ordered:
        raise ValueError("points must be non-empty")
    peak_raw = detect_gold_peak(ordered, metric="gold")
    peak_lc = detect_gold_peak(ordered, metric="gold_lc")
    primary = peak_lc or peak_raw
    bound = None
    if primary is not None:
        bound = KlBound(
            kl=primary.kl,
            gold=primary.gold,
            metric=primary.metric,
            shape=primary.shape,
            n=primary.n,
        )
    return KlFrontier(
        points=list(ordered),
        peak_raw=peak_raw,
        peak_lc=peak_lc,
        bound=bound,
    )


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_gold_vs_kl(
    frontier: KlFrontier,
    path: str | Path | None = None,
    *,
    title: str = "gold vs kl (raw + length-controlled)",
) -> Path:
    """plot gold raw and lc vs kl; mark the bound."""
    if not frontier.points:
        raise ValueError("frontier.points must be non-empty")
    plt = _pyplot()
    out = resolve_output_path(path, default=DEFAULT_FRONTIER_PLOT)
    kls = [p.kl for p in frontier.points]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(
        kls,
        [float("nan") if p.gold is None else p.gold for p in frontier.points],
        marker="o",
        color="C0",
        label="gold raw",
    )
    ax.plot(
        kls,
        [float("nan") if p.gold_lc is None else p.gold_lc for p in frontier.points],
        marker="s",
        color="C1",
        label="gold lc",
    )
    if frontier.bound is not None:
        ax.axvline(
            frontier.bound.kl,
            color="C3",
            linestyle="--",
            label=f"bound kl={frontier.bound.kl:.3g} ({frontier.bound.shape})",
        )
    ax.set_xlabel("kl")
    ax.set_ylabel("gold win-rate")
    ax.set_title(title)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote kl-frontier plot -> {out}")
    return out


def _nan_or(values: Sequence[float | None]) -> list[float]:
    return [float("nan") if v is None else float(v) for v in values]


def plot_inverted_u(
    frontier: KlFrontier,
    path: str | Path | None = None,
    *,
    title: str = "proxy vs gold vs log n",
) -> Path:
    """proxy climbs vs log n; gold raw+lc is the inverted-u (or plateau).

    twin y: left = gold win-rate, right = mean proxy. x = n on a log2
    axis when every point has n, else ln n (kl).
    """
    if not frontier.points:
        raise ValueError("frontier.points must be non-empty")
    if all(p.proxy is None for p in frontier.points):
        raise ValueError("inverted-u needs at least one proxy value")
    if all(p.gold is None and p.gold_lc is None for p in frontier.points):
        raise ValueError("inverted-u needs at least one gold value")

    plt = _pyplot()
    out = resolve_output_path(path, default=DEFAULT_INVERTED_U_PLOT)
    use_n = all(p.n is not None and p.n >= 1 for p in frontier.points)
    xs = (
        [float(p.n) for p in frontier.points]
        if use_n
        else [p.kl for p in frontier.points]
    )

    fig, ax_gold = plt.subplots(figsize=(7.5, 4.5))
    ax_proxy = ax_gold.twinx()
    ax_gold.plot(
        xs,
        _nan_or([p.gold for p in frontier.points]),
        marker="o",
        color="C0",
        label="gold raw",
    )
    ax_gold.plot(
        xs,
        _nan_or([p.gold_lc for p in frontier.points]),
        marker="s",
        color="C1",
        label="gold lc",
    )
    ax_proxy.plot(
        xs,
        _nan_or([p.proxy for p in frontier.points]),
        marker="^",
        color="C2",
        label="proxy",
    )
    if use_n:
        ax_gold.set_xscale("log", base=2)
        ax_gold.set_xticks(xs)
        ax_gold.set_xticklabels([str(int(x)) for x in xs])
        ax_gold.set_xlabel("n")
    else:
        ax_gold.set_xlabel("ln n")
    ax_gold.set_ylabel("gold win-rate")
    ax_gold.set_ylim(0.0, 1.0)
    ax_proxy.set_ylabel("mean proxy")
    ax_gold.set_title(title)
    h1, l1 = ax_gold.get_legend_handles_labels()
    h2, l2 = ax_proxy.get_legend_handles_labels()
    ax_gold.legend(h1 + h2, l1 + l2, loc="best")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote inverted-u plot -> {out}")
    return out


def write_kl_frontier(
    frontier: KlFrontier,
    path: str | Path | None = None,
    *,
    plot_path: str | Path | None = None,
) -> Path:
    """write frontier json; optionally the gold-vs-kl plot."""
    out = DEFAULT_FRONTIER_PATH if path is None else resolve_path(path)
    write_json(out, frontier.to_dict())
    bound = frontier.bound
    print(
        "wrote kl-frontier -> "
        f"{out} bound_kl={None if bound is None else bound.kl} "
        f"shape={None if bound is None else bound.shape}"
    )
    if plot_path is not None:
        plot_gold_vs_kl(frontier, plot_path)
    return out
