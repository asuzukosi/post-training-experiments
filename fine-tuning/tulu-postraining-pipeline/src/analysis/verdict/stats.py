"""mean / std / 95% ci over repeated judge runs."""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

# student-t 95% two-sided critical values by sample size n (df = n-1)
_T95_BY_N = {
    2: 12.706,
    3: 4.303,
    4: 3.182,
    5: 2.776,
    6: 2.571,
    7: 2.447,
    8: 2.365,
    9: 2.306,
    10: 2.262,
}


@dataclass
class RunSummary:
    """mean / std / 95% ci over repeated runs."""

    n: int
    mean: float
    std: float
    ci95_low: float
    ci95_high: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def t_crit_95(n: int) -> float:
    if n < 2:
        return float("inf")
    return _T95_BY_N.get(n, 1.96)


def summarize_runs(values: Sequence[float]) -> RunSummary:
    """mean/std and 95% ci of the mean over repeated runs."""
    xs = [float(v) for v in values]
    n = len(xs)
    if n == 0:
        raise ValueError("values must be non-empty")
    mean = sum(xs) / n
    if n == 1:
        return RunSummary(n=1, mean=mean, std=0.0, ci95_low=mean, ci95_high=mean)
    var = sum((x - mean) ** 2 for x in xs) / (n - 1)
    std = math.sqrt(var)
    se = std / math.sqrt(n)
    half = t_crit_95(n) * se
    return RunSummary(
        n=n,
        mean=mean,
        std=std,
        ci95_low=mean - half,
        ci95_high=mean + half,
    )
