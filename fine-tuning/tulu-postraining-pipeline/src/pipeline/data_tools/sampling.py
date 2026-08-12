"""shared sampling helpers for subset builders."""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def largest_remainder_quotas(weights: dict[str, float], n: int) -> dict[str, int]:
    """allocate `n` seats proportional to weights (hamilton / largest remainder)."""
    if n < 0:
        raise ValueError(f"n must be >= 0, got {n}")
    if not weights or n == 0:
        return {k: 0 for k in weights}
    total_w = sum(max(0.0, w) for w in weights.values())
    if total_w <= 0:
        raise ValueError("weights must sum to a positive value")
    raw = {k: n * max(0.0, w) / total_w for k, w in weights.items()}
    quotas = {k: int(v) for k, v in raw.items()}
    remain = n - sum(quotas.values())
    order = sorted(raw.keys(), key=lambda k: (raw[k] - quotas[k], k), reverse=True)
    for k in order[:remain]:
        quotas[k] += 1
    return quotas


def sample_from_bucket(
    rows: Sequence[dict[str, Any]],
    k: int,
    rng: Any,
) -> list[dict[str, Any]]:
    """sample up to k rows without replacement."""
    if k <= 0 or not rows:
        return []
    if k >= len(rows):
        return list(rows)
    idxs = rng.choice(len(rows), size=k, replace=False)
    return [rows[int(i)] for i in idxs]


def top_up_rows(
    selected: Sequence[dict[str, Any]],
    pool: Sequence[dict[str, Any]],
    num_samples: int,
    rng: Any,
) -> list[dict[str, Any]]:
    """if `selected` is short, fill from remaining `pool` rows."""
    out = list(selected)
    if len(out) >= num_samples:
        return out
    selected_ids = {id(r) for r in out}
    remainder = [r for r in pool if id(r) not in selected_ids]
    need = num_samples - len(out)
    out.extend(sample_from_bucket(remainder, need, rng))
    return out


def shuffle_rows(
    rows: Sequence[dict[str, Any]],
    rng: Any,
) -> list[dict[str, Any]]:
    """shuffle rows with `rng`."""
    if not rows:
        return []
    order = rng.permutation(len(rows))
    return [rows[int(i)] for i in order]
