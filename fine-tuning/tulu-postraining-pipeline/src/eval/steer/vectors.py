"""the sycophancy vector itself: contrastive maths, trait pairs, persistence.

v_l = mean(h_l | s+) - mean(h_l | s-), then unit-normalised. s+ / s- are
trait-eliciting vs trait-suppressing prompts.

NOTHING HERE TOUCHES A MODEL - that is the point of the split. Everything in this file
is pure tensor arithmetic or file i/o and is covered by cpu tests; the forward passes
that produce the hiddens live in `extract.py`, and putting the vector back into a
running model lives in `apply.py`.
"""
from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from eval.io import ID_KEY, load_jsonl
from prepare.paths import resolve_path

PROMPT_POS_KEY = "prompt_pos"
PROMPT_NEG_KEY = "prompt_neg"
EPS = 1e-8


@dataclass
class LayerVector:
    """one layer's unit contrastive vector + cap threshold."""

    layer: int
    vector: list[float]
    tau: float | None
    n_pos: int
    n_neg: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SycophancyVectors:
    model: str
    layers: list[LayerVector]

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "layers": [layer.to_dict() for layer in self.layers],
        }

    def by_layer(self) -> dict[int, LayerVector]:
        return {layer.layer: layer for layer in self.layers}


def middle_layer_ids(n_layers: int) -> list[int]:
    """inclusive [n/3, 2n/3)."""
    if n_layers < 1:
        raise ValueError(f"n_layers must be >= 1, got {n_layers}")
    lo = n_layers // 3
    hi = max(lo + 1, (2 * n_layers) // 3)
    return list(range(lo, hi))


def require_trait_pair(row: Mapping[str, Any], *, line_no: int | None = None) -> dict[str, str]:
    loc = f" at line {line_no}" if line_no is not None else ""
    if ID_KEY not in row:
        raise ValueError(f"trait pair missing {ID_KEY!r}{loc}")
    for key in (PROMPT_POS_KEY, PROMPT_NEG_KEY):
        val = row.get(key)
        if not isinstance(val, str) or not val.strip():
            raise ValueError(f"trait pair {key!r} must be a non-empty str{loc}")
    return {
        ID_KEY: str(row[ID_KEY]),
        PROMPT_POS_KEY: str(row[PROMPT_POS_KEY]),
        PROMPT_NEG_KEY: str(row[PROMPT_NEG_KEY]),
    }


def load_trait_pairs(path: str | Path) -> list[dict[str, str]]:
    rows = load_jsonl(path)
    if not rows:
        raise ValueError(f"no trait pairs in {path}")
    return [require_trait_pair(row, line_no=i) for i, row in enumerate(rows, start=1)]


def contrastive_vector(
    pos: Sequence[Sequence[float]],
    neg: Sequence[Sequence[float]],
) -> list[float]:
    """unit v = mean(s+) − mean(s−)."""
    import torch

    if not pos or not neg:
        raise ValueError("need at least one s+ and one s− hidden")
    pos_t = torch.tensor(pos, dtype=torch.float32)
    neg_t = torch.tensor(neg, dtype=torch.float32)
    if pos_t.ndim != 2 or neg_t.ndim != 2:
        raise ValueError("hiddens must be [n, hidden]")
    if pos_t.shape[-1] != neg_t.shape[-1]:
        raise ValueError(
            f"s+ / s− hidden size mismatch: {pos_t.shape[-1]} vs {neg_t.shape[-1]}"
        )
    v = pos_t.mean(dim=0) - neg_t.mean(dim=0)
    norm = float(torch.linalg.vector_norm(v))
    if norm < EPS:
        raise ValueError("contrastive vector is ~0; s+ and s− hiddens are identical")
    return (v / norm).tolist()


def cap_tau(hiddens: Sequence[Sequence[float]], vector: Sequence[float]) -> float:
    """25th percentile of <h, v> over training last-token hiddens."""
    import torch

    if not hiddens:
        raise ValueError("need hiddens to fit tau")
    h = torch.tensor(hiddens, dtype=torch.float32)
    v = torch.tensor(vector, dtype=torch.float32)
    proj = h @ v
    return float(torch.quantile(proj, 0.25))


def save_vectors(vectors: SycophancyVectors, path: str | Path) -> Path:
    out = resolve_path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(vectors.to_dict(), f)
        f.write("\n")
    print(f"sycophancy vectors: layers={len(vectors.layers)} wrote={out}")
    return out


def load_vectors(path: str | Path) -> SycophancyVectors:
    payload = json.loads(resolve_path(path).read_text(encoding="utf-8"))
    layers = [
        LayerVector(
            layer=int(row["layer"]),
            vector=[float(x) for x in row["vector"]],
            tau=None if row.get("tau") is None else float(row["tau"]),
            n_pos=int(row.get("n_pos") or 0),
            n_neg=int(row.get("n_neg") or 0),
        )
        for row in payload["layers"]
    ]
    return SycophancyVectors(model=str(payload.get("model") or ""), layers=layers)
