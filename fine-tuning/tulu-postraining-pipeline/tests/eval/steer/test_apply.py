"""unit tests for α-steer and activation capping. no gpu."""
from __future__ import annotations

import pytest
import torch

from eval.steer.apply import cap_hidden, parse_alphas, steer_hidden


def test_steer_hidden_adds_alpha_v() -> None:
    h = torch.zeros(2, 3)
    v = torch.tensor([1.0, 0.0, 0.0])
    out = steer_hidden(h, v, alpha=-2.0)
    assert out[0, 0] == pytest.approx(-2.0)
    assert out[1, 1] == pytest.approx(0.0)


def test_cap_hidden_clamps_excess_to_tau() -> None:
    v = torch.tensor([1.0, 0.0])
    h = torch.tensor([[2.0, 0.0], [0.1, 0.0]])
    out = cap_hidden(h, v, tau=0.5)
    # first row was 2.0 along v -> clipped to 0.5; second already below
    assert float((out[0] * v).sum()) == pytest.approx(0.5)
    assert float((out[1] * v).sum()) == pytest.approx(0.1)


def test_parse_alphas() -> None:
    assert parse_alphas(None) == [-2.0, -1.0, 0.0, 1.0, 2.0]
    assert parse_alphas("-2,0,2") == [-2.0, 0.0, 2.0]
    with pytest.raises(ValueError, match="non-empty"):
        parse_alphas([])


def test_residual_hook_adds_alpha_v() -> None:
    import torch.nn as nn

    from eval.steer.apply import register_residual_hook

    class _block(nn.Module):
        def forward(self, x):
            return x

    class _inner(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([_block(), _block()])

    class _model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = _inner()

    m = _model()
    handle = register_residual_hook(
        m, layer=0, vector=[1.0, 0.0], alpha=2.0
    )
    try:
        out = m.model.layers[0](torch.zeros(1, 2, 2))
    finally:
        handle.remove()
    assert float(out[0, 0, 0]) == pytest.approx(2.0)
    # hook removed: second pass is unsteered
    out2 = m.model.layers[0](torch.zeros(1, 2, 2))
    assert float(out2[0, 0, 0]) == pytest.approx(0.0)
