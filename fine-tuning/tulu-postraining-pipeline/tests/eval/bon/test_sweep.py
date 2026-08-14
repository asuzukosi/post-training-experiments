"""unit tests for proxy bon selection and the n-sweep helper."""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from eval.bon import group_candidates, run_bon_sweep, select_top1_by_proxy
from eval.bon.proxy import score_proxy_incremental
from eval.bon.sweep import parse_n_values
from eval.io import append_jsonl


def _row(
    prompt_id: str,
    sample_idx: int,
    completion: str,
    *,
    proxy_score: float,
    prompt: str = "q",
    avg_logprob: float | None = None,
) -> dict:
    row = {
        "id": f"{prompt_id}__{sample_idx}",
        "prompt_id": prompt_id,
        "prompt": prompt,
        "completion": completion,
        "sample_idx": sample_idx,
        "proxy_score": proxy_score,
    }
    if avg_logprob is not None:
        row["avg_logprob"] = avg_logprob
    return row


def test_group_candidates_keeps_proxy_score() -> None:
    grouped = group_candidates([_row("p0", 0, "aa", proxy_score=1.5)])
    assert grouped["p0"][0]["proxy_score"] == 1.5


def test_select_top1_by_proxy_uses_nested_prefix() -> None:
    grouped = group_candidates(
        [
            _row("p0", 0, "a", proxy_score=0.1),
            _row("p0", 1, "b", proxy_score=0.4),
            _row("p0", 2, "c", proxy_score=0.9),
            _row("p0", 3, "d", proxy_score=0.2),
        ]
    )
    n2 = select_top1_by_proxy(grouped, 2)
    n4 = select_top1_by_proxy(grouped, 4)
    assert n2["p0"]["completion"] == "b"
    assert n4["p0"]["completion"] == "c"


def test_select_top1_by_proxy_tie_breaks_to_lower_sample_idx() -> None:
    grouped = group_candidates(
        [
            _row("p0", 0, "a", proxy_score=1.0),
            _row("p0", 1, "b", proxy_score=1.0),
        ]
    )
    assert select_top1_by_proxy(grouped, 2)["p0"]["sample_idx"] == 0


def test_select_top1_by_proxy_requires_scores_and_idx() -> None:
    with pytest.raises(ValueError, match="proxy_score"):
        select_top1_by_proxy(
            group_candidates(
                [{"id": "p0__0", "prompt_id": "p0", "prompt": "q", "completion": "a"}]
            ),
            1,
        )
    grouped = group_candidates([_row("p0", 0, "a", proxy_score=0.1)])
    with pytest.raises(ValueError, match="missing sample_idx"):
        select_top1_by_proxy(grouped, 2)



def test_score_proxy_incremental_skips_existing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gens = tmp_path / "gens.jsonl"
    append_jsonl(gens, _row("p0", 0, "a", proxy_score=0.0))
    scored = tmp_path / "scored.jsonl"
    append_jsonl(scored, _row("p0", 0, "a", proxy_score=9.0))

    def boom(*_args, **_kwargs):
        raise AssertionError("should not rescore completed ids")

    monkeypatch.setattr("eval.bon.proxy.score_with_rm", boom)
    out = score_proxy_incremental(
        gens, rm_checkpoint=tmp_path / "rm", output_path=scored
    )
    assert out == scored


def test_score_proxy_incremental_writes_scores(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gens = tmp_path / "gens.jsonl"
    append_jsonl(
        gens,
        {
            "id": "p0__0",
            "prompt_id": "p0",
            "prompt": "q",
            "completion": "a",
            "sample_idx": 0,
        },
    )

    monkeypatch.setattr(
        "eval.bon.proxy.score_with_rm", lambda rows, **_kwargs: [3.5] * len(rows)
    )
    out = score_proxy_incremental(
        gens, rm_checkpoint=tmp_path / "rm", output_path=tmp_path / "scored.jsonl"
    )
    rows = json.loads(out.read_text().splitlines()[0])
    assert rows["proxy_score"] == 3.5


def test_run_bon_sweep_records_proxy_gold_kl(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    lexicographic_score,
) -> None:
    gens = tmp_path / "gens.jsonl"
    for row in (
        _row("p0", 0, "aa", proxy_score=0.1, avg_logprob=-2.0),
        _row("p0", 1, "zz", proxy_score=0.8, avg_logprob=-1.0),
        _row("p1", 0, "mm", proxy_score=0.2, avg_logprob=-2.0),
        _row("p1", 1, "nn", proxy_score=0.9, avg_logprob=-1.0),
    ):
        append_jsonl(gens, row)

    monkeypatch.setattr("eval.judge.score_with_judgearena", lexicographic_score)
    report = run_bon_sweep(
        generations_path=gens,
        output_dir=tmp_path / "sweep",
        n_values=(1, 2),
        judge_model="fake-32b",
    )
    by_n = {p.n: p for p in report.points}
    assert by_n[1].kl == pytest.approx(0.0)
    assert by_n[2].kl == pytest.approx(math.log(2))
    assert by_n[1].mean_proxy == pytest.approx(0.15)
    assert by_n[2].mean_proxy == pytest.approx(0.85)
    assert by_n[1].gold_win_rate == 0.5
    assert by_n[1].gold_win_rate_lc == 0.5
    assert by_n[2].gold_win_rate == 1.0
    assert (tmp_path / "sweep" / "sweep.json").is_file()
    assert (tmp_path / "sweep" / "n1" / "selections.jsonl").is_file()
    assert (tmp_path / "sweep" / "n2" / "judge.jsonl").is_file()
