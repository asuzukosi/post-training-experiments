"""unit tests for bon rs-sft row write."""
from __future__ import annotations

from pathlib import Path

import pytest

from eval.bon import build_rs_sft_row, run_bon_selection
from eval.io import append_jsonl


def test_build_rs_sft_row(candidate) -> None:
    row = build_rs_sft_row(candidate("p0", 3, "top"), n_candidates=8)
    assert row["messages"][0] == {"role": "user", "content": "write a bio"}
    assert row["messages"][1] == {"role": "assistant", "content": "top"}
    assert row["sample_idx"] == 3
    assert row["n_candidates"] == 8


def test_run_bon_selection_writes_rs_sft(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    candidate,
    lexicographic_score,
) -> None:
    gens = tmp_path / "gens.jsonl"
    for row in (
        candidate("p0", 0, "aa"),
        candidate("p0", 1, "zz"),
        candidate("p1", 0, "mm"),
        candidate("p1", 1, "nn"),
    ):
        append_jsonl(gens, row)

    captured: dict[str, list] = {}

    def fake_save(rows, path):
        captured["rows"] = list(rows)
        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)
        return out

    monkeypatch.setattr("eval.judge.score_with_judgearena", lexicographic_score)
    monkeypatch.setattr("eval.bon.select.save_rows", fake_save)

    processed = tmp_path / "rs_sft"
    out = run_bon_selection(
        generations_path=gens,
        output_dir=tmp_path / "bon",
        processed_path=processed,
        judge_model="fake-14b",
    )
    assert out == processed
    rows = captured["rows"]
    assert len(rows) == 2
    by_prompt_id = {r["prompt_id"]: r for r in rows}
    assert by_prompt_id["p0"]["messages"][1]["content"] == "zz"
    assert by_prompt_id["p1"]["messages"][1]["content"] == "nn"
    assert by_prompt_id["p0"]["source"] == "rs_sft"
    assert by_prompt_id["p0"]["n_candidates"] == 2
    assert (tmp_path / "bon" / "bon_selections.jsonl").is_file()
