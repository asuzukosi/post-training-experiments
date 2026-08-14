"""unit tests for constraint-pair builder (no gpu).

invariant: stored prompt includes the constraint; chosen/rejected user turns
use that same prompt; chosen = with-constraint gen, rejected = without.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from data_tools.structured import (
    WITH_SUFFIX,
    WITHOUT_SUFFIX,
    assert_prompt_includes_constraint,
    build_generation_items,
    build_structured_pairs,
    gen_item_id,
    load_authored_prompts,
    pair_from_completions,
    render_constrained_prompt,
)


def _row(i: int = 0) -> dict:
    return {
        "id": f"c{i:03d}",
        "instruction": f"write a bio of person {i}.",
        "constraint": "reply with exactly 3 bullet points.",
    }



def test_assert_prompt_includes_constraint() -> None:
    prompt = render_constrained_prompt("write a bio", "use exactly 3 bullets")
    assert_prompt_includes_constraint(prompt, "use exactly 3 bullets")
    with pytest.raises(ValueError, match="constraint"):
        assert_prompt_includes_constraint("write a bio", "use exactly 3 bullets")



def test_pair_stores_constraint_on_both_sides() -> None:
    row = _row()
    constrained = render_constrained_prompt(row["instruction"], row["constraint"])
    pair = pair_from_completions(
        row,
        constrained_completion="- a\n- b\n- c",
        unconstrained_completion="ada lovelace was a mathematician.",
    )
    assert pair["prompt"] == constrained
    assert pair["prompt_id"] == row["id"]
    assert pair["chosen"][0]["content"] == constrained
    assert pair["rejected"][0]["content"] == constrained
    assert pair["chosen"][1]["content"] == "- a\n- b\n- c"
    assert pair["rejected"][1]["content"] == "ada lovelace was a mathematician."
    assert pair["rejected"][0]["content"] != row["instruction"]



def test_build_structured_pairs_from_gen_records() -> None:
    rows = [_row(0), _row(1)]
    records = []
    for row in rows:
        items = build_generation_items([row])
        records.append(
            {"id": items[0]["id"], "prompt": items[0]["prompt"], "completion": "with"}
        )
        records.append(
            {"id": items[1]["id"], "prompt": items[1]["prompt"], "completion": "without"}
        )
    pairs = build_structured_pairs(rows, records)
    assert len(pairs) == 2
    assert all(r["constraint"] in p["prompt"] for r, p in zip(rows, pairs))
    assert all(p["chosen"][1]["content"] == "with" for p in pairs)
    assert all(p["rejected"][1]["content"] == "without" for p in pairs)





def test_prepare_structured_writes_pairs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from prepare import structured as structured_prep

    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text(json.dumps(_row()) + "\n", encoding="utf-8")
    cfg = {
        "prompts_path": str(prompts),
        "processed_path": str(tmp_path / "processed"),
        "generations_path": str(tmp_path / "gen.jsonl"),
        "generator_model": "fake-model",
        "max_tokens": 16,
        "temperature": 0.0,
        "batch_size": 4,
    }
    captured: dict[str, list] = {}

    def fake_save(rows, path):
        captured["rows"] = list(rows)
        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)
        return out

    monkeypatch.setattr(structured_prep, "save_rows", fake_save)

    def fake_gen(prompts, *, model, max_tokens, temperature, top_p):
        return [
            "with-arm" if "exactly 3 bullet" in p else "without-arm" for p in prompts
        ]

    monkeypatch.setattr("eval.generate.vllm_generate", fake_gen)
    out = structured_prep.prepare_structured(cfg)
    assert out == Path(cfg["processed_path"])
    pairs = captured["rows"]
    assert len(pairs) == 1
    pair = pairs[0]
    assert "exactly 3 bullet" in pair["prompt"]
    assert pair["chosen"][0]["content"] == pair["prompt"]
    assert pair["rejected"][0]["content"] == pair["prompt"]
    assert pair["chosen"][1]["content"] == "with-arm"
    assert pair["rejected"][1]["content"] == "without-arm"


