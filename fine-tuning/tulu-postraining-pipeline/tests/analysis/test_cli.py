"""unit tests for analysis cli helpers and script wiring (no matplotlib)."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from analysis.io import merge_stage_map, parse_stage_pairs


def _load_script(name: str):
    path = Path(__file__).resolve().parents[2] / "scripts" / "analysis" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"analysis_script_{name}", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _skills(ifeval: float, mmlu: float) -> dict:
    return {"ifeval_prompt_strict": ifeval, "mmlu_acc": mmlu}


def _style(win_raw: float, win_lc: float) -> dict:
    return {
        "style_b": {"mean_chars": 1000.0, "markdown_rate": 0.4},
        "raw": {"win_rate_b": win_raw},
        "length_controlled": {"win_rate_b": win_lc},
        "mean_char_delta_b_minus_a": 50.0,
    }


def _h2h_summary(wins: list[float]) -> dict:
    return {
        "reports": [
            {
                "run": i,
                "raw": {"win_rate_b": w},
                "length_controlled": {"win_rate_b": w - 0.05},
            }
            for i, w in enumerate(wins, start=1)
        ]
    }


def test_parse_stage_pairs_and_merge(tmp_path: Path) -> None:
    assert parse_stage_pairs(["sft=a.json", "ppo=b.json"]) == {
        "sft": "a.json",
        "ppo": "b.json",
    }
    with pytest.raises(ValueError, match="STAGE=PATH"):
        parse_stage_pairs(["nope"])

    map_path = tmp_path / "skills_map.json"
    map_path.write_text(json.dumps({"base": "base.json"}), encoding="utf-8")
    merged = merge_stage_map(map_path, ["sft=sft.json"])
    assert merged == {"base": "base.json", "sft": "sft.json"}


def test_attribution_and_verdict_scripts(tmp_path: Path) -> None:
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    skills_paths = {}
    for stage, ifeval, mmlu in [
        ("base", 0.2, 0.4),
        ("sft", 0.45, 0.38),
        ("dpo-b0.05", 0.44, 0.37),
        ("dpo-b0.1", 0.43, 0.37),
        ("ppo", 0.44, 0.36),
    ]:
        path = skills_dir / f"{stage}.json"
        path.write_text(json.dumps(_skills(ifeval, mmlu)), encoding="utf-8")
        skills_paths[stage] = str(path)

    style_dir = tmp_path / "style"
    style_dir.mkdir()
    style_paths = {}
    for stage, raw, lc in [
        ("dpo-b0.05", 0.58, 0.52),
        ("dpo-b0.1", 0.55, 0.51),
        ("ppo", 0.57, 0.53),
    ]:
        path = style_dir / f"{stage}.json"
        path.write_text(json.dumps(_style(raw, lc)), encoding="utf-8")
        style_paths[stage] = str(path)

    skills_map = tmp_path / "skills_map.json"
    style_map = tmp_path / "style_map.json"
    skills_map.write_text(json.dumps(skills_paths), encoding="utf-8")
    style_map.write_text(json.dumps(style_paths), encoding="utf-8")

    metrics = tmp_path / "metrics"
    attribution = _load_script("attribution")
    assert (
        attribution.main(
            [
                "--skills-json",
                str(skills_map),
                "--style-json",
                str(style_map),
                "--metrics-dir",
                str(metrics),
            ]
        )
        == 0
    )
    assert (metrics / "stage_attribution.json").is_file()

    dpo_sum = tmp_path / "dpo_summary.json"
    ppo_sum = tmp_path / "ppo_summary.json"
    dpo_sum.write_text(json.dumps(_h2h_summary([0.55, 0.56, 0.54])), encoding="utf-8")
    ppo_sum.write_text(json.dumps(_h2h_summary([0.57, 0.58, 0.56])), encoding="utf-8")
    verdict = _load_script("verdict")
    assert (
        verdict.main(
            [
                "--dpo-name",
                "dpo-b0.1",
                "--dpo-summary",
                str(dpo_sum),
                "--ppo-summary",
                str(ppo_sum),
                "--metrics-dir",
                str(metrics),
            ]
        )
        == 0
    )
    payload = json.loads((metrics / "dpo_vs_ppo_verdict.json").read_text())
    assert payload["dpo_name"] == "dpo-b0.1"
    assert (metrics / "dpo_vs_ppo_verdict.md").is_file()

    rs_sum = tmp_path / "rs_summary.json"
    rs_sum.write_text(json.dumps(_h2h_summary([0.70, 0.71, 0.69])), encoding="utf-8")
    bias_path = tmp_path / "judge_bias.json"
    bias_path.write_text(
        json.dumps(
            {
                "n": 3,
                "position": {"disagreement_rate": 0.0},
                "length": {"slope": 0.01},
                "self_preference": {"self_pref_rate": None, "n_mixed": 0},
                "logprob": {"agreement_rate": None},
            }
        ),
        encoding="utf-8",
    )
    rs_verdict = _load_script("rs_verdict")
    assert (
        rs_verdict.main(
            [
                "--summary",
                str(rs_sum),
                "--dpo-name",
                "dpo-b0.1",
                "--judge-bias",
                str(bias_path),
                "--metrics-dir",
                str(metrics),
            ]
        )
        == 0
    )
    rs_payload = json.loads((metrics / "rs_sft_vs_dpo_verdict.json").read_text())
    assert rs_payload["primary_winner"] == "rs_sft"
    assert rs_payload["judge_bias"]["position"]["disagreement_rate"] == 0.0
    assert (metrics / "rs_sft_vs_dpo_verdict.md").is_file()


def test_attribution_script_missing_skills() -> None:
    attribution = _load_script("attribution")
    assert attribution.main([]) == 2
