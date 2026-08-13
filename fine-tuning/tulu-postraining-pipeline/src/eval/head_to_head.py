"""head-to-head orchestration: generate a/b → judge → style report.

sequential on one gpu: finish model a gens, then model b, then load judge.
"""
from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from eval.generate import generate_incremental
from eval.io import load_jsonl
from eval.judge import DEFAULT_JUDGE_MODEL, judge_incremental, judgment_id
from eval.style import (
    DEFAULT_MAX_REL_LENGTH_DIFF,
    report_head_to_head_style,
)
from prepare.paths import ROOT, resolve_path


def model_ref(raw: str | Path) -> str:
    """resolve local checkpoint paths; leave hub ids unchanged."""
    s = str(raw)
    p = Path(s)
    if p.is_absolute():
        return str(p)
    # project-relative artifact
    if s.startswith(("results/", "data/", "./")) or (ROOT / s).exists():
        return str(resolve_path(s))
    return s  # e.g. Qwen/Qwen2.5-1.5B


def load_prompt_items(path: str | Path) -> list[dict[str, Any]]:
    """load jsonl rows with `id` + `prompt` (prompt may be raw user text)."""
    rows = load_jsonl(path)
    if not rows:
        raise ValueError(f"no prompts in {path}")
    out: list[dict[str, Any]] = []
    for row in rows:
        if "id" not in row or "prompt" not in row:
            raise ValueError(f"prompt row needs id+prompt: {row!r}")
        out.append({"id": str(row["id"]), "prompt": str(row["prompt"])})
    return out


def apply_chat_template_to_prompts(
    items: Sequence[Mapping[str, Any]],
    *,
    tokenizer_source: str | Path,
) -> list[dict[str, Any]]:
    """render user prompts with chat template + generation prompt."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(tokenizer_source), trust_remote_code=True)
    rendered: list[dict[str, Any]] = []
    for item in items:
        messages = [{"role": "user", "content": item["prompt"]}]
        text = tok.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        rendered.append({"id": str(item["id"]), "prompt": text})
    return rendered


def _index_by_id(rows: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    out: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        out[str(row["id"])] = row
    return out


def build_judge_items(
    prompts: Sequence[Mapping[str, Any]],
    gens_a: Sequence[Mapping[str, Any]],
    gens_b: Sequence[Mapping[str, Any]],
    *,
    model_a: str,
    model_b: str,
    run: int,
) -> list[dict[str, Any]]:
    """join generation jsonl rows into judge pairs for one run."""
    by_a = _index_by_id(gens_a)
    by_b = _index_by_id(gens_b)
    pairs: list[dict[str, Any]] = []
    for item in prompts:
        pid = str(item["id"])
        if pid not in by_a or pid not in by_b:
            raise ValueError(f"missing generation for prompt id={pid}")
        pairs.append(
            {
                "id": judgment_id(pid, run=run),
                "prompt": item["prompt"],
                "completion_a": str(by_a[pid].get("completion") or ""),
                "completion_b": str(by_b[pid].get("completion") or ""),
                "model_a": model_a,
                "model_b": model_b,
                "run": run,
                "prompt_id": pid,
            }
        )
    return pairs


def run_head_to_head(
    *,
    model_a: str | Path,
    model_b: str | Path,
    prompts_path: str | Path,
    output_dir: str | Path,
    runs: int = 3,
    judge_model: str = DEFAULT_JUDGE_MODEL,
    max_rel_length_diff: float = DEFAULT_MAX_REL_LENGTH_DIFF,
) -> dict[str, Any]:
    """run multi-run head-to-head; write gens/judgments/style under output_dir."""
    if runs < 1:
        raise ValueError(f"runs must be >= 1, got {runs}")

    out_dir = resolve_path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_a_s = model_ref(model_a)
    model_b_s = model_ref(model_b)

    raw_prompts = load_prompt_items(prompts_path)
    # store raw user text for judge display; generate with templated strings
    user_prompts = [{"id": r["id"], "prompt": r["prompt"]} for r in raw_prompts]
    gen_prompts = apply_chat_template_to_prompts(
        user_prompts,
        tokenizer_source=model_a_s,
    )

    name_a = Path(model_a_s).name
    name_b = Path(model_b_s).name
    all_reports: list[dict[str, Any]] = []

    for run in range(1, runs + 1):
        gens_a_path = out_dir / f"gens_{name_a}_r{run}.jsonl"
        gens_b_path = out_dir / f"gens_{name_b}_r{run}.jsonl"
        judge_path = out_dir / f"judge_{name_a}_vs_{name_b}_r{run}.jsonl"

        print(f"head-to-head run={run}/{runs}: generate {name_a}")
        generate_incremental(
            gen_prompts,
            model=model_a_s,
            output_path=gens_a_path,
        )
        print(f"head-to-head run={run}/{runs}: generate {name_b}")
        generate_incremental(
            gen_prompts,
            model=model_b_s,
            output_path=gens_b_path,
        )

        pairs = build_judge_items(
            user_prompts,
            load_jsonl(gens_a_path),
            load_jsonl(gens_b_path),
            model_a=model_a_s,
            model_b=model_b_s,
            run=run,
        )
        print(f"head-to-head run={run}/{runs}: judge {judge_model}")
        judge_incremental(
            pairs,
            judge_model=judge_model,
            output_path=judge_path,
        )

        report = report_head_to_head_style(
            load_jsonl(judge_path),
            max_rel_length_diff=max_rel_length_diff,
        )
        report_path = out_dir / f"style_{name_a}_vs_{name_b}_r{run}.json"
        payload = report.to_dict()
        payload["run"] = run
        payload["model_a"] = model_a_s
        payload["model_b"] = model_b_s
        payload["judge_path"] = str(judge_path)
        with report_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
            f.write("\n")
        print(
            f"head-to-head run={run}: raw_win_b={report.raw.win_rate_b} "
            f"lc_win_b={report.length_controlled.win_rate_b} wrote={report_path}"
        )
        all_reports.append(payload)

    summary = {
        "model_a": model_a_s,
        "model_b": model_b_s,
        "judge_model": judge_model,
        "runs": runs,
        "output_dir": str(out_dir),
        "reports": all_reports,
    }
    summary_path = out_dir / f"summary_{name_a}_vs_{name_b}.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")
    print(f"head-to-head done: summary={summary_path}")
    return summary
