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
from eval.judge import DEFAULT_JUDGE_MODEL, judge_incremental
from eval.style import report_head_to_head_style
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
) -> list[dict[str, Any]]:
    """join generation jsonl rows into judge pairs."""
    by_a = _index_by_id(gens_a)
    by_b = _index_by_id(gens_b)
    pairs: list[dict[str, Any]] = []
    for item in prompts:
        prompt_id = str(item["id"])
        if prompt_id not in by_a or prompt_id not in by_b:
            raise ValueError(f"missing generation for prompt id={prompt_id}")
        pairs.append(
            {
                "id": prompt_id,
                "prompt": item["prompt"],
                "completion_a": str(by_a[prompt_id].get("completion") or ""),
                "completion_b": str(by_b[prompt_id].get("completion") or ""),
                "model_a": model_a,
                "model_b": model_b,
                "prompt_id": prompt_id,
            }
        )
    return pairs


def cached_generations(
    prompts: Sequence[Mapping[str, Any]],
    *,
    model: str,
    gens_dir: Path,
) -> Path:
    """generate once per model into a shared directory, reused across comparisons.

    the same checkpoint appears in several head-to-heads — sft is in four of them — and
    generation is deterministic at temperature 0 over a frozen prompt set, so every
    comparison after the first has nothing to recompute. `generate_incremental` already
    skips ids it has finished, so a shared path turns 12 generation passes into 6 and
    drops the same number of vllm engine inits.

    the hazard a shared cache introduces is serving one model's completions as another's
    — two checkpoints whose directories share a basename would collide silently — so the
    model recorded in the file is checked rather than trusted.
    """
    gens_dir.mkdir(parents=True, exist_ok=True)
    path = gens_dir / f"gens_{Path(model).name}.jsonl"
    generate_incremental(prompts, model=model, output_path=path)
    rows = load_jsonl(path)
    wrong = sorted({str(r.get("model")) for r in rows} - {model})
    if wrong:
        raise ValueError(
            f"cached generations at {path} were produced by {wrong}, not {model!r}; "
            "two checkpoints share a basename — give one an explicit --output-dir"
        )
    return path


def run_head_to_head(
    *,
    model_a: str | Path,
    model_b: str | Path,
    prompts_path: str | Path,
    output_dir: str | Path,
    gens_dir: str | Path | None = None,
    judge_model: str = DEFAULT_JUDGE_MODEL,
) -> dict[str, Any]:
    """one judged pass over the prompt set; write judgments and the style report.

    ONE pass, not three. generation and judging both run at temperature 0, so repeating
    the identical comparison returns the identical number — averaging repeats would
    report a standard deviation of zero and manufacture confidence. the uncertainty that
    matters is over PROMPTS, and that comes from the binomial interval on the judged
    pairs, which `analysis.verdict` computes.

    `gens_dir` defaults to a sibling of `output_dir` shared by every comparison, so a
    model that appears in several of them generates once.
    """
    out_dir = resolve_path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    shared_gens = resolve_path(gens_dir) if gens_dir is not None else out_dir.parent / "generations"
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
    judge_path = out_dir / f"judge_{name_a}_vs_{name_b}.jsonl"

    print(f"head-to-head: generate {name_a}")
    gens_a_path = cached_generations(gen_prompts, model=model_a_s, gens_dir=shared_gens)
    print(f"head-to-head: generate {name_b}")
    gens_b_path = cached_generations(gen_prompts, model=model_b_s, gens_dir=shared_gens)

    pairs = build_judge_items(
        user_prompts,
        load_jsonl(gens_a_path),
        load_jsonl(gens_b_path),
        model_a=model_a_s,
        model_b=model_b_s,
    )
    print(f"head-to-head: judge {judge_model}")
    judge_incremental(
        pairs,
        judge_model=judge_model,
        output_path=judge_path,
    )

    report = report_head_to_head_style(load_jsonl(judge_path))
    summary = report.to_dict()
    summary.update(
        {
            "model_a": model_a_s,
            "model_b": model_b_s,
            "judge_model": judge_model,
            "judge_path": str(judge_path),
            "generations_dir": str(shared_gens),
            "output_dir": str(out_dir),
        }
    )
    summary_path = out_dir / f"summary_{name_a}_vs_{name_b}.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")
    print(f"head-to-head done: raw_win_b={report.raw.win_rate_b} summary={summary_path}")
    return summary
