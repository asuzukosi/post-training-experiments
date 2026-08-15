"""compare tuned vs untuned rlhf eval jsonl side by side."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

# defaults from the deeplearning.ai rlhf lesson
DEFAULT_PARAMETER_VALUES = {
    "preference_dataset": (
        "gs://vertex-ai/generative-ai/rlhf/text_small/"
        "summarize_from_feedback_tfds/comparisons/train/*.jsonl"
    ),
    "prompt_dataset": (
        "gs://vertex-ai/generative-ai/rlhf/text_small/reddit_tfds/train/*.jsonl"
    ),
    "eval_dataset": (
        "gs://vertex-ai/generative-ai/rlhf/text_small/reddit_tfds/val/*.jsonl"
    ),
    "large_model_reference": "llama-2-7b",
    "reward_model_train_steps": 1410,
    "reinforcement_learning_train_steps": 320,
    "reward_model_learning_rate_multiplier": 1.0,
    "reinforcement_learning_rate_multiplier": 1.0,
    "kl_coeff": 0.1,
    "instruction": "Summarize in less than 50 words",
}


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as handle:
        for line in handle:
            rows.append(json.loads(line))
    return rows


def print_d(d: dict) -> None:
    for key, val in d.items():
        print(f"key:{key}\nval:{val}\n")


def build_comparison(tuned_path: Path, untuned_path: Path) -> pd.DataFrame:
    tuned = load_jsonl(tuned_path)
    untuned = load_jsonl(untuned_path)
    print("tuned sample:")
    print_d(tuned[0])
    print("untuned sample:")
    print_d(untuned[0])

    prompts = [sample["inputs"]["inputs_pretokenized"] for sample in tuned]
    untuned_completion = [sample["prediction"] for sample in untuned]
    tuned_completions = [sample["prediction"] for sample in tuned]
    return pd.DataFrame(
        {
            "prompts": prompts,
            "base_model": untuned_completion,
            "trained_model": tuned_completions,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="compare rlhf eval results")
    parser.add_argument(
        "--tuned",
        type=Path,
        default=Path("eval_results_tuned.jsonl"),
    )
    parser.add_argument(
        "--untuned",
        type=Path,
        default=Path("eval_results_untuned.jsonl"),
    )
    parser.add_argument(
        "--show-params",
        action="store_true",
        help="print default vertex rlhf parameter_values",
    )
    parser.add_argument("--out", type=Path, default=None, help="optional csv path")
    args = parser.parse_args()

    if args.show_params:
        print(json.dumps(DEFAULT_PARAMETER_VALUES, indent=2))

    missing = [p for p in (args.tuned, args.untuned) if not p.exists()]
    if missing:
        for path in missing:
            print(f"file not found: {path}")
        return

    pd.set_option("display.max_colwidth", None)
    results = build_comparison(args.tuned, args.untuned)
    print(results)
    if args.out:
        results.to_csv(args.out, index=False)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
