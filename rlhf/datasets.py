"""explore preference and prompt datasets used for rlhf."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as handle:
        for line in handle:
            rows.append(json.loads(line))
    return rows


def print_d(d: dict) -> None:
    for key, val in d.items():
        print(f"key:{key}\nval:{val}\n")


def explore_preference(path: Path, index: int) -> None:
    data = load_jsonl(path)
    print(f"loaded {len(data)} preference rows from {path}")
    sample = data[index]
    print(f"type: {type(sample)}")
    print(f"keys: {sample.keys()}")
    print(f"input_text:\n{sample.get('input_text')}\n")
    if len(data) > 2:
        print(f"sample[2] input_text tail: {data[2]['input_text'][-50:]}")
    print(f"candidate_0:\n{sample.get('candidate_0')}\n")
    print(f"candidate_1:\n{sample.get('candidate_1')}\n")
    print(f"choice: {sample.get('choice')}")


def explore_prompt(path: Path, index: int) -> None:
    data = load_jsonl(path)
    print(f"loaded {len(data)} prompt rows from {path}")
    print_d(data[index])


def main() -> None:
    parser = argparse.ArgumentParser(description="inspect rlhf preference/prompt jsonl")
    parser.add_argument(
        "--preference",
        type=Path,
        default=Path("sample_preference.jsonl"),
    )
    parser.add_argument(
        "--prompt",
        type=Path,
        default=Path("sample_prompt.jsonl"),
    )
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument(
        "--skip-preference",
        action="store_true",
        help="only inspect prompt dataset",
    )
    parser.add_argument(
        "--skip-prompt",
        action="store_true",
        help="only inspect preference dataset",
    )
    args = parser.parse_args()

    if not args.skip_preference:
        if not args.preference.exists():
            print(f"preference file not found: {args.preference}")
        else:
            explore_preference(args.preference, args.index)

    if not args.skip_prompt:
        if not args.prompt.exists():
            print(f"prompt file not found: {args.prompt}")
        else:
            explore_prompt(args.prompt, args.index)


if __name__ == "__main__":
    main()
