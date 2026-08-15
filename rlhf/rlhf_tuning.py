"""compile and optionally run the vertex ai rlhf pipeline."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

DEFAULT_PKG = Path("rlhf_pipeline.yaml")

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


def train_steps(dataset_size: int, batch_size: int, epochs: int) -> int:
    steps_per_epoch = math.ceil(dataset_size / batch_size)
    return steps_per_epoch * epochs


def compile_pipeline(package_path: Path) -> None:
    from google_cloud_pipeline_components.preview.llm import rlhf_pipeline
    from kfp import compiler

    compiler.Compiler().compile(
        pipeline_func=rlhf_pipeline,
        package_path=str(package_path),
    )
    print(f"wrote {package_path}")
    with package_path.open() as handle:
        for i, line in enumerate(handle):
            if i >= 20:
                break
            print(line.rstrip())


def run_pipeline(
    package_path: Path,
    parameter_values: dict,
    display_name: str,
    region: str,
) -> None:
    import google.cloud.aiplatform as aiplatform

    project_id = os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("PROJECT_ID")
    staging_bucket = os.environ.get("STAGING_BUCKET")
    if not project_id or not staging_bucket:
        raise SystemExit(
            "set GOOGLE_CLOUD_PROJECT (or PROJECT_ID) and STAGING_BUCKET to run"
        )

    aiplatform.init(project=project_id, location=region)
    job = aiplatform.PipelineJob(
        display_name=display_name,
        pipeline_root=staging_bucket,
        template_path=str(package_path),
        parameter_values=parameter_values,
    )
    job.run()


def main() -> None:
    parser = argparse.ArgumentParser(description="vertex ai rlhf pipeline helpers")
    sub = parser.add_subparsers(dest="command", required=True)

    calc = sub.add_parser("calc-steps", help="compute reward/rl train steps")
    calc.add_argument("--pref-size", type=int, default=3000)
    calc.add_argument("--prompt-size", type=int, default=2000)
    calc.add_argument("--batch-size", type=int, default=64)
    calc.add_argument("--reward-epochs", type=int, default=30)
    calc.add_argument("--rl-epochs", type=int, default=10)

    compile_p = sub.add_parser("compile", help="compile rlhf_pipeline.yaml")
    compile_p.add_argument("--out", type=Path, default=DEFAULT_PKG)

    run_p = sub.add_parser("run", help="submit pipeline job to vertex ai")
    run_p.add_argument("--package", type=Path, default=DEFAULT_PKG)
    run_p.add_argument("--display-name", default="tutorial-rlhf-tuning")
    run_p.add_argument("--region", default="europe-west4")
    run_p.add_argument("--pref-size", type=int, default=3000)
    run_p.add_argument("--prompt-size", type=int, default=2000)
    run_p.add_argument("--batch-size", type=int, default=64)
    run_p.add_argument("--reward-epochs", type=int, default=30)
    run_p.add_argument("--rl-epochs", type=int, default=10)
    run_p.add_argument("--kl-coeff", type=float, default=0.1)
    run_p.add_argument("--instruction", default="Summarize in less than 50 words")
    run_p.add_argument("--model", default="llama-2-7b")
    run_p.add_argument(
        "--preference-dataset",
        default=DEFAULT_PARAMETER_VALUES["preference_dataset"],
    )
    run_p.add_argument(
        "--prompt-dataset",
        default=DEFAULT_PARAMETER_VALUES["prompt_dataset"],
    )
    run_p.add_argument(
        "--eval-dataset",
        default=DEFAULT_PARAMETER_VALUES["eval_dataset"],
    )

    args = parser.parse_args()

    if args.command == "calc-steps":
        reward_steps = train_steps(args.pref_size, args.batch_size, args.reward_epochs)
        rl_steps = train_steps(args.prompt_size, args.batch_size, args.rl_epochs)
        print(f"reward steps/epoch: {math.ceil(args.pref_size / args.batch_size)}")
        print(f"reward_model_train_steps: {reward_steps}")
        print(f"rl steps/epoch: {math.ceil(args.prompt_size / args.batch_size)}")
        print(f"reinforcement_learning_train_steps: {rl_steps}")
        return

    if args.command == "compile":
        compile_pipeline(args.out)
        return

    parameter_values = {
        "preference_dataset": args.preference_dataset,
        "prompt_dataset": args.prompt_dataset,
        "eval_dataset": args.eval_dataset,
        "large_model_reference": args.model,
        "reward_model_train_steps": train_steps(
            args.pref_size, args.batch_size, args.reward_epochs
        ),
        "reinforcement_learning_train_steps": train_steps(
            args.prompt_size, args.batch_size, args.rl_epochs
        ),
        "reward_model_learning_rate_multiplier": 1.0,
        "reinforcement_learning_rate_multiplier": 1.0,
        "kl_coeff": args.kl_coeff,
        "instruction": args.instruction,
    }
    print("parameter_values:", parameter_values)
    run_pipeline(args.package, parameter_values, args.display_name, args.region)


if __name__ == "__main__":
    main()
