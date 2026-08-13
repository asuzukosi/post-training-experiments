"""rewardbench-chat gate for the rm stage.

findings: score raw chat subsets (alpacaeval-easy/hard/length, ~2.4k);
pass if chat accuracy >= 0.65, warn if < 0.70, fail (stop) if < 0.65.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from prepare.paths import ROOT, resolve_path

# raw chat pool from findings (alpacaeval-easy/hard/length ≈ 2415 rows)
DEFAULT_CHAT_SUBSETS = (
    "alpacaeval-easy",
    "alpacaeval-hard",
    "alpacaeval-length",
)

DEFAULT_MIN_ACC = 0.65
DEFAULT_WARN_ACC = 0.70
DEFAULT_RB_DATASET = "allenai/reward-bench"
DEFAULT_RB_SPLIT = "raw"
DEFAULT_METRICS_DIR = ROOT / "results" / "metrics"


class GateStatus(str, Enum):
    """rm acceptance gate outcome."""

    PASS = "pass"  # >= warn_acc (target band)
    WARN = "warn"  # >= min_acc but < warn_acc
    FAIL = "fail"  # < min_acc — stop the pipeline


@dataclass
class GateResult:
    """rewardbench-chat gate summary."""

    chat_accuracy: float
    n: int
    status: GateStatus
    min_acc: float = DEFAULT_MIN_ACC
    warn_acc: float = DEFAULT_WARN_ACC
    subset_accuracies: dict[str, float] = field(default_factory=dict)
    subset_counts: dict[str, int] = field(default_factory=dict)
    split: str = DEFAULT_RB_SPLIT

    def raise_if_failed(self) -> None:
        """raise if status is fail (pipeline stop)."""
        if self.status is GateStatus.FAIL:
            raise RuntimeError(
                f"rewardbench-chat gate failed: acc={self.chat_accuracy:.4f} "
                f"< min_acc={self.min_acc:.2f} (n={self.n})"
            )


def decide_gate(
    chat_accuracy: float,
    *,
    min_acc: float = DEFAULT_MIN_ACC,
    warn_acc: float = DEFAULT_WARN_ACC,
) -> GateStatus:
    """map chat accuracy to pass / warn / fail."""
    if min_acc > warn_acc:
        raise ValueError(f"min_acc ({min_acc}) must be <= warn_acc ({warn_acc})")
    if chat_accuracy < min_acc:
        return GateStatus.FAIL
    if chat_accuracy < warn_acc:
        return GateStatus.WARN
    return GateStatus.PASS


def load_reward_bench_chat(
    *,
    dataset_id: str = DEFAULT_RB_DATASET,
    split: str = DEFAULT_RB_SPLIT,
    chat_subsets: tuple[str, ...] = DEFAULT_CHAT_SUBSETS,
):
    """load reward-bench rows restricted to chat subsets."""
    from datasets import load_dataset

    ds = load_dataset(dataset_id, split=split)
    subset_set = set(chat_subsets)
    if "subset" not in ds.column_names:
        raise ValueError(f"{dataset_id}:{split} missing 'subset' column")
    kept = ds.filter(lambda row: row["subset"] in subset_set)
    print(
        f"reward-bench {split}: kept {len(kept)}/{len(ds)} chat rows "
        f"subsets={sorted(subset_set)}"
    )
    return kept


def _render_pair(
    tokenizer: Any,
    prompt: str,
    response: str,
) -> str:
    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )


def _score_text(model: Any, tokenizer: Any, text: str, max_length: int) -> float:
    import torch

    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )
    encoded = {k: v.to(model.device) for k, v in encoded.items()}
    with torch.no_grad():
        out = model(**encoded)
    # sequence classification scalar (num_labels=1)
    logits = out.logits.squeeze()
    return float(logits.reshape(-1)[0].item())


def score_reward_bench_chat(
    rm_checkpoint: str | Path,
    *,
    dataset_id: str = DEFAULT_RB_DATASET,
    split: str = DEFAULT_RB_SPLIT,
    chat_subsets: tuple[str, ...] = DEFAULT_CHAT_SUBSETS,
    max_length: int = 2048,
    batch_log_every: int = 200,
) -> tuple[float, int, dict[str, float], dict[str, int]]:
    """score rm pairwise accuracy on rewardbench chat subsets.

    returns (chat_accuracy, n, subset_accuracies, subset_counts).
    """
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    ckpt = resolve_path(rm_checkpoint)
    if not ckpt.exists():
        raise FileNotFoundError(f"rm checkpoint not found: {ckpt}")

    ds = load_reward_bench_chat(
        dataset_id=dataset_id,
        split=split,
        chat_subsets=chat_subsets,
    )
    if len(ds) == 0:
        raise ValueError("no reward-bench chat rows after subset filter")

    print(f"loading rm for gate: {ckpt}")
    tokenizer = AutoTokenizer.from_pretrained(str(ckpt), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForSequenceClassification.from_pretrained(
        str(ckpt),
        num_labels=1,
        trust_remote_code=True,
        torch_dtype="auto",
    )
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    correct = 0
    subset_correct: dict[str, int] = {}
    subset_total: dict[str, int] = {}

    for i, row in enumerate(ds):
        prompt = row["prompt"] or ""
        chosen = row["chosen"] or ""
        rejected = row["rejected"] or ""
        subset = str(row.get("subset") or "unknown")

        chosen_text = _render_pair(tokenizer, prompt, chosen)
        rejected_text = _render_pair(tokenizer, prompt, rejected)
        r_chosen = _score_text(model, tokenizer, chosen_text, max_length)
        r_rejected = _score_text(model, tokenizer, rejected_text, max_length)
        hit = r_chosen > r_rejected
        correct += int(hit)
        subset_total[subset] = subset_total.get(subset, 0) + 1
        subset_correct[subset] = subset_correct.get(subset, 0) + int(hit)

        if batch_log_every and (i + 1) % batch_log_every == 0:
            print(f"reward-bench scored {i + 1}/{len(ds)}")

    n = len(ds)
    acc = correct / n
    subset_acc = {
        s: subset_correct[s] / subset_total[s] for s in sorted(subset_total)
    }
    print(f"reward-bench chat accuracy={acc:.4f} ({correct}/{n})")
    for s, a in subset_acc.items():
        print(f"  subset {s}: {a:.4f} ({subset_correct[s]}/{subset_total[s]})")
    return acc, n, subset_acc, subset_total


def run_reward_bench_gate(
    rm_checkpoint: str | Path,
    *,
    min_acc: float = DEFAULT_MIN_ACC,
    warn_acc: float = DEFAULT_WARN_ACC,
    dataset_id: str = DEFAULT_RB_DATASET,
    split: str = DEFAULT_RB_SPLIT,
    chat_subsets: tuple[str, ...] = DEFAULT_CHAT_SUBSETS,
    max_length: int = 2048,
    metrics_path: str | Path | None = None,
    raise_on_fail: bool = True,
) -> GateResult:
    """score rewardbench-chat and apply the pass/warn/fail gate."""
    acc, n, subset_acc, subset_counts = score_reward_bench_chat(
        rm_checkpoint,
        dataset_id=dataset_id,
        split=split,
        chat_subsets=chat_subsets,
        max_length=max_length,
    )
    status = decide_gate(acc, min_acc=min_acc, warn_acc=warn_acc)
    result = GateResult(
        chat_accuracy=acc,
        n=n,
        status=status,
        min_acc=min_acc,
        warn_acc=warn_acc,
        subset_accuracies=subset_acc,
        subset_counts=subset_counts,
        split=split,
    )
    print(
        f"rewardbench-chat gate: status={status.value} "
        f"acc={acc:.4f} min={min_acc:.2f} warn={warn_acc:.2f}"
    )

    out_path = (
        resolve_path(metrics_path)
        if metrics_path is not None
        else DEFAULT_METRICS_DIR / "rb_gate_chat.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(result)
    payload["status"] = result.status.value
    payload["rm_checkpoint"] = str(resolve_path(rm_checkpoint))
    with out_path.open("w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    print(f"wrote gate metrics -> {out_path}")

    if raise_on_fail:
        result.raise_if_failed()
    return result
