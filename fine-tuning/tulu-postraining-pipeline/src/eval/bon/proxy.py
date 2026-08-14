"""proxy-rm scoring and best-of-n selection (argmax score, not tournament)."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from data_tools.chat import ensure_pad_token, render_chat
from eval.bon.candidates import (
    COMPLETION_KEY,
    PROMPT_ID_KEY,
    SAMPLE_IDX_KEY,
)
from eval.io import ID_KEY, PROMPT_KEY, append_jsonl, load_completed_ids, load_jsonl
from eval.judge import DEFAULT_JUDGE_BATCH_SIZE
from prepare.paths import resolve_path

PROXY_SCORE_KEY = "proxy_score"
LOGPROB_KEY = "avg_logprob"
DEFAULT_RM_MAX_LENGTH = 2048


def proxy_score(row: Mapping[str, Any]) -> float:
    if PROXY_SCORE_KEY not in row or row[PROXY_SCORE_KEY] is None:
        raise ValueError(
            f"generation row missing {PROXY_SCORE_KEY!r} "
            f"(id={row.get(ID_KEY)!r})"
        )
    value = float(row[PROXY_SCORE_KEY])
    if value != value:
        raise ValueError(
            f"generation row {PROXY_SCORE_KEY!r} is nan (id={row.get(ID_KEY)!r})"
        )
    return value


def pool_for_n(
    candidates: Sequence[Mapping[str, Any]],
    n: int,
    *,
    prompt_id: str,
) -> list[dict[str, Any]]:
    """nested slice: sample_idx 0..n-1. requires those idx to exist."""
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    by_idx = {int(c[SAMPLE_IDX_KEY]): dict(c) for c in candidates}
    missing = [i for i in range(n) if i not in by_idx]
    if missing:
        raise ValueError(
            f"prompt_id {prompt_id!r} missing sample_idx {missing} for n={n}"
        )
    return [by_idx[i] for i in range(n)]


def pick_proxy_winner(pool: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """highest proxy_score; tie -> lower sample_idx."""
    if not pool:
        raise ValueError("empty candidate pool")
    return max(
        (dict(c) for c in pool),
        key=lambda c: (proxy_score(c), -int(c[SAMPLE_IDX_KEY])),
    )


def select_top1_by_proxy(
    grouped: Mapping[str, Sequence[Mapping[str, Any]]],
    n: int,
) -> dict[str, dict[str, Any]]:
    """argmax proxy_score among the first n samples per prompt."""
    winners: dict[str, dict[str, Any]] = {}
    for prompt_id in grouped:
        pool = pool_for_n(grouped[prompt_id], n, prompt_id=prompt_id)
        winners[prompt_id] = pick_proxy_winner(pool)
    return winners


def score_with_rm(
    rows: Sequence[Mapping[str, Any]],
    *,
    rm_checkpoint: str | Path,
    max_length: int = DEFAULT_RM_MAX_LENGTH,
    batch_size: int = DEFAULT_JUDGE_BATCH_SIZE,
) -> list[float]:
    """sequence-classification scores for prompt+completion rows."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    ckpt = resolve_path(rm_checkpoint)
    if not ckpt.exists():
        raise FileNotFoundError(f"rm checkpoint not found: {ckpt}")

    tokenizer = AutoTokenizer.from_pretrained(str(ckpt), trust_remote_code=True)
    tokenizer = ensure_pad_token(tokenizer)
    model = AutoModelForSequenceClassification.from_pretrained(
        str(ckpt),
        num_labels=1,
        trust_remote_code=True,
        torch_dtype="auto",
    )
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    texts = [
        render_chat(
            [
                {"role": "user", "content": str(row[PROMPT_KEY])},
                {"role": "assistant", "content": str(row[COMPLETION_KEY])},
            ],
            tokenizer,
            add_generation_prompt=False,
        )
        for row in rows
    ]
    scores: list[float] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        encoded = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}
        with torch.no_grad():
            logits = model(**encoded).logits.squeeze(-1)
        values = logits.detach().float().reshape(-1).tolist()
        scores.extend(float(v) for v in values)
        print(
            f"rm score: {min(start + batch_size, len(texts))}/{len(texts)} "
            f"ckpt={ckpt}"
        )
    if len(scores) != len(rows):
        raise RuntimeError(
            f"rm returned {len(scores)} scores for {len(rows)} rows"
        )
    return scores


def score_proxy_incremental(
    generations_path: str | Path,
    *,
    rm_checkpoint: str | Path,
    output_path: str | Path,
    max_length: int = DEFAULT_RM_MAX_LENGTH,
    batch_size: int = DEFAULT_JUDGE_BATCH_SIZE,
) -> Path:
    """attach proxy_score; skip ids already in output_path."""
    src = resolve_path(generations_path)
    out = resolve_path(output_path)
    rows = load_jsonl(src)
    if not rows:
        raise ValueError(f"no generations in {src}")
    done = load_completed_ids(out)
    pending = [r for r in rows if str(r.get(ID_KEY)) not in done]
    print(
        f"rm score: total={len(rows)} done={len(done)} "
        f"pending={len(pending)} out={out}"
    )
    if not pending:
        return out
    already: list[Mapping[str, Any]] = []
    to_score: list[Mapping[str, Any]] = []
    for row in pending:
        if row.get(PROXY_SCORE_KEY) is not None:
            already.append(row)
        else:
            to_score.append(row)
    for row in already:
        append_jsonl(out, row)
    if to_score:
        scores = score_with_rm(
            to_score,
            rm_checkpoint=rm_checkpoint,
            max_length=max_length,
            batch_size=batch_size,
        )
        for row, score in zip(to_score, scores, strict=True):
            append_jsonl(out, {**dict(row), PROXY_SCORE_KEY: score})
    print(f"rm score done: wrote={len(pending)} path={out}")
    return out
