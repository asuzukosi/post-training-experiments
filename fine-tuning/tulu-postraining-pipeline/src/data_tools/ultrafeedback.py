"""ultrafeedback preference-pair subset builders"""
from __future__ import annotations
from collections.abc import Iterable, Sequence
from typing import Any

from data_tools.sampling import sample_from_bucket, shuffle_rows

DEFAULT_RM_NUM_PAIRS = 20_000
DEFAULT_RM_SEED = 42

DEFAULT_DPO_NUM_PAIRS = 10_000
DEFAULT_DPO_SEED = 42
DEFAULT_DPO_MIN_MARGIN = 0.0

DEFAULT_PPO_NUM_PROMPTS = 1_500
DEFAULT_PPO_SEED = 42


def last_assistant_content(messages: Sequence[dict[str, Any]]) -> str:
    """content of the last assistant turn (empty string if none)."""
    for m in reversed(messages or []):
        if m.get("role") == "assistant":
            return m.get("content") or ""
    return ""


def has_empty_preference_side(row: dict[str, Any]) -> bool:
    """true if chosen or rejected last-assistant content is empty / whitespace."""
    chosen = last_assistant_content(row.get("chosen") or [])
    rejected = last_assistant_content(row.get("rejected") or [])
    return (not chosen.strip()) or (not rejected.strip())


def drop_empty_preference_rows(
    rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """drop pairs with an empty chosen or rejected assistant side."""
    return [r for r in rows if not has_empty_preference_side(r)]


def preference_score_margin(row: dict[str, Any]) -> float | None:
    """score_chosen - score_rejected, or none if either score is missing."""
    sc = row.get("score_chosen")
    sr = row.get("score_rejected")
    if sc is None or sr is None:
        return None
    return float(sc) - float(sr)


def drop_low_margin_preference_rows(
    rows: Sequence[dict[str, Any]],
    *,
    min_margin: float,
) -> list[dict[str, Any]]:
    """drop pairs with missing margin or margin <= `min_margin`."""
    kept = []
    for row in rows:
        margin = preference_score_margin(row)
        if margin is not None and margin > min_margin:
            kept.append(row)
    return kept


def filter_ultrafeedback_preference_rows(
    rows: Sequence[dict[str, Any]],
    *,
    min_margin: float | None = None,
) -> list[dict[str, Any]]:
    """apply ultrafeedback prep filters (no sampling).

    rm: empty-side drop only (`min_margin=None`).
    dpo: optionally also drop weak margins (`min_margin=0` or `0.5`).
    """
    kept = drop_empty_preference_rows(rows)
    if min_margin is not None:
        kept = drop_low_margin_preference_rows(kept, min_margin=min_margin)
    return kept


def prompt_ids_of(rows: Sequence[dict[str, Any]]) -> set[str]:
    """set of prompt_id strings from preference rows."""
    return {str(r.get("prompt_id") or "") for r in rows}


def one_row_per_prompt_id(
    rows: Sequence[dict[str, Any]],
    rng: Any,
) -> list[dict[str, Any]]:
    """keep one row per prompt_id (random among duplicates)."""
    by_id: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        pid = str(row.get("prompt_id") or "")
        by_id.setdefault(pid, []).append(row)
    out: list[dict[str, Any]] = []
    for pid in sorted(by_id):
        group = by_id[pid]
        if len(group) == 1:
            out.append(group[0])
        else:
            out.append(group[int(rng.integers(0, len(group)))])
    return out


def sample_preference_pairs(
    rows: Sequence[dict[str, Any]],
    num_pairs: int,
    rng: Any,
) -> list[dict[str, Any]]:
    """sample `num_pairs` rows without replacement (caller should dedupe ids first)."""
    return sample_from_bucket(rows, num_pairs, rng)


def build_ultrafeedback_rm_subset(
    rows: Sequence[dict[str, Any]],
    *,
    num_pairs: int = DEFAULT_RM_NUM_PAIRS,
    seed: int = DEFAULT_RM_SEED,
) -> list[dict[str, Any]]:
    """build an ultrafeedback rm subset from train_prefs (function only — no disk writes).

    drops empty chosen/rejected sides, collapses duplicate prompt_ids to one row,
    then samples `num_pairs`. no margin filter (bt still uses low-margin pairs).
    """
    import numpy as np

    if num_pairs < 1:
        raise ValueError(f"num_pairs must be >= 1, got {num_pairs}")

    kept = filter_ultrafeedback_preference_rows(rows, min_margin=None)
    rng = np.random.default_rng(seed)
    unique = one_row_per_prompt_id(kept, rng)
    if len(unique) < num_pairs:
        raise ValueError(
            f"only {len(unique)} unique prompt_ids after filters, need num_pairs={num_pairs}"
        )

    selected = sample_preference_pairs(unique, num_pairs, rng)
    selected = shuffle_rows(selected, rng)
    return selected


def drop_excluded_prompt_ids(
    rows: Sequence[dict[str, Any]],
    exclude_prompt_ids: Iterable[str],
) -> list[dict[str, Any]]:
    """drop rows whose prompt_id is in `exclude_prompt_ids`."""
    exclude = {str(pid) for pid in exclude_prompt_ids}
    if not exclude:
        return list(rows)
    return [r for r in rows if str(r.get("prompt_id") or "") not in exclude]


def assert_disjoint_prompt_ids(
    left: Sequence[dict[str, Any]],
    right: Sequence[dict[str, Any]],
) -> None:
    """raise if `left` and `right` share any prompt_id."""
    overlap = prompt_ids_of(left) & prompt_ids_of(right)
    if overlap:
        sample = sorted(overlap)[:3]
        raise ValueError(
            f"prompt_id overlap between sets ({len(overlap)} ids); examples={sample}"
        )


def build_ultrafeedback_dpo_subset(
    rows: Sequence[dict[str, Any]],
    *,
    exclude_prompt_ids: Iterable[str] | None = None,
    num_pairs: int = DEFAULT_DPO_NUM_PAIRS,
    seed: int = DEFAULT_DPO_SEED,
    min_margin: float | None = DEFAULT_DPO_MIN_MARGIN,
) -> list[dict[str, Any]]:
    """build an ultrafeedback dpo subset disjoint from rm by prompt_id.

    `exclude_prompt_ids` should be the rm subset's prompt_ids. drops empty sides,
    optionally drops weak margins (`min_margin=0` keeps margin > 0), dedupes
    prompt_ids, then samples. asserts empty intersection with the exclude set.
    """
    import numpy as np

    if num_pairs < 1:
        raise ValueError(f"num_pairs must be >= 1, got {num_pairs}")

    exclude = {str(pid) for pid in (exclude_prompt_ids or [])}
    kept = filter_ultrafeedback_preference_rows(rows, min_margin=min_margin)
    kept = drop_excluded_prompt_ids(kept, exclude)

    rng = np.random.default_rng(seed)
    unique = one_row_per_prompt_id(kept, rng)
    if len(unique) < num_pairs:
        raise ValueError(
            f"only {len(unique)} unique prompt_ids after filters/exclusions, "
            f"need num_pairs={num_pairs}"
        )

    selected = sample_preference_pairs(unique, num_pairs, rng)
    selected = shuffle_rows(selected, rng)

    overlap = prompt_ids_of(selected) & exclude
    if overlap:
        sample = sorted(overlap)[:3]
        raise ValueError(
            f"dpo prompt_id overlap with exclude set ({len(overlap)} ids); "
            f"examples={sample}"
        )
    return selected


def preference_row_to_ppo_prompt(row: dict[str, Any]) -> dict[str, str]:
    """keep prompt text + id only (labels unused at ppo train time)."""
    return {
        "prompt": str(row.get("prompt") or ""),
        "prompt_id": str(row.get("prompt_id") or ""),
    }


def drop_empty_prompt_rows(
    rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """drop rows with empty / whitespace-only prompt text."""
    return [r for r in rows if (r.get("prompt") or "").strip()]


def build_ultrafeedback_ppo_prompts(
    rows: Sequence[dict[str, Any]],
    *,
    num_prompts: int = DEFAULT_PPO_NUM_PROMPTS,
    seed: int = DEFAULT_PPO_SEED,
) -> list[dict[str, str]]:
    """build ppo prompt pool from test_prefs (function only — no disk writes).

    drops empty prompts, collapses duplicate prompt_ids, samples `num_prompts`,
    returns `{prompt, prompt_id}` rows (no chosen/rejected).
    """
    import numpy as np

    if num_prompts < 1:
        raise ValueError(f"num_prompts must be >= 1, got {num_prompts}")

    kept = drop_empty_prompt_rows(rows)
    rng = np.random.default_rng(seed)
    unique = one_row_per_prompt_id(kept, rng)
    if len(unique) < num_prompts:
        raise ValueError(
            f"only {len(unique)} unique prompt_ids after filters, "
            f"need num_prompts={num_prompts}"
        )

    selected = sample_from_bucket(unique, num_prompts, rng)
    selected = shuffle_rows(selected, rng)
    return [preference_row_to_ppo_prompt(r) for r in selected]
