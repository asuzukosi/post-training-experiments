"""tulu-3 sft mixture subset builder."""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from data_tools.chat import rendered_token_length
from data_tools.decontam import DEFAULT_NGRAM_N, filter_contaminated
from data_tools.sampling import (
    largest_remainder_quotas,
    sample_from_bucket,
    shuffle_rows,
    top_up_rows,
)

# keywords -> family; first match wins (dict order = priority). matches eda notebook.
SOURCE_FAMILY_KEYWORDS: dict[tuple[str, ...], str] = {
    ("wildjailbreak", "coconot", "hard_coded"): "safety",
    ("synthetic_finalresp",): "preference_ish",
    ("math", "gsm8k", "numina"): "math",
    ("code", "evol_code"): "code",
    ("aya",): "multilingual",
}

DEFAULT_SFT_NUM_SAMPLES = 25_000
DEFAULT_SFT_SEED = 42
DEFAULT_WEAK_ASSISTANT_CHARS = 50

# per-family target fractions (independent; must sum to 1.0 when composed below).
MATH_FRAC = 0.40
CODE_FRAC = 0.25
GENERAL_IFT_FRAC = 0.18
MULTILINGUAL_FRAC = 0.08
SAFETY_FRAC = 0.05
PREFERENCE_ISH_FRAC = 0.04

DEFAULT_FAMILY_DISTRIBUTION: dict[str, float] = {
    "math": MATH_FRAC,
    "code": CODE_FRAC,
    "general_ift": GENERAL_IFT_FRAC,
    "multilingual": MULTILINGUAL_FRAC,
    "safety": SAFETY_FRAC,
    "preference_ish": PREFERENCE_ISH_FRAC,
}


def source_family(source: str) -> str:
    """map a tulu `source` string to a family bucket."""
    s = (source or "").lower()
    for keywords, family in SOURCE_FAMILY_KEYWORDS.items():
        if any(k in s for k in keywords):
            return family
    return "general_ift"


def has_empty_assistant(messages: Sequence[dict[str, Any]]) -> bool:
    """true if any assistant turn is empty / whitespace-only."""
    for m in messages or []:
        if m.get("role") == "assistant" and not (m.get("content") or "").strip():
            return True
    return False


def assistant_char_count(messages: Sequence[dict[str, Any]]) -> int:
    """total assistant content chars across turns."""
    total = 0
    for m in messages or []:
        if m.get("role") == "assistant":
            total += len(m.get("content") or "")
    return total


def messages_decontam_text(messages: Sequence[dict[str, Any]]) -> str:
    """flatten message contents for 8-gram decontam."""
    parts = []
    for m in messages or []:
        text = (m.get("content") or "").strip()
        if text:
            parts.append(text)
    return "\n".join(parts)


def drop_empty_assistant_rows(
    rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """drop rows with any empty/whitespace assistant turn."""
    return [r for r in rows if not has_empty_assistant(r.get("messages") or [])]


def drop_weak_assistant_rows(
    rows: Sequence[dict[str, Any]],
    *,
    min_chars: int = DEFAULT_WEAK_ASSISTANT_CHARS,
) -> list[dict[str, Any]]:
    """drop rows whose total assistant text is shorter than `min_chars`."""
    return [
        r for r in rows if assistant_char_count(r.get("messages") or []) >= min_chars
    ]


def drop_contaminated_message_rows(
    rows: Sequence[dict[str, Any]],
    bank: set[str],
    *,
    n: int = DEFAULT_NGRAM_N,
) -> list[dict[str, Any]]:
    """drop rows whose flattened messages overlap the n-gram bank."""
    return filter_contaminated(
        rows,
        bank,
        text_fn=lambda r: messages_decontam_text(r.get("messages") or []),
        n=n,
    )


def drop_over_max_length_rows(
    rows: Sequence[dict[str, Any]],
    tokenizer: Any,
    *,
    max_length: int = 4096,
) -> list[dict[str, Any]]:
    """drop rows whose rendered chat exceeds `max_length` tokens."""
    kept = []
    for row in rows:
        if rendered_token_length(row.get("messages") or [], tokenizer) <= max_length:
            kept.append(row)
    return kept


def filter_tulu_sft_rows(
    rows: Sequence[dict[str, Any]],
    *,
    drop_weak_assistant: bool = True,
    weak_assistant_chars: int = DEFAULT_WEAK_ASSISTANT_CHARS,
    ngram_bank: set[str] | None = None,
    tokenizer: Any | None = None,
    max_length: int = 4096,
    drop_over_max_length: bool = True,
) -> list[dict[str, Any]]:
    """apply sft prep filters in findings order (no sampling)."""
    kept = drop_empty_assistant_rows(rows)
    if drop_weak_assistant:
        kept = drop_weak_assistant_rows(kept, min_chars=weak_assistant_chars)
    if ngram_bank:
        kept = drop_contaminated_message_rows(kept, ngram_bank)
    if drop_over_max_length and tokenizer is not None:
        kept = drop_over_max_length_rows(kept, tokenizer, max_length=max_length)
    return kept


def resolve_family_weights(
    family_distribution: dict[str, float],
    family_counts: dict[str, int],
) -> dict[str, float]:
    """intersect target distribution with available families and renormalize.

    families present in the pool but missing from `family_distribution` get weight 0
    (they are not sampled unless the caller adds them to the dict).
    """
    if not family_distribution:
        raise ValueError("family_distribution must be non-empty")
    for family, weight in family_distribution.items():
        if weight < 0:
            raise ValueError(f"family_distribution[{family!r}] must be >= 0, got {weight}")
    available = {
        family: float(family_distribution[family])
        for family, count in family_counts.items()
        if count > 0 and family in family_distribution and family_distribution[family] > 0
    }
    if not available:
        raise ValueError(
            "no overlap between family_distribution and non-empty family pools; "
            f"distribution_keys={sorted(family_distribution)} "
            f"pool_keys={sorted(k for k, c in family_counts.items() if c > 0)}"
        )
    total = sum(available.values())
    if total <= 0:
        raise ValueError("family_distribution weights must sum to a positive value")
    return {family: weight / total for family, weight in available.items()}


def bucket_by_family_source(
    rows: Sequence[dict[str, Any]],
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """group rows as family -> source -> rows."""
    buckets: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for row in rows:
        family = source_family(str(row.get("source") or ""))
        source = str(row.get("source") or "unknown")
        buckets.setdefault(family, {}).setdefault(source, []).append(row)
    return buckets


def family_pool_counts(
    buckets: dict[str, dict[str, list[dict[str, Any]]]],
) -> dict[str, int]:
    """row counts per family from family->source buckets."""
    return {
        family: sum(len(v) for v in sources.values())
        for family, sources in buckets.items()
    }


def sample_stratified_by_family_source(
    buckets: dict[str, dict[str, list[dict[str, Any]]]],
    num_samples: int,
    rng: Any,
    *,
    family_distribution: dict[str, float] = DEFAULT_FAMILY_DISTRIBUTION,
) -> list[dict[str, Any]]:
    """sample with per-family quotas from `family_distribution`, then by source within family."""
    family_counts = family_pool_counts(buckets)
    family_quotas = largest_remainder_quotas(
        resolve_family_weights(family_distribution, family_counts),
        num_samples,
    )
    selected: list[dict[str, Any]] = []
    for family, fam_n in sorted(family_quotas.items()):
        sources = buckets.get(family, {})
        source_weights = {s: float(len(rows_s)) for s, rows_s in sources.items() if rows_s}
        if not source_weights:
            continue
        source_quotas = largest_remainder_quotas(source_weights, fam_n)
        for source, k in sorted(source_quotas.items()):
            selected.extend(sample_from_bucket(sources[source], k, rng))
    return selected


def build_tulu_sft_subset(
    rows: Sequence[dict[str, Any]],
    *,
    num_samples: int = DEFAULT_SFT_NUM_SAMPLES,
    seed: int = DEFAULT_SFT_SEED,
    drop_weak_assistant: bool = True,
    weak_assistant_chars: int = DEFAULT_WEAK_ASSISTANT_CHARS,
    ngram_bank: set[str] | None = None,
    family_distribution: dict[str, float] | None = None,
    tokenizer: Any | None = None,
    max_length: int = 4096,
    drop_over_max_length: bool = True,
) -> list[dict[str, Any]]:
    """build a stratified tulu sft subset (function only — no disk writes).

    `family_distribution` maps family -> fraction (e.g. DEFAULT_FAMILY_DISTRIBUTION).
    missing/empty families in the pool are skipped and remaining weights renormalized.

    """
    import numpy as np

    if num_samples < 1:
        raise ValueError(f"num_samples must be >= 1, got {num_samples}")
    dist = family_distribution if family_distribution is not None else DEFAULT_FAMILY_DISTRIBUTION

    kept = filter_tulu_sft_rows(
        rows,
        drop_weak_assistant=drop_weak_assistant,
        weak_assistant_chars=weak_assistant_chars,
        ngram_bank=ngram_bank,
        tokenizer=None,
        max_length=max_length,
        drop_over_max_length=False,
    )
    if len(kept) < num_samples:
        raise ValueError(
            f"only {len(kept)} rows left after filters, need num_samples={num_samples}"
        )

    rng = np.random.default_rng(seed)
    buckets = bucket_by_family_source(kept)
    selected = sample_stratified_by_family_source(
        buckets,
        num_samples,
        rng,
        family_distribution=dist,
    )
    selected = top_up_rows(selected, kept, num_samples, rng)
    selected = shuffle_rows(selected, rng)
    selected = selected[:num_samples]

    if drop_over_max_length and tokenizer is not None:
        before = len(selected)
        selected = drop_over_max_length_rows(selected, tokenizer, max_length=max_length)
        dropped = before - len(selected)
        if dropped:
            print(
                f"dropped {dropped} rows over max_length={max_length} "
                f"(not backfilled): {before} -> {len(selected)}"
            )
    return selected
