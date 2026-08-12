"""8-gram decontamination bank and filters."""
from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence

DEFAULT_NGRAM_N = 8


def normalize_text(text: str) -> str:
    """lowercase + collapse whitespace (same as eda ngram bank)."""
    return " ".join((text or "").lower().split())


def iter_ngrams(text: str, n: int = DEFAULT_NGRAM_N) -> Iterable[str]:
    """yield whitespace `n`-grams from normalized text."""
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    toks = normalize_text(text).split()
    if len(toks) < n:
        return
    for i in range(len(toks) - n + 1):
        yield " ".join(toks[i : i + n])


def build_ngram_bank(
    texts: Iterable[str],
    n: int = DEFAULT_NGRAM_N,
) -> set[str]:
    """build unique `n`-gram set from eval (or other) texts."""
    bank: set[str] = set()
    for text in texts:
        bank.update(iter_ngrams(text, n=n))
    return bank


def text_overlaps_bank(
    text: str,
    bank: set[str],
    n: int = DEFAULT_NGRAM_N,
) -> bool:
    """true if any `n`-gram in `text` is in `bank`."""
    if not bank:
        return False
    for gram in iter_ngrams(text, n=n):
        if gram in bank:
            return True
    return False


def filter_contaminated(
    rows: Sequence,
    bank: set[str],
    *,
    text_fn: Callable[[object], str],
    n: int = DEFAULT_NGRAM_N,
) -> list:
    """drop rows whose extracted text overlaps the n-gram bank.

    `text_fn` pulls the decontam string from a row (e.g. prompt / messages dump).
    """
    kept = []
    for row in rows:
        if not text_overlaps_bank(text_fn(row), bank, n=n):
            kept.append(row)
    return kept
