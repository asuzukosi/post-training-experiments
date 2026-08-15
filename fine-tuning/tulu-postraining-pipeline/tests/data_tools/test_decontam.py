"""unit tests for 8-gram decontam helpers (no network / no gpu).

asserts the scan actually drops overlapping train rows and keeps clean ones.
"""
from __future__ import annotations

import pytest

from data_tools.decontam import (
    DEFAULT_NGRAM_N,
    build_ngram_bank,
    filter_contaminated,
    iter_ngrams,
    normalize_text,
    text_overlaps_bank,
)
from data_tools.tulu_sft import drop_contaminated_message_rows, messages_decontam_text


def test_build_ngram_bank_unique() -> None:
    bank = build_ngram_bank(
        [
            "alpha beta gamma delta epsilon zeta eta theta",
            "alpha beta gamma delta epsilon zeta eta theta iota",
        ]
    )
    assert "alpha beta gamma delta epsilon zeta eta theta" in bank
    assert "beta gamma delta epsilon zeta eta theta iota" in bank
    assert len(bank) == 2


def test_text_overlaps_bank_true_and_false() -> None:
    eval_prompt = "the quick brown fox jumps over the lazy dog today"
    bank = build_ngram_bank([eval_prompt])
    contaminated = "prefix the quick brown fox jumps over the lazy dog today suffix"
    clean = "completely unrelated words about cooking pasta with garlic oil herbs"
    assert text_overlaps_bank(contaminated, bank) is True
    assert text_overlaps_bank(clean, bank) is False
    assert text_overlaps_bank(clean, set()) is False


def test_filter_contaminated_removes_overlap_keeps_clean() -> None:
    eval_prompt = "please solve this math problem about primes carefully now"
    bank = build_ngram_bank([eval_prompt])
    rows = [
        {"id": "bad", "text": f"user said: {eval_prompt}"},
        {"id": "good", "text": "write a short poem about autumn leaves falling slowly"},
    ]
    kept = filter_contaminated(rows, bank, text_fn=lambda r: r["text"])
    assert [r["id"] for r in kept] == ["good"]


def test_drop_contaminated_message_rows() -> None:
    eval_prompt = "list three benefits of daily walking for heart health please"
    bank = build_ngram_bank([eval_prompt])
    dirty = {
        "id": "dirty",
        "messages": [
            {"role": "user", "content": eval_prompt},
            {"role": "assistant", "content": "1. cardio 2. mood 3. weight"},
        ],
    }
    clean = {
        "id": "clean",
        "messages": [
            {"role": "user", "content": "explain how photosynthesis converts light energy"},
            {"role": "assistant", "content": "chloroplasts capture photons and fix carbon."},
        ],
    }
    assert eval_prompt in messages_decontam_text(dirty["messages"])
    kept = drop_contaminated_message_rows([dirty, clean], bank)
    assert [r["id"] for r in kept] == ["clean"]
