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
from data_tools.ultrafeedback import (
    build_ultrafeedback_prompt_pool,
    build_ultrafeedback_rm_subset,
    drop_contaminated_preference_rows,
    drop_contaminated_prompt_rows,
    preference_decontam_text,
)


def _pair(prompt: str, chosen: str, rejected: str, *, prompt_id: str) -> dict:
    return {
        "prompt": prompt,
        "prompt_id": prompt_id,
        "chosen": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": chosen},
        ],
        "rejected": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": rejected},
        ],
        "score_chosen": 9.0,
        "score_rejected": 3.0,
    }


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


def test_preference_decontam_covers_prompt_and_both_responses() -> None:
    """rm and dpo train on all three fields, so a hit in any one has to drop the pair."""
    eval_prompt = "name the capital city of australia and explain why briefly"
    bank = build_ngram_bank([eval_prompt])
    clean = _pair("write a haiku about rain", "drops fall", "rain", prompt_id="clean")
    rows = [
        _pair(eval_prompt, "canberra", "sydney", prompt_id="via-prompt"),
        _pair("q", eval_prompt, "b", prompt_id="via-chosen"),
        _pair("q", "a", eval_prompt, prompt_id="via-rejected"),
        clean,
    ]
    assert eval_prompt in preference_decontam_text(rows[2])
    kept = drop_contaminated_preference_rows(rows, bank)
    assert [r["prompt_id"] for r in kept] == ["clean"]


def test_prompt_decontam_ignores_responses() -> None:
    """the ppo pool keeps only the prompt, so a hit in a discarded response is not a drop."""
    eval_prompt = "describe the water cycle in three short numbered steps"
    bank = build_ngram_bank([eval_prompt])
    rows = [
        _pair(eval_prompt, "a", "b", prompt_id="via-prompt"),
        _pair("unrelated question about pasta", eval_prompt, "b", prompt_id="response-only"),
    ]
    kept = drop_contaminated_prompt_rows(rows, bank)
    assert [r["prompt_id"] for r in kept] == ["response-only"]


def test_builders_decontaminate_before_sampling() -> None:
    """a contaminated row is replaced, not left as a hole — the subset stays full size."""
    eval_prompt = "explain the difference between weather and climate for students"
    bank = build_ngram_bank([eval_prompt])
    rows = [_pair(eval_prompt, "a", "b", prompt_id="dirty")]
    rows += [_pair(f"clean question {i}", "a", "b", prompt_id=f"c{i}") for i in range(4)]

    rm = build_ultrafeedback_rm_subset(rows, num_pairs=4, seed=0, ngram_bank=bank)
    assert len(rm) == 4
    assert "dirty" not in {r["prompt_id"] for r in rm}

    ppo = build_ultrafeedback_prompt_pool(rows, num_prompts=4, seed=0, ngram_bank=bank)
    assert len(ppo) == 4
    assert "dirty" not in {r["prompt_id"] for r in ppo}


def test_builders_without_a_bank_keep_everything() -> None:
    """decontam is opt-in; --skip-decontam has to leave the old behaviour intact."""
    eval_prompt = "explain the difference between weather and climate for students"
    rows = [_pair(eval_prompt, "a", "b", prompt_id="dirty")]
    rows += [_pair(f"clean question {i}", "a", "b", prompt_id=f"c{i}") for i in range(4)]

    rm = build_ultrafeedback_rm_subset(rows, num_pairs=5, seed=0)
    assert "dirty" in {r["prompt_id"] for r in rm}
