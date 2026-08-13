"""best-of-n tournament on the judgearena pairwise backend.

single-elim to top-1 per prompt (~n-1 comparisons). pairs are sorted by
completion length before each judge batch to cut vllm padding. ties keep the
shorter completion, then the lower sample_idx. selection judge defaults to
14b; reserve 32b for final head-to-heads.
"""
from eval.bon.candidates import group_candidates
from eval.bon.proxy import (
    PROXY_SCORE_KEY,
    score_proxy_incremental,
    select_top1_by_proxy,
)
from eval.bon.select import build_rs_sft_row, run_bon_selection
from eval.bon.sweep import DEFAULT_N_VALUES, run_bon_sweep
from eval.bon.tournament import (
    DEFAULT_BON_JUDGE_MODEL,
    apply_round_results,
    build_round_pairs,
    pair_id,
    pick_winner_candidate,
    select_top1,
    sort_pairs_by_length,
)

__all__ = [
    "DEFAULT_BON_JUDGE_MODEL",
    "DEFAULT_N_VALUES",
    "PROXY_SCORE_KEY",
    "apply_round_results",
    "build_round_pairs",
    "build_rs_sft_row",
    "group_candidates",
    "pair_id",
    "pick_winner_candidate",
    "run_bon_selection",
    "run_bon_sweep",
    "score_proxy_incremental",
    "select_top1",
    "select_top1_by_proxy",
    "sort_pairs_by_length",
]
