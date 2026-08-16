"""sycophancy flip rate: does the model abandon a correct answer when pushed back on?

    probes.py    the questions, from `meg-tong/sycophancy-eval`, and collecting the
                 answer-then-rebuttal pair for each one
    scoring.py   which option the model picked, and whether it changed its mind
"""
from eval.steer.flip_rate.probes import (
    ANSWER_KEY,
    CONTAMINATED_SOURCE,
    DEFAULT_PUSHBACK,
    DEFAULT_REPEATS,
    DEFAULT_TEMPERATURE,
    FIRST_KEY,
    PUSHBACK_KEY,
    SECOND_KEY,
    SYCOPHANCY_EVAL_CACHE,
    SYCOPHANCY_EVAL_URL,
    followup_prompt,
    load_flip_probes,
    load_sycophancy_eval,
    require_probe,
    run_flip_trials,
    to_probe,
    usable_probe,
)
from eval.steer.flip_rate.scoring import (
    FLIPPED,
    HELD,
    INELIGIBLE,
    UNSCORABLE,
    FlipRateSummary,
    chosen_letter,
    classify_trial,
    score_flip_rate,
)

__all__ = [
    "ANSWER_KEY",
    "CONTAMINATED_SOURCE",
    "DEFAULT_PUSHBACK",
    "DEFAULT_REPEATS",
    "DEFAULT_TEMPERATURE",
    "FIRST_KEY",
    "FLIPPED",
    "FlipRateSummary",
    "HELD",
    "INELIGIBLE",
    "PUSHBACK_KEY",
    "SECOND_KEY",
    "SYCOPHANCY_EVAL_CACHE",
    "SYCOPHANCY_EVAL_URL",
    "UNSCORABLE",
    "chosen_letter",
    "classify_trial",
    "followup_prompt",
    "load_flip_probes",
    "load_sycophancy_eval",
    "require_probe",
    "run_flip_trials",
    "score_flip_rate",
    "to_probe",
    "usable_probe",
]
