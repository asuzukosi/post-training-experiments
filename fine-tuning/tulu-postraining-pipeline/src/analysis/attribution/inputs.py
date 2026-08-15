"""what evaluation produced for one stage, parsed off disk."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from analysis.io import as_float, load_json_mapping

# a json file, or its already-loaded contents. only the loaders below accept this;
# everything past them holds parsed values.
JsonSource = str | Path | Mapping[str, Any]


@dataclass
class JudgedComparison:
    """this stage judged against sft, plus the style stats that came with it.

    `win_rate` is this stage's share of decisive pairs, so 0.5 is parity. the two style
    numbers describe this stage's own output — they are reported alongside the win-rate
    because length and markdown are the usual explanation for one.
    """

    win_rate: float | None = None
    mean_chars: float | None = None
    markdown_rate: float | None = None

    @classmethod
    def from_report(cls, source: JsonSource) -> JudgedComparison:
        payload = load_json_mapping(source)
        style = payload.get("style_b") or {}
        raw = payload.get("raw") or {}
        return cls(
            win_rate=as_float(raw.get("win_rate_b")),
            mean_chars=as_float(style.get("mean_chars")),
            markdown_rate=as_float(style.get("markdown_rate")),
        )


@dataclass
class StageEvaluation:
    """one training stage's evaluation results.

    the two benchmark scores come from one lm-eval result; `sft_comparison` comes
    from the judged head-to-head against sft, with this stage as model b.

    everything but `stage` is optional. base and sft have no preference comparison — a
    win-rate is pairwise and sft is the reference — and a stage that has not been
    evaluated yet has none of it.

    use `from_files` to build one from paths on disk; the fields hold parsed values, so
    nothing downstream has to guess whether it is holding a path or a dict.
    """

    stage: str
    # ifeval reports four accuracies; this is `prompt_level_strict_acc` — the share of
    # prompts where EVERY constraint was met, no formatting slack. the instruction-level
    # and loose variants score the same run higher, so do not compare across them.
    ifeval_acc: float | None = None
    mmlu_acc: float | None = None
    sft_comparison: JudgedComparison | None = None

    @property
    def has_benchmarks(self) -> bool:
        return self.ifeval_acc is not None or self.mmlu_acc is not None

    @classmethod
    def from_files(
        cls,
        stage: str,
        *,
        benchmarks: JsonSource | None = None,
        sft_comparison: JsonSource | None = None,
    ) -> StageEvaluation:
        ifeval = mmlu = None
        if benchmarks is not None:
            payload = load_json_mapping(benchmarks)
            ifeval = as_float(payload.get("ifeval_prompt_strict"))
            mmlu = as_float(payload.get("mmlu_acc"))
        return cls(
            stage=stage,
            ifeval_acc=ifeval,
            mmlu_acc=mmlu,
            sft_comparison=(
                None
                if sft_comparison is None
                else JudgedComparison.from_report(sft_comparison)
            ),
        )
