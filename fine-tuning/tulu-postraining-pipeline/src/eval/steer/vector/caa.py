"""the contrastive activation addition sycophancy set — what the vector is trained on.

1,000 bio-plus-opinion prompts with the sycophantic answer labelled. this is what the
`steering-vectors` library trains on in its own sycophancy example, so the pairing is
exercised upstream rather than only by us.
"""
from __future__ import annotations

import json
import urllib.request
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from prepare.paths import resolve_path

# the contrastive activation addition sycophancy set — 1,000 bio-plus-opinion prompts
# with the sycophantic answer labelled. this is what the steering-vectors library trains
# on in its own sycophancy example, so the pairing is exercised upstream.
CAA_SYCOPHANCY_URL = (
    "https://raw.githubusercontent.com/nrimsky/CAA/main/"
    "datasets/generate/sycophancy/generate_dataset.json"
)
CAA_CACHE = "data/raw/caa_sycophancy_generate.json"
MATCHING_KEY = "answer_matching_behavior"
NOT_MATCHING_KEY = "answer_not_matching_behavior"


def require_caa_row(row: Any, *, index: int | None = None) -> dict[str, str]:
    """a caa row: one prompt plus the two answers that differ only in the trait."""
    loc = f" at row {index}" if index is not None else ""
    for key in ("question", MATCHING_KEY, NOT_MATCHING_KEY):
        val = row.get(key)
        if not isinstance(val, str) or not val.strip():
            raise ValueError(f"caa row {key!r} must be a non-empty str{loc}")
    if row[MATCHING_KEY].strip() == row[NOT_MATCHING_KEY].strip():
        raise ValueError(f"caa row has identical answers{loc}; no contrast to learn")
    return {
        "question": row["question"],
        MATCHING_KEY: row[MATCHING_KEY],
        NOT_MATCHING_KEY: row[NOT_MATCHING_KEY],
    }


def to_training_samples(rows: Sequence[Any]) -> list[tuple[str, str]]:
    """(sycophantic, non-sycophantic) prompt pairs for `train_steering_vector`.

    each pair is the SAME question with a different answer appended, so the two prompts
    differ only in the final token. that is the whole point of the caa construction:
    hand-written pairs of two different sentences also differ in topic, wording and
    length, and the vector absorbs all of it.

    DIRECTION: positive is the sycophantic answer, so +alpha steers TOWARDS sycophancy
    and -alpha away from it. the experiment asks whether negative alpha reduces the flip
    rate; getting this backwards would look exactly like steering that does not work.
    """
    out: list[tuple[str, str]] = []
    for i, row in enumerate(rows, start=1):
        r = require_caa_row(row, index=i)
        q = r["question"].rstrip()
        out.append((f"{q}\n{r[MATCHING_KEY].strip()}", f"{q}\n{r[NOT_MATCHING_KEY].strip()}"))
    return out


def load_caa_sycophancy(path: str | Path | None = None) -> list[dict[str, str]]:
    """load the caa sycophancy set, downloading and caching it on first use."""
    import urllib.request

    target = resolve_path(path or CAA_CACHE)
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading caa sycophancy set -> {target}")
        urllib.request.urlretrieve(CAA_SYCOPHANCY_URL, target)
    rows = json.loads(target.read_text(encoding="utf-8"))
    if not rows:
        raise ValueError(f"no caa rows in {target}")
    return [require_caa_row(r, index=i) for i, r in enumerate(rows, start=1)]
