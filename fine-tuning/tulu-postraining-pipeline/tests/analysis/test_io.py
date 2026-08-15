"""T3 — analysis/io.py, the layer every analysis module reads and writes through.

individually low-risk, but a wrong coercion or a misplaced output path corrupts every
downstream artifact at once, and the symptom appears in the report rather than here.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from analysis.io import (
    as_float,
    load_json_list,
    load_json_mapping,
    merge_stage_map,
    parse_stage_pairs,
    resolve_output_path,
    write_json,
)


def test_as_float_preserves_none_rather_than_coercing_to_zero() -> None:
    """None means "not measured"; 0.0 means "measured as zero" — never conflate them.

    every metric in the attribution table is Optional for exactly this reason: a stage
    that has not run must not appear as a stage that scored 0.
    """
    assert as_float(None) is None
    assert as_float(0) == 0.0
    assert as_float("0.5") == pytest.approx(0.5)
    assert as_float(1) == 1.0


def test_as_float_rejects_junk_loudly() -> None:
    """a metric that silently became NaN would propagate through every mean and CI."""
    with pytest.raises(ValueError):
        as_float("not-a-number")


def test_as_float_passes_nan_through_so_callers_can_see_it() -> None:
    """documents current behaviour: NaN is NOT filtered here."""
    assert math.isnan(as_float(float("nan")))


def test_write_json_round_trips_and_creates_parents(tmp_path: Path) -> None:
    out = tmp_path / "deep" / "nested" / "metrics.json"
    written = write_json(out, {"win_rate": 0.61, "n": 3})
    assert written.is_file()
    assert json.loads(written.read_text(encoding="utf-8")) == {"win_rate": 0.61, "n": 3}
    assert written.read_text(encoding="utf-8").endswith("\n")


def test_load_json_mapping_accepts_a_mapping_or_a_path(tmp_path: Path) -> None:
    assert load_json_mapping({"a": 1}) == {"a": 1}
    p = tmp_path / "m.json"
    p.write_text(json.dumps({"b": 2}), encoding="utf-8")
    assert load_json_mapping(p) == {"b": 2}


def test_load_json_mapping_rejects_a_list_and_load_json_list_rejects_an_object(
    tmp_path: Path,
) -> None:
    """shape mismatches must raise here, not surface as a confusing error mid-analysis."""
    lst = tmp_path / "l.json"
    lst.write_text(json.dumps([1, 2]), encoding="utf-8")
    with pytest.raises(ValueError, match="expected json object"):
        load_json_mapping(lst)

    obj = tmp_path / "o.json"
    obj.write_text(json.dumps({"a": 1}), encoding="utf-8")
    with pytest.raises(ValueError, match="expected json list"):
        load_json_list(obj)


def test_resolve_output_path_uses_default_when_none_and_creates_the_parent(
    tmp_path: Path,
) -> None:
    default = tmp_path / "results" / "metrics" / "d.json"
    got = resolve_output_path(None, default=default)
    assert got == default
    assert got.parent.is_dir()

    explicit = tmp_path / "elsewhere" / "e.json"
    got2 = resolve_output_path(explicit, default=default)
    assert got2 == explicit
    assert got2.parent.is_dir()


def test_stage_pairs_parse_and_override_the_json_map(tmp_path: Path) -> None:
    """STAGE=PATH cli overrides must win over the json map, not merge ambiguously."""
    assert parse_stage_pairs(["sft=/a/b.json"]) == {"sft": "/a/b.json"}
    for bad in (["sft"], ["=x"], ["sft="]):
        with pytest.raises(ValueError, match="expected STAGE=PATH"):
            parse_stage_pairs(bad)

    m = tmp_path / "map.json"
    m.write_text(json.dumps({"sft": "from_json", "ppo": "keep"}), encoding="utf-8")
    merged = merge_stage_map(m, ["sft=from_cli"])
    assert merged == {"sft": "from_cli", "ppo": "keep"}
