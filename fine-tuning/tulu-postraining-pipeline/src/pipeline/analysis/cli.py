"""cli for stage attribution, beta/displacement, chattiness, dpo-vs-ppo verdict."""
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from pipeline.analysis.attribution import (
    build_stage_attribution_table,
    write_stage_attribution_table,
)
from pipeline.analysis.beta_plots import (
    BetaArmPoint,
    DisplacementSeries,
    detect_displacement,
    plot_beta_vs_kl_winrate,
    plot_displacement,
    plot_displacement_arms,
)
from pipeline.analysis.chattiness import (
    chattiness_point_from_style_report,
    plot_length_markdown,
    plot_raw_vs_length_controlled,
    summarize_chattiness,
)
from pipeline.analysis.io import (
    DEFAULT_METRICS_DIR,
    DEFAULT_PLOTS_DIR,
    load_json_mapping,
    write_json,
)
from pipeline.analysis.verdict import (
    arm_from_head_to_head_summary,
    build_dpo_ppo_verdict,
    write_dpo_ppo_verdict,
)
from pipeline.prepare.paths import resolve_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "run analysis reports: attribution, beta/displacement plots, "
            "chattiness, or dpo-vs-ppo verdict"
        )
    )

    modes = p.add_argument_group("modes (pick one or more)")
    modes.add_argument(
        "--attribution",
        action="store_true",
        help="build stage-attribution table from skills (+ optional style) maps",
    )
    modes.add_argument(
        "--beta-plots",
        action="store_true",
        help="plot beta vs kl/win-rate from --beta-arms-json",
    )
    modes.add_argument(
        "--displacement",
        action="store_true",
        help="plot + detect preference displacement from --displacement-json",
    )
    modes.add_argument(
        "--chattiness",
        action="store_true",
        help="plot raw vs length-controlled + length/markdown from style map",
    )
    modes.add_argument(
        "--verdict",
        action="store_true",
        help="write dpo-vs-ppo equal-data verdict from vs-sft summaries",
    )

    p.add_argument(
        "--skills-json",
        type=Path,
        default=None,
        help='json map {"base": "skills.json", "sft": "...", ...} for --attribution',
    )
    p.add_argument(
        "--skills",
        action="append",
        default=[],
        metavar="STAGE=PATH",
        help="repeatable stage=skills.json override/addition for --attribution",
    )
    p.add_argument(
        "--style-json",
        type=Path,
        default=None,
        help='json map {"dpo-b0.05": "style.json", ...} for attribution/chattiness',
    )
    p.add_argument(
        "--style",
        action="append",
        default=[],
        metavar="STAGE=PATH",
        help="repeatable stage=style.json for attribution/chattiness",
    )
    p.add_argument(
        "--beta-arms-json",
        type=Path,
        default=None,
        help="json list of {beta, kl?, win_rate_vs_sft?, win_rate_vs_sft_lc?}",
    )
    p.add_argument(
        "--displacement-json",
        type=Path,
        default=None,
        help="json list of {beta, steps, chosen_logps, rejected_logps}",
    )
    p.add_argument(
        "--dpo-summary",
        type=Path,
        default=None,
        help="head-to-head summary json (sft vs dpo) for --verdict",
    )
    p.add_argument(
        "--ppo-summary",
        type=Path,
        default=None,
        help="head-to-head summary json (sft vs ppo) for --verdict",
    )
    p.add_argument(
        "--dpo-name",
        type=str,
        default="dpo",
        help="label for dpo arm in verdict (e.g. dpo-b0.1)",
    )
    p.add_argument(
        "--ppo-name",
        type=str,
        default="ppo",
        help="label for ppo arm in verdict",
    )
    p.add_argument(
        "--metrics-dir",
        type=Path,
        default=None,
        help=f"metrics output dir (default: {DEFAULT_METRICS_DIR})",
    )
    p.add_argument(
        "--plots-dir",
        type=Path,
        default=None,
        help=f"plots output dir (default: {DEFAULT_PLOTS_DIR})",
    )
    p.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="for --attribution: write table even if stages are missing",
    )
    return p.parse_args(argv)


def _selected_modes(args: argparse.Namespace) -> list[str]:
    modes: list[str] = []
    if args.attribution:
        modes.append("attribution")
    if args.beta_plots:
        modes.append("beta_plots")
    if args.displacement:
        modes.append("displacement")
    if args.chattiness:
        modes.append("chattiness")
    if args.verdict:
        modes.append("verdict")
    return modes


def _parse_stage_pairs(items: Sequence[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"expected STAGE=PATH, got {item!r}")
        stage, path = item.split("=", 1)
        stage = stage.strip()
        path = path.strip()
        if not stage or not path:
            raise ValueError(f"expected STAGE=PATH, got {item!r}")
        out[stage] = path
    return out


def _merge_stage_map(
    json_path: Path | None,
    pairs: Sequence[str],
) -> dict[str, str]:
    out: dict[str, str] = {}
    if json_path is not None:
        payload = load_json_mapping(json_path)
        for stage, path in payload.items():
            out[str(stage)] = str(path)
    out.update(_parse_stage_pairs(pairs))
    return out


def _load_json_list(path: Path) -> list[Any]:
    resolved = resolve_path(path)
    with resolved.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"expected json list at {resolved}")
    return data


def _beta_arms_from_json(path: Path) -> list[BetaArmPoint]:
    rows = _load_json_list(path)
    arms: list[BetaArmPoint] = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError(f"beta arm must be an object, got {row!r}")
        if "beta" not in row:
            raise ValueError(f"beta arm missing beta: {row!r}")
        arms.append(
            BetaArmPoint(
                beta=float(row["beta"]),
                kl=None if row.get("kl") is None else float(row["kl"]),
                win_rate_vs_sft=(
                    None
                    if row.get("win_rate_vs_sft") is None
                    else float(row["win_rate_vs_sft"])
                ),
                win_rate_vs_sft_lc=(
                    None
                    if row.get("win_rate_vs_sft_lc") is None
                    else float(row["win_rate_vs_sft_lc"])
                ),
            )
        )
    return arms


def _displacement_from_json(path: Path) -> list[DisplacementSeries]:
    rows = _load_json_list(path)
    series_list: list[DisplacementSeries] = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError(f"displacement series must be an object, got {row!r}")
        series_list.append(
            DisplacementSeries(
                beta=float(row["beta"]),
                steps=[int(x) for x in row["steps"]],
                chosen_logps=[float(x) for x in row["chosen_logps"]],
                rejected_logps=[float(x) for x in row["rejected_logps"]],
            )
        )
    return series_list


def _run_attribution(args: argparse.Namespace, metrics_dir: Path) -> int:
    skills = _merge_stage_map(args.skills_json, args.skills)
    if not skills:
        print(
            "error: --attribution needs --skills-json and/or --skills STAGE=PATH",
            file=sys.stderr,
        )
        return 2
    style = _merge_stage_map(args.style_json, args.style)
    table = build_stage_attribution_table(
        skills=skills,
        style_vs_sft=style or None,
        require_complete=not args.allow_incomplete,
    )
    out = write_stage_attribution_table(table, metrics_dir / "stage_attribution.json")
    print(f"attribution complete={table.complete} missing={table.missing} -> {out}")
    return 0


def _run_beta_plots(args: argparse.Namespace, plots_dir: Path) -> int:
    if args.beta_arms_json is None:
        print("error: --beta-plots needs --beta-arms-json", file=sys.stderr)
        return 2
    arms = _beta_arms_from_json(args.beta_arms_json)
    out = plot_beta_vs_kl_winrate(arms, plots_dir / "beta_vs_kl_winrate.png")
    print(f"beta plots -> {out}")
    return 0


def _run_displacement(args: argparse.Namespace, metrics_dir: Path, plots_dir: Path) -> int:
    if args.displacement_json is None:
        print("error: --displacement needs --displacement-json", file=sys.stderr)
        return 2
    series_list = _displacement_from_json(args.displacement_json)
    flags = [detect_displacement(s) for s in series_list]
    flags_path = write_json(metrics_dir / "displacement_flags.json", {"arms": flags})
    print(f"displacement flags -> {flags_path}")
    for series in series_list:
        plot_displacement(series, plots_dir / f"displacement_b{series.beta:g}.png")
    if len(series_list) > 1:
        plot_displacement_arms(series_list, plots_dir / "displacement_by_beta.png")
    return 0


def _run_chattiness(args: argparse.Namespace, metrics_dir: Path, plots_dir: Path) -> int:
    style = _merge_stage_map(args.style_json, args.style)
    if not style:
        print(
            "error: --chattiness needs --style-json and/or --style STAGE=PATH",
            file=sys.stderr,
        )
        return 2
    points = [
        chattiness_point_from_style_report(stage, path) for stage, path in style.items()
    ]
    summary_path = write_json(
        metrics_dir / "chattiness_summary.json",
        {"stages": summarize_chattiness(points)},
    )
    plot_raw_vs_length_controlled(points, plots_dir / "chattiness_raw_vs_lc.png")
    plot_length_markdown(points, plots_dir / "chattiness_length_markdown.png")
    print(f"chattiness summary -> {summary_path}")
    return 0


def _run_verdict(args: argparse.Namespace, metrics_dir: Path) -> int:
    if args.dpo_summary is None or args.ppo_summary is None:
        print(
            "error: --verdict needs --dpo-summary and --ppo-summary",
            file=sys.stderr,
        )
        return 2
    dpo = arm_from_head_to_head_summary(args.dpo_name, args.dpo_summary)
    ppo = arm_from_head_to_head_summary(args.ppo_name, args.ppo_summary)
    verdict = build_dpo_ppo_verdict(dpo, ppo)
    write_dpo_ppo_verdict(
        verdict,
        metrics_dir / "dpo_vs_ppo_verdict.json",
        markdown_path=metrics_dir / "dpo_vs_ppo_verdict.md",
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    modes = _selected_modes(args)
    if not modes:
        print(
            "error: select at least one mode: "
            "--attribution/--beta-plots/--displacement/--chattiness/--verdict",
            file=sys.stderr,
        )
        return 2

    metrics_dir = resolve_path(args.metrics_dir) if args.metrics_dir else DEFAULT_METRICS_DIR
    plots_dir = resolve_path(args.plots_dir) if args.plots_dir else DEFAULT_PLOTS_DIR
    metrics_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    try:
        if "attribution" in modes:
            code = _run_attribution(args, metrics_dir)
            if code:
                return code
        if "beta_plots" in modes:
            code = _run_beta_plots(args, plots_dir)
            if code:
                return code
        if "displacement" in modes:
            code = _run_displacement(args, metrics_dir, plots_dir)
            if code:
                return code
        if "chattiness" in modes:
            code = _run_chattiness(args, metrics_dir, plots_dir)
            if code:
                return code
        if "verdict" in modes:
            code = _run_verdict(args, metrics_dir)
            if code:
                return code
    except (FileNotFoundError, ValueError, RuntimeError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except ImportError as exc:
        print(f"error: missing dependency for analysis plots: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
