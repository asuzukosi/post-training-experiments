"""cli for prepare_data stages."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pipeline.prepare.config import load_config
from pipeline.prepare.paths import CONFIG_DIR
from pipeline.prepare.prefs import prepare_dpo, prepare_ppo, prepare_rm
from pipeline.prepare.sft import prepare_sft


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="prepare pipeline datasets")
    p.add_argument("--sft", action="store_true", help="build tulu sft 25k subset")
    p.add_argument("--rm", action="store_true", help="build ultrafeedback rm 20k")
    p.add_argument("--dpo", action="store_true", help="build disjoint dpo 10k")
    p.add_argument(
        "--ppo-prompts",
        action="store_true",
        help="build ppo 1.5k prompts from test_prefs",
    )
    p.add_argument("--all", action="store_true", help="run sft + rm + dpo + ppo-prompts")
    p.add_argument(
        "--config-dir",
        type=Path,
        default=CONFIG_DIR,
        help="directory with sft/rm/dpo/ppo yaml configs",
    )
    p.add_argument(
        "--skip-decontam",
        action="store_true",
        help="skip eval 8-gram decontam for sft (debug only)",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    do_sft = args.sft or args.all
    do_rm = args.rm or args.all
    do_dpo = args.dpo or args.all
    do_ppo = args.ppo_prompts or args.all

    if not any((do_sft, do_rm, do_dpo, do_ppo)):
        print("nothing to do; pass --sft/--rm/--dpo/--ppo-prompts/--all", file=sys.stderr)
        return 2

    config_dir = args.config_dir
    if do_sft:
        prepare_sft(load_config("sft", config_dir), skip_decontam=args.skip_decontam)
    if do_rm:
        prepare_rm(load_config("rm", config_dir))
    if do_dpo:
        prepare_dpo(load_config("dpo", config_dir), load_config("rm", config_dir))
    if do_ppo:
        prepare_ppo(load_config("ppo", config_dir))

    print("prepare_data done")
    return 0
