"""dataset prepare stages (sft / rm / dpo / ppo) + cli."""
from pipeline.prepare.cli import main, parse_args
from pipeline.prepare.config import load_config
from pipeline.prepare.decontam import build_eval_decontam_bank
from pipeline.prepare.io import load_processed_rows, save_rows
from pipeline.prepare.paths import CONFIG_DIR, ROOT, resolve_path
from pipeline.prepare.prefs import prepare_dpo, prepare_ppo, prepare_rm
from pipeline.prepare.sft import prepare_sft

__all__ = [
    "CONFIG_DIR",
    "ROOT",
    "build_eval_decontam_bank",
    "load_config",
    "load_processed_rows",
    "main",
    "parse_args",
    "prepare_dpo",
    "prepare_ppo",
    "prepare_rm",
    "prepare_sft",
    "resolve_path",
    "save_rows",
]
