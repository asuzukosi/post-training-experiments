#!/usr/bin/env python3
"""push the prepared datasets to a hub dataset repo, so a rented box does not rebuild them.

prep needs the full tulu and ultrafeedback downloads plus the decontam bank before it can
sample anything. that is minutes of gpu time spent on cpu work. uploading the prepared
`data/processed/` once turns it into a single download.

the trainers load with `load_from_disk`, so the folder goes up unchanged and comes back
into the same place:

  python scripts/prepare/push.py
  hf download <user>/<repo> --repo-type dataset --local-dir data/processed

examples:
  python scripts/prepare/push.py                       # private, default repo name
  python scripts/prepare/push.py --public --repo my-sets
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from hub import DEFAULT_HUB_PRIVATE, read_hf_token, resolve_hub_username
from prepare.paths import resolve_path

DEFAULT_REPO = "tulu-postraining-data"
DEFAULT_PATH = "data/processed"


def describe_sets(folder: Path) -> str:
    """a dataset card listing what is actually on disk, not what we expect to be."""
    from datasets import load_from_disk

    lines = [
        "# Prepared post-training sets",
        "",
        "Built by `scripts/prepare/*.py` from `allenai/tulu-3-sft-mixture` and",
        "`HuggingFaceH4/ultrafeedback_binarized`, seed 42. Every set is 8-gram",
        "decontaminated against the benchmarks it is scored on (MMLU, IFEval,",
        "RewardBench), before sampling — so a dropped row is replaced, not left short.",
        "",
        "| Set | Rows | Columns |",
        "|---|---|---|",
    ]
    for path in sorted(p for p in folder.iterdir() if p.is_dir()):
        ds = load_from_disk(str(path))
        lines.append(f"| `{path.name}` | {ds.num_rows:,} | {', '.join(ds.column_names)} |")
    lines += [
        "",
        "Load with `load_from_disk` after downloading, the same way the trainers do:",
        "",
        "```sh",
        "hf download <repo> --repo-type dataset --local-dir data/processed",
        "```",
        "",
    ]
    return "\n".join(lines)


def push_processed(
    *,
    repo: str,
    folder: Path,
    private: bool,
    username: str | None = None,
) -> str:
    from huggingface_hub import HfApi

    if not folder.is_dir():
        raise FileNotFoundError(f"nothing to push: {folder}")
    sets = sorted(p.name for p in folder.iterdir() if p.is_dir())
    if not sets:
        raise ValueError(f"no prepared sets in {folder}")

    token = read_hf_token()
    repo_id = f"{resolve_hub_username(username, token=token)}/{repo}"
    api = HfApi(token=token)

    print(f"pushing {len(sets)} sets to {repo_id} (private={private}): {', '.join(sets)}")
    api.create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)
    (folder / "README.md").write_text(describe_sets(folder), encoding="utf-8")
    api.upload_folder(
        folder_path=str(folder),
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=f"prepared sets: {', '.join(sets)}",
    )
    print(f"pushed -> https://huggingface.co/datasets/{repo_id}")
    return repo_id


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="push prepared datasets to the hub")
    p.add_argument("--repo", type=str, default=DEFAULT_REPO, help="repo name (not the full id)")
    p.add_argument("--username", type=str, default=None, help="hub user; else whoami")
    p.add_argument("--path", type=Path, default=Path(DEFAULT_PATH))
    p.add_argument(
        "--public",
        action="store_true",
        help="create the repo public; private by default, like the checkpoint pushes",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        push_processed(
            repo=args.repo,
            folder=resolve_path(args.path),
            private=DEFAULT_HUB_PRIVATE and not args.public,
            username=args.username,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print("push done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
