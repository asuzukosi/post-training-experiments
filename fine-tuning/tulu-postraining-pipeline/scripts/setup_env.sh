#!/usr/bin/env bash
# every pod: source scripts/setup_env.sh
# loads /workspace/.env, installs this project's pinned requirements (no -U), sets hf/wandb.
# one-time secrets (hf token + wandb key on the volume): bash fine-tuning/setup_secrets.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

ENV_FILE="${ENV_FILE:-/workspace/.env}"
if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
  echo "sourced $ENV_FILE"
else
  echo "warning: $ENV_FILE not found; run fine-tuning/setup_secrets.sh once on a volume-mounted pod"
fi

export HF_HOME="${HF_HOME:-/workspace/hf}"
export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"

if [[ -z "${WANDB_API_KEY:-}" ]]; then
  echo "warning: WANDB_API_KEY unset"
else
  # wandb key lives in /workspace/.env because ~/.netrc does not persist across pods
  wandb login "$WANDB_API_KEY" >/dev/null 2>&1 || true
fi

echo "HF_HOME=$HF_HOME"
echo "installing pinned requirements (no upgrades beyond the lock file)"
python -m pip install -r "$ROOT/requirements.txt"

if [[ -f "$ROOT/pyproject.toml" ]]; then
  python -m pip install -e "$ROOT"
fi

echo "setup_env done (cwd=$ROOT)"
