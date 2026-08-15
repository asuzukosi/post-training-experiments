#!/usr/bin/env bash
# One-time secrets setup for the post-training experiments.
# Run ONCE on any RunPod pod that has the `posttraining-data` volume mounted at /workspace.
# Everything it writes lives on the volume, so it persists across pods.
#
#   bash setup_secrets.sh
#
# Afterwards, on every new pod just run:   set -a; source /workspace/.env; set +a
set -euo pipefail

VOL=/workspace
[ -d "$VOL" ] || { echo "ERROR: $VOL not mounted — attach the network volume first."; exit 1; }
mkdir -p "$VOL/hf"
export HF_HOME="$VOL/hf"

echo "== post-training secrets setup =="
read -rsp "Hugging Face token (write scope): " HF_TOK; echo
read -rsp "Weights & Biases API key:        " WB_KEY; echo

python -m pip install -q -U huggingface_hub wandb >/dev/null 2>&1 || true

# 1) HF: `hf auth login` persists the token to $HF_HOME/token (on the volume) —
#    huggingface_hub / transformers / push_to_hub all read it automatically.
hf auth login --token "$HF_TOK"

# 2) W&B: the key does NOT persist across pods (~/.netrc is ephemeral), so store it in the
#    volume .env that every pod sources.
cat > "$VOL/.env" <<EOF
export HF_HOME=$VOL/hf
export HF_XET_HIGH_PERFORMANCE=1
export WANDB_API_KEY=$WB_KEY
EOF
chmod 600 "$VOL/.env"
wandb login "$WB_KEY" >/dev/null 2>&1 || true   # also log in on this pod now

echo
echo "Done. Secrets live on the volume:"
echo "  HF token   -> $HF_HOME/token"
echo "  W&B key    -> $VOL/.env"
echo
echo "On every NEW pod, run:   set -a; source /workspace/.env; set +a"
