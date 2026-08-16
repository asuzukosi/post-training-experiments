# RunPod Network Volume — recreation record

> **Status: LIVE.** `posttraining-data` = **`zdhbaj21wa`, 200 GB, EU-RO-1**, created **2026-08-16**
> for the tulu pipeline's first real training runs. Billing ~£11/mo on provisioned size until deleted.
>
> **Not US-CA-2, and not 300 GB.** US-CA-2 was `Low` on both A100 SXM and H100 SXM at creation time,
> and volumes cannot move between regions — EU-RO-1 had A100 PCIe *and* A100 SXM 80 GB at normal
> stock. 200 GB covers this pipeline alone (~139 GB: 1.5B + the 32B and 14B judges + one run's
> checkpoints); the other bets' models are deliberately not populated. Volumes grow only.
>
> The previous volume (`plgb5r5v05`, 300 GB, US-CA-2) was deleted 2026-08-12 to stop its idle charge
> during Rung 1. The recipe below is what recreates this one.

Everything the volume held is **free to re-download** — all open weights, no gated repos. The original
population took **~11 minutes** on an H100 and cost about **$0.60**. Recreating it is cheap; leaving it
idle is not.

---

## 1. When to recreate

Recreate **immediately before the first Phase-5/6 run**, not earlier. Rung 1 (`Phase R`) runs 0.5B
models on a cheap GPU with no volume — models download in seconds to container disk.

The volume exists to make **checkpoints survive pod loss**. Until you are training something you
cannot afford to restart, you do not need it.

## 2. Create it

```bash
# 250 GB is the recommendation: ~124 GB of content + headroom for checkpoints.
# A 1.5B full-FT checkpoint is ~15.4 GB (3.1 GB bf16 weights + 12.3 GB fp32 AdamW state);
# save_total_limit: 3 means ~46 GB peak per training run, and tulu now has 13+ runs.
# Volumes GROW ONLY — never shrink — so start smaller and resize up if needed.
runpodctl network-volume create --name posttraining-data --size 250 --data-center-id US-CA-2
```

**Datacenter choice.** US-CA-2 was picked because it is network-volume-capable *and* stocks both
H100-class and cheaper GPUs. Re-verify at creation time — availability moves:

```bash
runpodctl datacenter list -o json | python3 -c 'import json,sys; d=[x for x in json.load(sys.stdin) if x["id"]=="US-CA-2"][0]; [print(g["displayName"], g.get("stockStatus")) for g in d["gpuAvailability"] if g.get("stockStatus")]'
```

Only **19 datacenters support network volumes**: AP-IN-2, AP-JP-1, CA-MTL-3, CA-MTL-4, EU-FR-1,
EU-NL-1, EU-RO-1, EUR-IS-1, EUR-IS-3, EUR-IS-4, EUR-NO-1, EUR-NO-2, US-CA-2, US-IL-1, US-MD-1,
US-MO-2, US-NC-1, US-NE-1, US-TX-3. Volumes attach to **secure-cloud pods only**, in their own
datacenter, and **cannot be moved** between regions.

## 3. Mount it and restore secrets

```bash
TERM_AT=$(date -u -v+4H +%Y-%m-%dT%H:%M:%SZ)
runpodctl pod create --name loader \
  --gpu-id "NVIDIA A100 80GB PCIe" --gpu-count 1 \
  --data-center-ids EU-RO-1 --cloud-type SECURE \
  --network-volume-id <NEW_VOLUME_ID> --volume-mount-path /workspace \
  --container-disk-in-gb 30 \
  --image "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04" \
  --ports "22/tcp" --ssh --terminate-after "$TERM_AT"

runpodctl ssh info <POD_ID>      # prints the ssh command + key path
```

> **The image must be a `runpod/*` one, not Docker Hub's `pytorch/pytorch`.** The `--ssh` flag only
> injects your key as the `PUBLIC_KEY` env var; something inside the image has to read it and start
> `sshd`. RunPod's images ship that start script, the vanilla PyTorch image does not — it has no
> `openssh-server` at all. Symptom: `Connection refused`, with `runtime: null` and `ports: null` on
> the pod. Cost of learning this the hard way: ~$0.60 and 25 minutes.
>
> **Python 3.12 is required** — `judgearena` declares `>=3.12` and the image ships 3.11, with no
> conda and no `python3.12`. Use `uv`, and **put the venv on the container disk, never on
> `/workspace`**:
>
> ```bash
> curl -LsSf https://astral.sh/uv/install.sh | sh
> export PATH="$HOME/.local/bin:$PATH"
> uv venv --python 3.12 /root/venv
> VIRTUAL_ENV=/root/venv uv pip install -r requirements.lock   # the lock, not requirements.txt
> ```
>
> **Why not on the volume.** Measured, both on this box: installing to `/workspace` wrote at
> **116 MB/min** and took ~20 minutes, because uv cannot hardlink across filesystems and had to copy
> ~60,000 files over MooseFS. The same install to `/root` took **2 seconds** from the warm cache.
> Worse, the penalty repeats at runtime — `import torch, transformers, trl, vllm` takes **15 s** from
> container disk and had not finished after several minutes from the volume. Every training run,
> eval and pytest pays that before doing any work.
>
> The venv does not survive pod loss, which is fine: `requirements.lock` (314 packages, in the repo)
> reproduces it exactly, and a rebuild is seconds with a warm cache. Checkpoints and the HF cache are
> what the volume is for.
>
> Verified on this pod: Python 3.12.14 / CUDA 12.4 / A100 80 GB PCIe — torch 2.5.1+cu124 with CUDA
> visible, transformers 4.51.0, trl 0.19.1, vllm 0.7.3.

**Secrets were on the volume and are gone.** Re-run once on the mounted pod — it prompts for an HF
write token and a W&B key:

```bash
bash fine-tuning/setup_secrets.sh
# writes: /workspace/hf/token  (via `hf auth login`)
#         /workspace/.env      (WANDB_API_KEY, chmod 600)
```

Then on **every** pod: `set -a; source /workspace/.env; set +a` (wrapped by `scripts/setup_env.sh`).

## 4. Repopulate

```bash
export HF_HOME=/workspace/hf HF_XET_HIGH_PERFORMANCE=1

# models
for m in \
  Qwen/Qwen2.5-0.5B \
  Qwen/Qwen2.5-0.5B-Instruct \
  Qwen/Qwen2.5-1.5B \
  Qwen/Qwen2.5-1.5B-Instruct \
  Qwen/Qwen2.5-14B-Instruct \
  Qwen/Qwen2.5-32B-Instruct \
  HuggingFaceTB/SmolLM2-1.7B-Instruct \
  allenai/OLMo-2-0425-1B-Instruct ; do hf download "$m"; done

# datasets
for d in \
  HuggingFaceH4/ultrafeedback_binarized \
  allenai/tulu-3-sft-mixture \
  openai/gsm8k \
  HuggingFaceH4/MATH-500 \
  ChilleD/SVAMP \
  cais/mmlu \
  meta-math/MetaMathQA \
  AI-MO/NuminaMath-CoT \
  google/IFEval \
  tatsu-lab/alpaca_eval \
  allenai/reward-bench ; do hf download "$d" --repo-type dataset; done
```

> **Use `hf`, not `huggingface-cli`** — the latter is removed in `huggingface_hub >= 1.x`. And
> `HF_XET_HIGH_PERFORMANCE=1`, not the deprecated `HF_HUB_ENABLE_HF_TRANSFER`.

## 5. What to download, and why

**Models — ~120 GB**

| Model | Size | Needed by |
|---|---:|---|
| `Qwen/Qwen2.5-0.5B` | 0.95 GB | distillation smoke tests; Rung-1 experiments |
| `Qwen/Qwen2.5-0.5B-Instruct` | 0.95 GB | Rung-1 judge stand-in |
| `Qwen/Qwen2.5-1.5B` | 2.9 GB | **the subject** — tulu all stages, rlvr GRPO base |
| `Qwen/Qwen2.5-1.5B-Instruct` | 2.9 GB | alfworld policy; distillation student reference |
| `Qwen/Qwen2.5-14B-Instruct` | 28 GB | distillation teacher 2; cheap judge for high-volume passes |
| `Qwen/Qwen2.5-32B-Instruct` | **62 GB** | teacher 1; judge; the gold signal. Largest single item. |
| `HuggingFaceTB/SmolLM2-1.7B-Instruct` | 20 GB* | tulu O7 — non-Qwen completions for the self-preference test |
| `allenai/OLMo-2-0425-1B-Instruct` | 2.8 GB | tulu O7 — second non-Qwen generator |

\* SmolLM2 is bloated by ONNX/extra artifacts; only the safetensors are needed.

**Datasets — ~3.7 GB**

| Dataset | Size | Needed by |
|---|---:|---|
| `HuggingFaceH4/ultrafeedback_binarized` | 405 MB | tulu RM / DPO / PPO prompts |
| `allenai/tulu-3-sft-mixture` | 1.4 GB | tulu SFT; distillation prompt slice |
| `AI-MO/NuminaMath-CoT` | 1.2 GB | distillation prompts |
| `meta-math/MetaMathQA` | 378 MB | rlvr prompts + contamination scan |
| `cais/mmlu` | 258 MB | skills guardrail across projects |
| `tatsu-lab/alpaca_eval` | 73 MB | judged win-rate |
| `allenai/reward-bench` | 15 MB | tulu RM gate (≥65% chat) |
| `openai/gsm8k` | 5.7 MB | rlvr headline; distillation eval |
| `HuggingFaceH4/MATH-500` | 442 KB | rlvr transfer check |
| `ChilleD/SVAMP` | 501 KB | distillation OOD transfer |
| `google/IFEval` | 212 KB | instruction-following guardrail |

**Do NOT re-download** — these served projects that no longer exist, and cost ~8 GB and ~40 MB:
`Qwen/Qwen2.5-3B-Instruct`, `internlm/internlm2-1_8b-reward`, `nvidia/HelpSteer2`,
`truthfulqa/truthful_qa`, `google-research-datasets/mbpp`, `openai/openai_humaneval`.

**Not in the HF cache:** `agentic-rl-alfworld` fetches its game data separately at runtime via
`alfworld-download` into `$ALFWORLD_DATA=/workspace/alfworld_data`.

## 6. Cost and gotchas

- Storage bills on **provisioned** size, not used — deleting files inside changes nothing. ~£0.055/GB-mo
  (250 GB ≈ **£14/mo**; the 300 GB version was £17/mo). It bills whether or not a pod is running.
- **Delete the volume between bets.** Over a 3-month programme, idle storage is ~£50 — more than the
  cheapest bet costs to run.
- **Prune each run's step checkpoints after its Hub push.** Every training run pushes to a private HF
  repo, so the volume only needs the *active* run's checkpoints, not every run's.
- Balance, not the spend limit, is the gate: `spendLimit` is **$/hr** (a rate cap), not a budget.
- SSH key: `~/.runpod/ssh/runpodctl-ssh-key`. Always `runpodctl pod delete <id>` when done.
