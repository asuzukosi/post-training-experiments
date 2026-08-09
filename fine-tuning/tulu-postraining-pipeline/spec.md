# Tulu Post-training Pipeline (SFT → RM → DPO → PPO)

Engineering spec for a self-contained experiment: build the canonical three-stage post-training
recipe end-to-end at 1.5B — instruction fine-tuning (SFT), a Bradley-Terry reward model (RM), then
preference optimization with DPO, then PPO — measuring what each stage adds rather than assuming it.

---

## 0. Overview

The standard post-training recipe has three stages with distinct, separable effects: **SFT** teaches
format and instruction-following; **RM** learns a scalar preference signal; **DPO** aligns to
preferences. Preference tuning is also known to inflate response length and markdown ("chattiness"),
which can flatter judged win-rates. This project builds all three stages and attributes the gains
stage-by-stage with direct measurement, producing a reusable SFT checkpoint, reward model, and
DPO-tuned model.

---


## 1. Objectives

1. **O1 — Stage attribution:** Measure what each stage adds — instruction-following (format), judged
   quality, and skills — separately after SFT and after DPO.
2. **O2 — β sensitivity:** Quantify how DPO's β (0.05 vs 0.1) moves the KL–quality trade-off, and
   detect **preference displacement** (both chosen *and* rejected log-probs falling).
3. **O3 — Chattiness effect:** Measure the length/markdown shift DPO induces and how much of the
   judged win-rate survives length control.
4. **O4 — DPO vs PPO:** At equal prompts/data, determine whether DPO matches PPO at this scale.

**Expected outcomes:** SFT fixes format but not warmth; DPO gives a clear judged win that shrinks
under length control; β=0.05 drifts further (more KL, more style); chosen and rejected log-probs both
fall (displacement); PPO ≈ DPO or slightly ahead at this scale.

---

## 2. Environment & prerequisites

**Compute:** RunPod. Network volume **`posttraining-data`** (`plgb5r5v05`, 200 GB, **US-CA-2**,
secure-cloud only); cache at `/workspace/hf`; inventory at `/workspace/_setup/MANIFEST.md`.

**Launch a pod** (A100 80 GB for training stages; RTX 4090 for eval):
```bash
runpodctl pod create --name tulu-postraining \
  --gpu-id "NVIDIA A100-SXM4-80GB" --gpu-count 1 \
  --data-center-ids US-CA-2 --cloud-type SECURE \
  --network-volume-id plgb5r5v05 --volume-mount-path /workspace \
  --container-disk-in-gb 30 --ports "22/tcp" --ssh \
  --image "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04" \
  --terminate-after "$(date -u -v+6H +%Y-%m-%dT%H:%M:%SZ)"
```
On every pod: `export HF_HOME=/workspace/hf HF_XET_HIGH_PERFORMANCE=1`.

**Constraint:** open-weights only — no gated models; judging runs locally (no external API).

**Models (all cached):**

| Role | Model | Notes |
|---|---|---|
| Base | `Qwen/Qwen2.5-1.5B` | trained through all stages |
| RM init | the SFT checkpoint from S1 | the RM is initialised from the instruction-tuned model, then its LM head is replaced with a scalar reward head |
| Judge (eval) | `Qwen/Qwen2.5-32B-Instruct` (drop to 14B for speed) | position-swapped, temp 0 |

**Data (all cached):**
- **SFT:** 25K-sample stratified subset of `allenai/tulu-3-sft-mixture` (decontaminate against eval sets first).
- **RM:** 20K pairs from `HuggingFaceH4/ultrafeedback_binarized` (`train_prefs`).
- **DPO:** a **disjoint** 10K pairs from the same split.
- **Eval:** `allenai/reward-bench` (RM), `tatsu-lab/alpaca_eval` (judged win-rate), `google/IFEval` + `cais/mmlu` (guardrails).

**Notes:** volumes attach only to secure-cloud pods in US-CA-2; use the `hf` CLI (not
`huggingface-cli`); delete the pod when done. Add anything with
`bash /workspace/_setup/add_content.sh model|dataset <id>`.

---

**Secrets, code & artifacts** — set up once; everything persists on the volume.

*Keys (one-time, on a pod with the volume mounted):*
```bash
export HF_HOME=/workspace/hf
hf auth login --token hf_XXXX          # write scope; token persists at /workspace/hf/token
cat > /workspace/.env <<'EOF'
export HF_HOME=/workspace/hf
export HF_XET_HIGH_PERFORMANCE=1
export WANDB_API_KEY=XXXX               # Weights & Biases (only key not persisted by a login)
EOF
chmod 600 /workspace/.env
```
*Every pod (one line; wrapped by `scripts/setup_env.sh`):* `set -a; source /workspace/.env; set +a`
— authenticates `hf` + `wandb` and points HF at the cache. W&B keys don't persist in `~/.netrc`
across pods, which is why the key lives in `/workspace/.env`; the HF token persists via
`/workspace/hf/token`.

*Code — clone the repo onto the volume so it persists; later pods just pull:*
```bash
cd /workspace && git clone https://github.com/<you>/<repo>.git
git config --global credential.helper 'store --file /workspace/.git-credentials'   # PAT saved on volume
# later pods: cd /workspace/<repo> && git pull
```

*Artifacts out —* models → HF Hub (`push_to_hub=True`, `hub_model_id="<you>/<base>_<task>_<datetime>"`,
`hub_private_repo=True`, or `hf upload <you>/<repo> results/checkpoints/<run>`); metrics/plots → W&B;
files → `scp -i ~/.runpod/ssh/runpodctl-ssh-key -P <PORT> -r root@<IP>:/workspace/<repo>/results ./`
(ip/port from `runpodctl ssh info`) or `runpodctl send`/`receive`. `results/` persists on the volume regardless.

> Plaintext secrets on a shared volume: `chmod 600 /workspace/.env`, never commit it (commit `.env.example`).

**Output naming.** Every trained checkpoint (and its W&B run) is named
`<base_name>_<task>_<dateandtime>` — `<base_name>` = short base id (`qwen2.5-1.5b`), `<task>` = the
stage/arm, `<dateandtime>` = filesystem-safe UTC stamp `YYYYMMDDThhmmZ`. Save under
`results/checkpoints/`. Examples: `qwen2.5-1.5b_sft_20260809T1730Z`,
`qwen2.5-1.5b_rm_20260809T1815Z`, `qwen2.5-1.5b_dpo-b0.05_20260809T1900Z`,
`qwen2.5-1.5b_dpo-b0.1_...`, `qwen2.5-1.5b_ppo_...`.

**Checkpointing & resume.** Training writes checkpoints to the volume so a failed or killed node
loses nothing — launch a new pod, re-mount the volume, resume. In every trainer config:
`output_dir=results/checkpoints/<run>`, `save_strategy="steps"`, `save_steps=50–100`,
`save_total_limit=3`, and launch with `resume_from_checkpoint=True` (auto-detects the latest
checkpoint, restoring model + optimizer + scheduler + step). Fix the W&B run id and set
`resume="allow"` so metrics continue on the same run. Long generation/scoring passes (Best-of-N,
teacher generation, difficulty filtering, judging) write results **incrementally** to the volume and
**skip already-completed items** on restart. Because `output_dir` is on the network volume (persists
across pods), resume survives full node loss — so cheaper interruptible/spot GPUs are viable.

**Conventions.** Pin a known-good dependency set in `requirements.txt` (TRL ↔ transformers ↔ vLLM are
version-sensitive) and install from it — never `pip install -U` mid-project. **Smoke-test** every stage
on a tiny subset before the full run. **Evaluation tooling:** `lm-eval` is the EleutherAI
lm-evaluation-harness (MMLU/IFEval/etc.); **judged win-rate uses one local judge** —
`Qwen2.5-32B-Instruct` served via vLLM's OpenAI-compatible endpoint, scored pairwise,
position-swapped, temperature 0 (no external API, no `alpaca-eval` package). Author any hand-made
input sets before running.

**Dataset exploration (before pipeline code).** Load the training and eval datasets locally (or via
`HF_HOME` on a volume-mounted pod) and analyze them in notebooks under `notebooks/` **before**
writing `src/pipeline/` prep/train code. Capture schema, length stats, mixture/source breakdown,
chosen/rejected structure, and eval-set fields that matter for decontamination. Findings inform the
25K SFT subset, the disjoint RM/DPO splits, and the PPO prompt pool.

---

## 3. Approach

```
Qwen2.5-1.5B base
   │  S1: SFT — 25K samples, 1 epoch, packing, prompt masking, lr 1e-5
   ▼
model_SFT ────────────────► eval: format + skills baseline
   │  S2: RM — 20K pairs, Bradley-Terry loss, 1 epoch ONLY (overfitting risk)
   ▼
rm_1.5B ──────────────────► RewardBench accuracy (target ≥65–70%)
   │  S3: DPO — 10K disjoint pairs, β ∈ {0.05, 0.1}, lr 5e-7, cached ref log-probs
   ▼
model_DPO(β×2) ───────────► eval: judged win-rate vs SFT, ± length control, KL, displacement
   │  S4: PPO — 1.5K prompts vs rm_1.5B, 1 grad step/batch
   ▼
model_PPO ────────────────► DPO-vs-PPO comparison at equal prompts (O4)
```

**Tooling:** `TRL` (`SFTTrainer`, `RewardTrainer`, `DPOTrainer`, `PPOTrainer`), `RewardBench`
(`allenai/reward-bench`), `lm-evaluation-harness` (IFEval/MMLU), `vLLM` (eval generation), `W&B`.

**3.0 Dataset notebooks** — under `notebooks/`, before any trainer implementation:
- Tulu-3 SFT mixture (`allenai/tulu-3-sft-mixture`): schema, sources, length stats, sample chat renders.
- UltraFeedback prefs (`HuggingFaceH4/ultrafeedback_binarized`, `train_prefs`): pair schema,
  chosen/rejected lengths, notes for a disjoint RM 20K / DPO 10K split.
- Eval sets for decontam awareness: `allenai/reward-bench`, `tatsu-lab/alpaca_eval`, `google/IFEval`,
  `cais/mmlu` — sizes/fields that must not leak into train.
Write a short findings note (notebook cells or `notebooks/findings.md`) that feeds data-prep decisions.

**3.1 SFT** — packing on, max_len 4096, lr 1e-5, warmup 3% + linear decay, 1 epoch, prompt masking.
Verify the chat template renders exactly once per role.

**3.2 RM** — linear head on the last non-pad token; BT log-sigmoid loss; **1 epoch**; effective batch
64 pairs. Log pair-accuracy; eval on RewardBench chat subsets.

**3.3 DPO ×2** — β ∈ {0.05, 0.1}, lr 5e-7, 1 epoch, cached reference log-probs (~50% peak-memory
saving). **Log chosen and rejected log-probs separately** — that pair of curves is the
preference-displacement detector.

**3.4 PPO** — KL β=0.05, clip 0.2, 1 gradient step/batch (confirm `clip_frac` ≈ 0); score only
EOS-terminated completions. Aligns `model_SFT` against `rm_1.5B` for the head-to-head with DPO.

---

## 4. Experiments & metrics

| Exp | Compares | Primary metric | Secondary |
|---|---|---|---|
| E1 (O1) | base vs SFT vs DPO | judged win-rate (pos-swapped) | IFEval, MMLU, format compliance |
| E2 (O2) | DPO β=0.05 vs β=0.1 | KL vs win-rate | chosen/rejected log-prob curves (displacement) |
| E3 (O3) | DPO judged ± length control | Δwin-rate after length correction | mean length, markdown rate |
| E4 (O4) | DPO vs PPO, same prompts | judged win-rate vs SFT | KL spent, wall-clock |

All judged evals ×3 runs, mean ± std. An MMLU drop > 5 pts is a broken-run signal.

---

## 5. Compute & cost

| Stage | GPU | ~Hours |
|---|---|---|
| SFT (25K × 1 epoch, 1.5B) | 1× A100 80 GB | 3 |
| RM (20K pairs, 1 epoch) | 1× A100 80 GB | 3 |
| DPO ×2 β arms (10K pairs) | 1× A100 80 GB | 5 |
| PPO | 1× A100 80 GB | 8 |
| Evals (RewardBench, judge, lm-eval) | 1× RTX 4090 24 GB | 3 |

- **Total:** ~22 GPU-hrs. **Picks:** A100 80 GB (~£1/hr), RTX 4090 (~£0.45/hr).
- **Estimated cost: ≈ £65.** **Wall-clock: ~2 days.**

---

## 6. Repo layout

```
tulu-postraining-pipeline/
├── README.md                      # findings summary
├── pyproject.toml                 # package metadata + pinned deps
├── requirements.txt               # trl, vllm, lm-eval, reward-bench, wandb, datasets
├── .env.example                   # WANDB_API_KEY, HF_HOME=/workspace/hf, HF_XET_HIGH_PERFORMANCE=1
├── .gitignore                     # data/, results/checkpoints/, .env, .git-credentials, *.log
├── configs/                       # one YAML per stage
│   ├── sft.yaml
│   ├── rm.yaml
│   ├── dpo.yaml
│   ├── ppo.yaml
│   └── eval.yaml
├── src/
│   └── pipeline/                  # importable package
│       ├── __init__.py
│       ├── data.py                # dataset prep, decontamination, disjoint splits
│       ├── sft.py
│       ├── reward_model.py
│       ├── dpo.py
│       ├── ppo.py
│       ├── eval.py                # per-stage battery + judged head-to-heads
│       └── analysis.py            # stage attribution, length, KL plots
├── scripts/                       # thin CLI entry points
│   ├── setup_env.sh               # wandb login + HF_HOME exports
│   ├── train_sft.py
│   ├── train_rm.py
│   ├── train_dpo.py
│   ├── train_ppo.py
│   └── run_eval.py
├── notebooks/                     # dataset eda before pipeline code
│   ├── tulu_sft_eda.ipynb
│   ├── ultrafeedback_eda.ipynb
│   ├── eval_sets_eda.ipynb
│   └── findings.md                # short notes feeding data-prep decisions
├── data/                          # gitignored: raw/ + processed/
├── results/
│   ├── checkpoints/               # <base>_<task>_<datetime>
│   ├── metrics/
│   └── plots/
└── tests/                         # smoke tests
```

---

## 7. Deliverables & acceptance criteria

- **Deliverables:** the SFT / RM / DPO(×β) checkpoints + a stage-attribution report — (1) what SFT vs
  DPO each added (format/skills/style table), (2) β-sensitivity + displacement curves, (3) chattiness
  report (raw vs length-controlled win-rate), (4) DPO-vs-PPO verdict.
- **Acceptance:** RM ≥ 65–70% on RewardBench chat; DPO beats SFT > 55% under the judge; a measured
  length effect reported both raw and length-controlled.

Optional extensions (native): an on-policy DPO variant (regenerate pairs from `model_SFT`); a
margin-loss RM ablation.

---

## 8. Risks & mitigations

- **RM: 1 epoch, no exceptions** — an overfit RM is hackable for the wrong reasons and poisons DPO.
- Keep the RM and DPO pair sets **disjoint**, or DPO evaluation is leaky.
- Chat-template bugs are silent — render and eyeball 5 examples before any run.
- Judge win-rates without length control flatter DPO — always report both.
