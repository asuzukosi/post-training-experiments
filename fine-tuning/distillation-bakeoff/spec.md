# Distillation Bake-off

Engineering spec for a self-contained experiment: on one student, one prompt set, and one eval
suite, measure whether **on-policy** distillation (single-teacher, multi-teacher, or self) transfers
capability better per training token than an offline sequence-level SFT baseline — and whether the
on-policy premium survives when the comparison is controlled by prompt count rather than tokens.

---

## Bet & success target

Serves **Bet 3 — the on-policy/self-distillation finding.**
- **Thesis:** on-policy (OPD/MOPD) and self (OPSD) distillation beat offline sequence-level KD at 1.5B on reasoning — or a clean null; either is a real small-scale result.
- **Headline target:** a **per-token controlled GSM8K pass@1 verdict** across offline-KD / OPD / MOPD / OPSD (equal tokens, ±std over 3 seeds), naming a winner or a documented null.
- **Artifact to ship:** the comparison + a **released distilled 1.5B reasoner** (the best arm, on HF).
- **Win / lose-but-ship:** a signed, CI-backed ranking is the win; a clean null (on-policy ≈ offline at this scale) is still a shippable finding few have published.

---

## 0. Overview

"Distillation" spans two distinct things that are often conflated: **sequence-level distillation**
(train the student on a stronger model's generated text) and **knowledge distillation proper** (match
the teacher's soft token distributions via KL). A third axis is **on-policy** distillation, where the
student generates and a teacher scores/filters its samples — which should reduce the train/inference
distribution mismatch of offline data. Naive single-teacher unfiltered self-training risks
distribution collapse. This project measures all of these on a fixed budget with a clean per-token
comparison.

**Method glossary (as used here):**
- **Sequence-level SFT (A):** SFT on teacher completions — the offline baseline.
- **On-policy distillation (C):** student samples N, teacher scores, SFT on the best (rejection-style).
- **MOPD (E):** multi-teacher on-policy — student samples, multiple teachers score and aggregate.
- **OPSD (F):** on-policy self-distillation — a stronger checkpoint of the student scores its own
  samples, with no external teacher.

---

## 1. Objectives

Scoped to the **on-policy question only** — one offline baseline against three on-policy variants, at
equal tokens. This is a method finding, not a released capability, so breadth is the first thing to go.

1. **O1 — Offline vs on-policy:** Per training token at 1.5B, does on-policy distillation (C) beat
   sequence-level offline SFT (A)?
2. **O3 — On-policy premium:** Test whether C beats A at **equal prompt count**, not just equal tokens
   — the two can disagree, and the prompt-controlled version is the honest comparison.
3. **O5 — MOPD:** Test whether multi-teacher on-policy (E) beats single-teacher on-policy (C).
4. **O6 — OPSD:** Test whether the student improves by distilling from a stronger checkpoint of
   **itself** (F), with no external teacher. Cheapest arm in the project and the most novel claim.

**Cut from this spec** *(recoverable if budget allows — none are load-bearing for the verdict)*:

| Cut | Was | Why |
|---|---|---|
| **O2** data-scaling knee (A at 10K/50K/200K) | 3 sizes | A is only needed as a baseline; one size at 50K suffices. Saves the 200K teacher generation outright. |
| **O4** two-teacher *offline* mix (Arm D) | £-cheap but off-question | Teacher diversity is tested on-policy by E; the offline mix answers a different question. |
| **Arm B** logit-level KD (GKD) | 6 GPU-hrs | A third *offline* flavour. Interesting, but the bet is offline-vs-on-policy, and A already anchors that axis. |
| **O7** rubric-scored on-policy (Arm G) | 6 GPU-hrs (with H) | A scoring-function ablation inside the on-policy arms — a refinement of a result we do not have yet. |
| **O8** narrow-vs-general (Arm H) | as above | A data-mixture question, not a distillation-method question. |

**Expected outcomes:** C beats A at equal tokens (on-policy data is worth more than more teacher
data); the on-policy premium narrows or vanishes when controlled by prompt count rather than tokens;
MOPD edges single-teacher on-policy but by less than the teacher-quality gap suggests; all gains come
with measurable style shift toward the teacher; **OPSD is the arm most prone to distribution
collapse** — watch entropy and n-gram diversity, and report collapse as a finding rather than
discarding the run.

---

## 2. Environment & prerequisites

**Compute:** RunPod. Network volume **`posttraining-data`** — **deleted 2026-08-12; it does not
exist.** Recreate it (250 GB, US-CA-2) *only* when starting a full-scale run — see
[`../VOLUME.md`](../VOLUME.md) for the exact create + repopulate + re-secrets recipe. **Rung 1 (§4b)
needs no volume.** Once created: cache at `/workspace/hf`. This is
the heaviest project — the 32B teacher plus on-policy arms need A100/H100 and a larger compute
budget (confirm account balance before starting).

**Launch pods** — serve the big teacher on A100/H100, run the small student trainer on a 4090:
```bash
# teacher-serving / on-policy scoring pod:
runpodctl pod create --name distill-teacher \
  --gpu-id "NVIDIA A100-SXM4-80GB" --gpu-count 1 \
  --data-center-ids US-CA-2 --cloud-type SECURE \
  --network-volume-id <NEW_VOLUME_ID> --volume-mount-path /workspace \
  --container-disk-in-gb 40 --ports "22/tcp" --ssh \
  --image "pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel" \
  --terminate-after "$(date -u -v+8H +%Y-%m-%dT%H:%M:%SZ)"
```
On every pod: `export HF_HOME=/workspace/hf HF_XET_HIGH_PERFORMANCE=1`.

> **Python 3.12 is required** — `judgearena` needs `>=3.12`, so the `py3.11` RunPod image will not
> install the stack. Create the env first: `conda create -y -n py312 python=3.12`, then use
> `/opt/conda/envs/py312/bin/python`. Verified on 3.12.13 / CUDA 12.4.

**Constraint:** open-weights only — no gated models. All on-policy scoring arms require the
**teacher and student to share a tokenizer**, so the whole roster stays in the Qwen2.5 family.

> **Staying in-family is worth more than tokenizer compatibility alone.** In published reasoning-
> distillation runs, a **same-family** teacher reached **45.0%** MATH-500 where a strong **cross-family**
> teacher reached only **33.6%** — an ~11-point gap attributed to shared tokenizer *plus* aligned
> prompting conventions and response style, which make the teacher's targets easier for the student to
> imitate. This is why a cross-family teacher is not a cheap "diversity" win: it changes the difficulty
> of the imitation task itself, confounding any teacher-quality comparison. Keep every teacher in the
> Qwen2.5 family, and if a cross-family teacher is ever added, treat it as its own arm, not a swap.

**Models:**

| Role | Model | Notes |
|---|---|---|
| Student | `Qwen/Qwen2.5-1.5B` (base); `Qwen/Qwen2.5-0.5B` for fast smoke tests | cached |
| Teacher 1 (main) | `Qwen/Qwen2.5-32B-Instruct` | cached; shared tokenizer → valid for KD / on-policy / MOPD / OPSD |
| Teacher 2 (diversity / MOPD) | `Qwen/Qwen2.5-14B-Instruct` | cached; shared tokenizer |
| OPSD reference | best prior student checkpoint (from C) | no external model |

**Data (cached):** prompts from `allenai/tulu-3-sft-mixture` (50K-prompt slice, deduped and
decontaminated with the included 8-gram scan); math from `openai/gsm8k`, `AI-MO/NuminaMath-CoT`;
OOD/eval from `HuggingFaceH4/MATH-500`, `ChilleD/SVAMP`, `cais/mmlu`, `google/IFEval`.

**Notes:** cost is dominated by 32B teacher serving — drop the teacher to `Qwen/Qwen2.5-14B-Instruct`
to cut on-policy-scoring cost. Volumes attach only to secure-cloud pods in US-CA-2; use `hf` (not
`huggingface-cli`); delete pods when done.

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
— authenticates `hf` + `wandb` and points HF at the cache. The W&B key lives in `/workspace/.env`
because it doesn't persist in `~/.netrc` across pods; the HF token persists via `/workspace/hf/token`.

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

**Output naming.** Every trained student checkpoint (and its W&B run) is named
`<base_name>_<task>_<dateandtime>` — `<base_name>` = short student id (`qwen2.5-1.5b` or
`qwen2.5-0.5b`), `<task>` = the arm, `<dateandtime>` = filesystem-safe UTC stamp `YYYYMMDDThhmmZ`.
Save under `results/checkpoints/`. Examples: `qwen2.5-1.5b_kd-seq_20260809T1730Z`,
`qwen2.5-1.5b_kd-logit_...`, `qwen2.5-1.5b_opd_...`, `qwen2.5-1.5b_mix_...`,
`qwen2.5-1.5b_mopd_...`, `qwen2.5-1.5b_opsd_...`, `qwen2.5-1.5b_opd-rubric_...`,
`qwen2.5-1.5b_narrow_...`.

**Checkpointing & resume.** Every arm writes checkpoints to the volume so a failed or killed node
loses nothing — launch a new pod, re-mount the volume, resume. In each trainer config:
`output_dir=results/checkpoints/<run>`, `save_strategy="steps"`, `save_steps=50–100`,
`save_total_limit=3`, launch with `resume_from_checkpoint=True` (restores model + optimizer +
scheduler + step). Fix the W&B run id and set `resume="allow"` so metrics continue on the same run.
The expensive teacher-generation and on-policy sampling/scoring passes write completions
**incrementally** to the volume and **skip already-completed prompts** on restart — critical here
since 32B teacher generation over the 50K prompts is the costliest step. Because `output_dir` is on the
network volume (persists across pods), resume survives full node loss.

**Conventions.** Pin a known-good dependency set in `requirements.txt` (TRL ↔ transformers ↔ vLLM are
version-sensitive, especially with the GKD/vLLM path) and install from it — never `pip install -U`
mid-project. **Smoke-test** every arm on the 0.5B student + a tiny subset before the full 1.5B run.
**Evaluation tooling:** `lm-eval` is the EleutherAI lm-evaluation-harness (MMLU/IFEval/GSM8K);
**judged win-rate uses one local judge** — `Qwen2.5-32B-Instruct` served via vLLM's OpenAI-compatible
endpoint, scored pairwise, position-swapped, temperature 0 (no external API, no `alpaca-eval` package).

---

## 3. Approach

```
50K prompts (deduped + decontaminated)
   │
   ├─► Teacher 1 completions (vLLM, temp 0.8) ─────────────► dataset D_T1 (50K)
   ├─► Teacher 2 completions (same 50K, MOPD only) ───────► dataset D_T2
   │
   ├─► Arm A: SFT on D_T1 at 50K ──────────────────────────► A   (offline baseline) (O1)
   ├─► Arm C: student samples N=8 → T1 scores → SFT top-1 ─► C  (50K prompts)      (O1/O3)
   ├─► Arm E: student samples N=8 → T1+T2 aggregate → SFT top-1 ► E (MOPD)         (O5)
   └─► Arm F: student samples N=8 → stronger self-ckpt scores → SFT top-1 ► F (OPSD)(O6)
```

**Four arms, not eight.** The bet is the on-policy/self-distillation verdict, so the comparison needs
exactly one offline baseline (A) and the three on-policy variants (C, E, F). Everything else was a
side question and is cut — see §1.

**Tooling:** `TRL` (`SFTTrainer` for A/C/D and on-policy arms; `GKDTrainer` for B — generalized
knowledge distillation with teacher logits, forward/reverse JSD mix), `vLLM` (teacher generation,
student sampling, teacher scoring), `distilabel` (pipeline orchestration), `lm-evaluation-harness`
+ `Inspect AI` (eval battery), `W&B` (per-arm loss/KL curves).

**3.1 Teacher data** — Teacher 1 completions for the 50K prompt set (vLLM, temp 0.8, top-p 0.95, max
1024 tokens); drop empty/refusal/format-broken outputs; dedupe. Teacher 2 completions for the same 50K
prompts, needed only for the MOPD arm.

**Filtering is part of the method, so record what it removes.** Log the drop rate and reason
(empty / refusal / no boxed answer / over-length / duplicate) per teacher. A teacher whose outputs are
filtered at a different rate is effectively contributing a different dataset size, which silently
breaks the equal-data premise of O1 and O5. Report the surviving example count per arm alongside every
result.

**3.2 Arm A — offline baseline (O1)** — sequence-level SFT on 50K Teacher-1 completions: prompt
masking, packing, lr 1e-5, 1 epoch. This is the line every on-policy arm must beat; it is not a
scaling study.

**Epoch count is a decision, not a default — and it interacts with teacher strength.** Published runs
show a *strong* teacher peaking at **epoch 1** (45.0%) and then declining (43.8%, 44.2%), while a
*weaker* teacher kept improving across three epochs (30.6% → 32.4% → 33.6%). With a 32B teacher, one
epoch is the right default here — but **evaluate a 2-epoch checkpoint on at least one arm** to confirm
the peak is where we assume. The cost is one extra eval pass, not a retrain.

> **Do not select checkpoints on validation loss.** In the same runs, omitting `<think>` tokens
> produced a *lower* validation loss while producing *worse* MATH-500 accuracy in almost every row.
> Loss and task metric decouple in distillation; every checkpoint decision here is made on the task
> metric. Fix the `<think>` handling once, apply it identically across all four arms, and state it —
> it is a silent ~1–4 point confound if arms differ.

**3.3 Arm C — on-policy distillation (O1/O3)** — student samples N=8/prompt (temp 1.0) on the same 50K
prompts; Teacher 1 scores each (1–10 rating, temp 0); SFT on the argmax per prompt. Because A and C
share a prompt set, the comparison can be reported **per token and per prompt** — report both, since
the on-policy premium often shrinks under prompt control.

**3.4 Arm E — MOPD (O5)** — student samples N=8; **both T1 and T2 score each sample** (temp 0);
aggregate (mean or max-agreement) → SFT on the argmax. Compare against C.

**3.5 Arm F — OPSD (O6)** — the best prior student checkpoint scores the current student's N=8
samples → SFT on the argmax. **No external model at any point.** Track collapse metrics closely
(distinct-n, self-BLEU, entropy): collapse is the expected failure mode here, and a documented
collapse is a result, not a discarded run.

---

## 4. Experiments & metrics

| Exp | Compares | Primary metric | Secondary |
|---|---|---|---|
| E1 (O1) | A vs C at equal tokens | judged win-rate vs student base | MMLU/IFEval/GSM8K, tokens-trained |
| E2 (O3) | A vs C at **equal prompts** | judged win-rate | response-length shift |
| E3 (O5) | E vs C | judged win-rate | teacher-agreement rate, diversity stats |
| E4 (O6) | F vs its seed checkpoint | judged win-rate | collapse metrics (distinct-n, self-BLEU, entropy) |

- All judged evals: position-swapped, temp-0 judge, ×3 runs, report mean ± std.
- Track style shift to teacher (length, markdown rate) — distillation inherits teacher style; report
  length-controlled win-rates alongside raw ones.

---

## 4b. Rung-1 dry run — validate before committing

**No full-scale run starts until this passes.** Rung 1 is this project's own pipeline exercised at
**0.5B on a cheap CUDA box with no persistent volume** — run it on **Vast.ai**, where the 20–33 GB
tier is ~$0.07–0.19/hr against RunPod's ~$0.25–0.28:

```bash
vastai search offers 'num_gpus=1 gpu_ram>=20 gpu_ram<=33 rentable=true reliability>0.95' -o dph
```

Filter on `disk_bw` and `inet_down` as well as price — host quality on a marketplace varies by two
orders of magnitude, and a slow disk dominates model-download time. RunPod RTX 4000 Ada (~$0.28) is
the fallback. Budget: **a few hours, ~$1–3.**

> **Nothing secret goes on this box.** Vast hosts physically own their hardware, so no write-scoped HF
> token, no GitHub PAT, no `ssh -A`. Rung 1 needs none of them: models are public, and results are
> copied off at teardown. `setup_secrets.sh` is **RunPod-only** — it logs in a *write* token.
> Full platform guidance: the global `gpu-cloud-platforms` skill.

**Everything runs on the pod, not the laptop.** The dev machine has a different torch, no vLLM (no
macOS build), and MPS can change results — 90% on CPU and CUDA versus **80% on MPS** for the same
model and prompts. Running Rung 1 on a cheap CUDA box with the **exact pinned stack** removes every
one of those variables at ~$0.28/hr. The only work that stays local is pure-Python and CPU-only
(n-gram scans, the verifier audit), where no GPU library is involved.

It answers three things that estimates cannot:

1. **Does the pinned stack install?** `torch 2.5.1 + vllm 0.7.3 + trl 0.19.1 + transformers 4.51.0`
   has never been resolved together on any machine. It is the highest-probability failure in the plan
   and costs ~$0.10 to discover here instead of on an H100.
2. **Do the CUDA-only paths work?** vLLM generation and judging are only ever unit-tested against
   injected fakes on the dev machine, because vLLM has no macOS build.
3. **What is the real tokens/sec?** Every hour in §5 is an estimate. One measurement at 0.5B replaces
   them; scaling 0.5B → 1.5B is roughly **3× compute** (approximate — attention does not scale
   linearly with parameters).

**This project's Rung-1 checklist:**

| Check | Retires |
|---|---|
| Teacher generation under vLLM (0.5B standing in for 32B) | Throughput per completion — the 32B generation line is the costliest step here, and it is currently an estimate |
| One on-policy loop end-to-end: sample N=8 → score → SFT top-1 | The Arm C mechanism, and the sample-score-train wiring before it runs at teacher scale |
| **OPSD collapse metrics** (distinct-n, self-BLEU, entropy) at 0.5B | The failure mode this spec predicts. A 0.5B model collapses *faster*, so it is the better place to confirm the metrics catch it |

**OPSD is the cheapest arm and needs no teacher**, which makes it the natural Rung-1 candidate: the
whole Arm F loop can run at 0.5B for pennies, and if the collapse metrics do not fire there, they will
not fire at 1.5B either.

---

## 5. Compute & cost

| Stage | GPU | ~Hours |
|---|---|---|
| Teacher 1 generation (50K completions, 32B) | 1× A100 80 GB / H100 (vLLM) | 4 |
| Teacher 2 generation (50K, 14B — MOPD only) | 1× A100 80 GB (vLLM) | 3 |
| Arm A SFT, 50K offline baseline (1.5B) | 1× RTX 4090 24 GB | 2 |
| Arm C on-policy (sample + score + SFT) | 1× A100 80 GB (teacher) + 1× 4090 (train) | 5 |
| Arm E MOPD (T1+T2 score + SFT) | 1× A100 80 GB (teachers) + 1× 4090 (train) | 6 |
| Arm F OPSD (self-score + SFT — no teacher) | 1× RTX 4090 24 GB | 3 |
| Eval battery (4 arms) | 1× RTX 4090 24 GB | 3 |

> A 32B teacher in bf16 is ~65 GB — serve it standalone with vLLM on one A100/H100 and run the small
> student trainer on a separate 4090; don't co-locate. Arm E needs both teachers resident for scoring.

- **Total:** ~26 GPU-hrs (down from ~47 for the full eight-arm version).
  **Picks:** A100 80 GB or H100 for teacher serving, RTX 4090 for the student trainer/eval.
- **Estimated cost: ≈ £70.** **Wall-clock: ~2 days.**
- ⚠️ **These rates are stale.** The table assumes A100 ≈ £1/hr, H100 ≈ £2/hr and RTX 4090 ≈ £0.45/hr;
  live RunPod *secure-cloud* rates are **A100 SXM £1.25 / A100 PCIe £1.09 / H100 £2.59 / 4090 £0.58**,
  ~25–30% higher. See [`../PORTFOLIO.md`](../PORTFOLIO.md) → **Budget**.
- **Teacher generation is shareable.** `tulu-postraining-pipeline`'s S5 stage also generates from
  `Qwen2.5-32B-Instruct`. If tulu runs first, reuse its completions where prompts overlap rather than
  regenerating — the single largest saving available here.
- **Cut order if budget tightens:** Arm E (MOPD) first — it costs both the 14B teacher generation and
  the dual-scoring pass, ~9 hrs together. **Never cut Arm F (OPSD):** it needs no teacher at all, is
  the cheapest arm at 3 hrs, and carries the most novel claim in the bet.

---

## 6. Repo layout

```
distillation-bakeoff/
├── README.md                      # findings summary
├── pyproject.toml                 # package metadata + pinned deps
├── requirements.txt               # trl (GKDTrainer), vllm, distilabel, lm-eval, inspect-ai, wandb
├── .env.example                   # WANDB_API_KEY, HF_HOME=/workspace/hf, HF_XET_HIGH_PERFORMANCE=1
├── .gitignore                     # data/, results/checkpoints/, .env, .git-credentials, *.log
├── configs/                       # one YAML per stage/arm
│   ├── teacher_gen.yaml
│   ├── sft.yaml
│   ├── gkd.yaml
│   ├── onpolicy.yaml
│   ├── rubric.yaml
│   └── eval.yaml
├── src/
│   └── distill/                   # importable package
│       ├── __init__.py
│       ├── data.py                # dedup, quality filter, 8-gram decontamination
│       ├── teacher_gen.py         # T1 + T2 completion engines
│       ├── kd.py                  # arm A (seq-level SFT) + arm B (logit KD)
│       ├── on_policy.py           # arms C/E/F/G (single/multi/self/rubric on-policy)
│       ├── narrow.py              # arm H (skill vs general)
│       ├── eval.py                # eval battery + style metrics
│       └── metrics.py             # collapse (distinct-n, self-BLEU), length control
├── scripts/                       # thin CLI entry points (pass --arm)
│   ├── setup_env.sh               # wandb login + HF_HOME exports
│   ├── generate_teacher.py
│   ├── train_kd.py                # --arm {seq,logit}
│   ├── train_on_policy.py         # --arm {opd,mix,mopd,opsd,rubric}
│   ├── train_narrow.py
│   └── run_eval.py
├── data/                          # gitignored: raw/ + processed/
├── results/
│   ├── checkpoints/               # <base>_<task>_<datetime>
│   ├── metrics/
│   └── plots/
└── tests/                         # smoke tests
```

---

## 7. Deliverables & acceptance criteria

- **Deliverables:** (1) offline-vs-on-policy table at equal tokens **and** equal prompts (A vs C),
  (2) on-policy premium verdict with CIs, (3) MOPD verdict (E vs C), (4) OPSD verdict (F vs seed, with
  collapse metrics), (5) per-arm style-shift measurements, (6) the released distilled 1.5B reasoner —
  whichever arm wins.
- **Acceptance:** a signed on-policy premium reported both per-token and per-prompt (**a null is a
  valid finding**, and the two views disagreeing is itself the result worth publishing); MOPD and OPSD
  each report a signed result; OPSD reports collapse metrics whether or not it wins; all numbers as
  mean ± std over 3 runs.

---

## 8. Risks & mitigations

- **On-policy, MOPD and OPSD require a shared tokenizer** — keep teachers in the Qwen2.5
  family; a non-Qwen teacher can only feed the sequence-level arms (A/D).
- **OPSD is the highest collapse risk** (self-training) — enforce filtering; track distinct-n /
  self-BLEU every arm.
- Distillation ceiling is the teacher — if the student plateaus below expectation, check teacher
  quality on the eval distribution first.
- Single-teacher unfiltered self-training is the collapse regime — keep filters + the diversity arm;
  never skip dedup.
- Style transfer is confounded with quality in judged evals — report length-controlled win-rates.
- Distilling on contaminated prompts inflates evals — run the decontamination scan before training.
- Teacher outputs carry license/ToS constraints — record dataset provenance.
