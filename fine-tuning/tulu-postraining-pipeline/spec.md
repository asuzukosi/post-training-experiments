# Tulu Post-training Pipeline (SFT → RM → DPO → PPO)

Engineering spec for a self-contained experiment: build the canonical three-stage post-training
recipe end-to-end at 1.5B — instruction fine-tuning (SFT), a Bradley-Terry reward model (RM), then
preference optimization with DPO, then PPO — measuring what each stage adds rather than assuming it.

---

## Bet & success target

The **base-recipe factory** every other bet starts from (supporting artifact).
- **Thesis:** a fully reproducible, open, cheap 1.5B post-training recipe — and a clean answer to "what does each stage actually buy, and does DPO match PPO?"
- **Headline target:** **released SFT / RM / DPO(×β) / PPO checkpoints** (HF) + a **stage-attribution table** (format/skills/style per stage) + a **DPO-vs-PPO verdict at equal data** (with CIs) + RM ≥65–70% on RewardBench-chat. (Goal is a reproducible recipe + reusable checkpoints, *not* beating the official Qwen2.5-1.5B-Instruct.)
- **Artifact to ship:** the checkpoints + the recipe + the stage-attribution report.
- **Win / lose-but-ship:** reusable checkpoints that seed Bets 1–3 + a documented DPO-vs-PPO answer = win regardless of which method wins.

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

Three groups, run in order. Each depends on the artifacts the previous one produces — the RM, the
policy checkpoints and the judge are built once and reused throughout.

**A. The recipe — what each stage buys (S1–S4)**

1. **O1 — Stage attribution:** Measure what each stage adds — instruction-following (format), judged
   quality, and skills — separately after SFT and after DPO.
2. **O2 — β sensitivity:** Quantify how DPO's β (0.05 vs 0.1) moves the KL–quality trade-off, and
   detect **preference displacement** (both chosen *and* rejected log-probs falling).
3. **O3 — Chattiness effect:** Measure the length/markdown shift DPO induces and how much of the
   judged win-rate survives length control.
4. **O4 — DPO vs PPO:** At equal prompts/data, determine whether DPO matches PPO at this scale.

**B. The data engine — how the preference data is made (S5–S7)**

5. **O5 — RS-SFT vs DPO:** on **identical** data, does rejection-sampling SFT
   (Best-of-N → keep top-1 → SFT, positive-only) or DPO yield the better 1.5B model? The DPO half is
   already trained by S3; only the RS arm is new. Verdict with CIs.
6. **O6 — Structured preferences:** does **constraint-based pair construction** —
   generate the same prompt *with* vs *without* an explicit constraint, `chosen` = the constrained
   output — measurably improve instruction-following? Measured on IFEval strict accuracy.
7. **O7 — Judge reliability:** quantify **length bias**, **self-preference bias** and
   **position bias** in the judge that produces every headline number here.

**C. The KL frontier — how far the RM can be pushed before it backfires (S8–S10)**

8. **O8 — Inverted-U:** Show a held-out **gold** signal peaks and declines as optimization pressure
   against the **proxy RM** increases, while the proxy score climbs monotonically. Best-of-N is the
   clean dial — its KL from the reference grows predictably as ≈ `log N`, with no RL instability.
9. **O9 — BoN vs PPO:** at equal *measured* KL, which method buys more gold quality? Expectation:
   online RL spends KL faster than Best-of-N.
10. **O10 — Mitigations:** do a higher KL penalty, **RM ensembling** (min-over-ensemble reward), or
    fixed-KL early-stopping shift the peak rightward?
11. **O11 — Qualitative degradation:** measure length, stock-phrase rate and sycophancy answer-flip
    rate as functions of KL, and test whether a **sycophancy steering vector** recovers quality at
    high KL without retraining.
12. **O12 — Reward whitening:** add a reward-whitening variant to the PPO arm, and report every gold
    curve **raw and length-controlled** so genuine decline is separable from length inflation.

**Gated extensions** — run only if O11 yields a usable steering vector; not required for acceptance.

13. **O13 — Steering vs character training:** a mini character-training run (constitution →
    self-critique-revision → DPO) against the best steering config. Which wins on **sycophancy
    reduction per unit of capability damage**, and which survives **multi-turn** conversation?
14. **O14 — Preference-pair screening:** does the sycophancy vector's projection rank which
    preference pairs *taught* the sycophancy? Cheap, and it explains the O8 mechanism directly — a
    proxy RM trained on pairs that reward flattery is *why* the gold signal turns over.

> **O7 is not optional garnish — it is a credibility prerequisite for O4 and O9.** The judge here is
> `Qwen2.5-32B-Instruct` scoring `Qwen2.5-1.5B` outputs, i.e. it scores its own model family, and it
> is also the **gold signal** for the whole KL frontier. Until self-preference is measured, both the
> DPO-vs-PPO verdict and the Goodhart curve carry an unquantified confound. Position bias is already
> controlled (JudgeArena pairwise swaps orderings) and length bias is already reported (§3.3);
> **self-preference is the gap.**

**Expected outcomes:** SFT fixes format but not warmth; DPO gives a clear judged win that shrinks
under length control; β=0.05 drifts further (more KL, more style); chosen and rejected log-probs both
fall (displacement); PPO ≈ DPO or slightly ahead. RS-SFT is competitive and simpler while DPO edges it
on preference metrics; structured prefs give a real IFEval gain; the judge shows non-zero length bias
and mild self-preference. Proxy score climbs ~monotonically in `log N` while gold traces an inverted-U
— at 1.5B the decline may be a shallow plateau, which plus clear qualitative degradation still counts;
PPO reaches lower gold-quality-per-KL than BoN; ensembling and a higher KL penalty delay the peak.

---

## 2. Environment & prerequisites

**Compute:** RunPod. Network volume **`posttraining-data`** — **deleted 2026-08-12; it does not
exist.** Recreate it (250 GB, US-CA-2) *only* when starting a full-scale run — see
[`../VOLUME.md`](../VOLUME.md) for the exact create + repopulate + re-secrets recipe. **Rung 1 (§4b)
needs no volume.** Once created: cache at `/workspace/hf`.

**Launch a pod** (A100 80 GB for training stages; RTX 4090 for eval):
```bash
runpodctl pod create --name tulu-postraining \
  --gpu-id "NVIDIA A100-SXM4-80GB" --gpu-count 1 \
  --data-center-ids US-CA-2 --cloud-type SECURE \
  --network-volume-id <NEW_VOLUME_ID> --volume-mount-path /workspace \
  --container-disk-in-gb 30 --ports "22/tcp" --ssh \
  --image "pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel" \
  --terminate-after "$(date -u -v+6H +%Y-%m-%dT%H:%M:%SZ)"
```
On every pod: `export HF_HOME=/workspace/hf HF_XET_HIGH_PERFORMANCE=1`.

> **Python 3.12 is required** — `judgearena` needs `>=3.12`, so the `py3.11` RunPod image will not
> install the stack. Create the env first: `conda create -y -n py312 python=3.12`, then use
> `/opt/conda/envs/py312/bin/python`. Verified on 3.12.13 / CUDA 12.4.

**Constraint:** open-weights only — no gated models; judging runs locally (no external API).

**Models (all cached):**

| Role | Model | Notes |
|---|---|---|
| Base | `Qwen/Qwen2.5-1.5B` | trained through all stages |
| RM init | the SFT checkpoint from S1 | the RM is initialised from the instruction-tuned model, then its LM head is replaced with a scalar reward head |
| Judge (eval) | `Qwen/Qwen2.5-32B-Instruct` (drop to 14B for speed) | **JudgeArena** + in-process vLLM backend; pairwise, position-swapped, temp 0 |
| Teacher — RS-SFT completions (S5) | `Qwen/Qwen2.5-32B-Instruct` | N=8/prompt; **drop to 14B for the high-volume selection pass** |
| Non-Qwen generators (S7 self-preference) | `HuggingFaceTB/SmolLM2-1.7B-Instruct`, `allenai/OLMo-2-0425-1B-Instruct` | supply non-Qwen completions so judge self-preference is measurable — both cached |

**Data (all cached):**
- **SFT:** 25K-sample stratified subset of `allenai/tulu-3-sft-mixture` (decontaminate against eval sets first).
- **RM:** 20K pairs from `HuggingFaceH4/ultrafeedback_binarized` (`train_prefs`).
- **DPO:** a **disjoint** 10K pairs from the same split.
- **Eval:** `allenai/reward-bench` (RM), `tatsu-lab/alpaca_eval` (judged win-rate), `google/IFEval` + `cais/mmlu` (guardrails).
- **Structured-preference prompts (S6, author locally):** ~300 instruction-following prompts with
  **explicit, checkable** constraints — word count, "start each sentence with G", exactly-3-bullets,
  JSON-only. This is hand-authoring work, not a download — the real cost of S6.
- **Optimization prompts (S8):** ~1.5K prompts the RM never trained on — UF `test_prefs` plus a
  `allenai/tulu-3-sft-mixture` slice. Held-out is the whole point: BoN against prompts in the RM's
  training set measures memorisation, not over-optimization.
- **Qualitative probes (S10, author locally):** ~200 "are you sure?" / opinion-disagreement prompts
  **with known-correct answers** (the known-correct answer is what makes flip-rate measurable rather
  than a vibe); a stock-phrase list ("As an AI language model", "Certainly!", "It's important to
  note"); ~100 trait-eliciting vs trait-suppressing pairs for vector extraction; and, for the gated
  O13 only, a 5-principle constitution (honest-uncertainty, no flattery, directness, calibrated
  confidence, no stock phrases) → 3–5K critique-revised pairs.

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
on a tiny subset before the full run. **Evaluate before train:** every model used as a training init
must have a recorded baseline eval **before** that training job starts — base before SFT; `model_SFT`
before RM / DPO / PPO; do not train “blind” and only evaluate afterward. Post-train evals still run
(and the RM RewardBench gate still applies after RM), but they never replace the pre-train baseline.
**Evaluation tooling:** `lm-eval` is the EleutherAI lm-evaluation-harness (MMLU/IFEval/etc.).
**Judged win-rate uses JudgeArena** ([OpenEuroLLM/JudgeArena](https://github.com/OpenEuroLLM/JudgeArena))
with a **local open-weights judge** — `Qwen2.5-32B-Instruct` (or 14B) via JudgeArena's in-process
**vLLM** backend (`VLLM/Qwen/Qwen2.5-32B-Instruct`). Pairwise, position-swapped, temperature 0.
No external API. **Do not use the `alpaca-eval` package** — JudgeArena owns the AlpacaEval /
Arena-Hard *protocol*; pin `judgearena` in `requirements.txt` instead. Pipeline code wraps
JudgeArena; it does not reimplement pairwise prompts or verdict parsing.

**Headline H2H set:** `tatsu-lab/alpaca_eval` eval split (805) via JudgeArena's AlpacaEval task
(length-controlled win-rate is the primary chat metric). Optional diagnostic: Arena-Hard.
**Custom prompt pools** (PPO 1.5k, RS-SFT tournament, O7 self-preference) use the **same
JudgeArena judge backend** on an exported local dataset — one judge implementation, not a second
homemade scorer.

**Eval GPU schedule (single GPU, sequential):** load candidate A → generate (incremental JSONL) →
unload → load candidate B → generate → unload → … → load JudgeArena judge → score pairs → unload.
Do **not** keep a standing OpenAI-compatible judge server. Optional: keep two 1.5B candidates
resident and generate **sequentially** (not concurrently); never co-reside the 32B judge with
candidates. Persist JudgeArena outputs under `results/metrics/` and **skip already-completed
pairs** on restart. Author any hand-made input sets before running.

**Dataset exploration (before pipeline code).** Load the training and eval datasets locally (or via
`HF_HOME` on a volume-mounted pod) and analyze them in notebooks under `notebooks/` **before**
writing `src/` prep/train code. Capture schema, length stats, mixture/source breakdown,
chosen/rejected structure, and eval-set fields that matter for decontamination. Findings inform the
25K SFT subset, the disjoint RM/DPO splits, and the PPO prompt pool.

---

## 3. Approach

```
Qwen2.5-1.5B base ────────► eval BEFORE S1 (IFEval/MMLU + format baseline)
   │  S1: SFT — 25K samples, 1 epoch, packing, prompt masking, lr 1e-5
   ▼
model_SFT ────────────────► eval BEFORE S2/S3/S4 (and post-SFT battery)
   │  S2: RM — 20K pairs, Bradley-Terry loss, 1 epoch ONLY (overfitting risk)
   ▼
rm_1.5B ──────────────────► RewardBench-chat gate (target ≥65–70%)
   │  S3: DPO — 10K disjoint pairs, β ∈ {0.05, 0.1}, lr 5e-7, cached ref log-probs
   ▼
model_DPO(β×2) ───────────► eval: judged win-rate vs SFT, ± length control, KL, displacement
   │  S4: PPO — 1.5K prompts vs rm_1.5B, 1 grad step/batch
   ▼
model_PPO ────────────────► DPO-vs-PPO comparison at equal prompts (O4)
```

**Evaluate-before-train (hard rule).** Do not launch a full train until the init checkpoint for that
stage has metrics on disk / W&B. Minimum pre-train suite: IFEval + MMLU for generative inits (base,
SFT, DPO, PPO policy); judged win-rate baselines as needed for stage attribution. RM training starts
only after the SFT init baseline exists; RM quality is then gated post-train on RewardBench-chat.

**Tooling:** `TRL` (`SFTTrainer`, `RewardTrainer`, `DPOTrainer`, `PPOTrainer`), `RewardBench`
(`allenai/reward-bench`), `lm-evaluation-harness` (IFEval/MMLU), `vLLM` (in-process candidate
generation), **JudgeArena** (pairwise judged win-rate + length-controlled AlpacaEval metric;
local vLLM judge), `W&B`.

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

### Stages S5–S7 — run after S1–S4

*All three reuse machinery S1–S4 already builds: `P3-1` vLLM generation (incremental,
skip-completed), `P3-2m` JudgeArena pairwise judge, `P3-3` length control, `P3-4` lm-eval, `P4-4` CI
writer.*

**3.5 RS-SFT arm (O5)** — the missing half of the RS-SFT-vs-DPO comparison.

```
DPO prompt set  ──► 32B teacher, N=8/prompt, temp 0.8, top-p 0.95
                        │
                        ▼  JudgeArena pairwise tournament, position-swapped, temp 0
                  top-1 per prompt ──► RS-SFT data ──► SFT (prompt-masked) ──► model_RS
```

Sort completions by token length before batched judging to minimise padding. **Use the same prompt
set as the 10K DPO split** — "identical data" is the whole point of the comparison; a different prompt
pool voids the verdict. **Cost control:** judging N=8 is the expensive step (~7 comparisons/prompt),
so run the selection pass on **`Qwen2.5-14B-Instruct`** and reserve the 32B for final head-to-heads.

> **Framing caveat, state it in the write-up:** the teacher here is 32B, i.e. this is
> teacher-distillation-plus-selection rather than *on-policy* rejection sampling of the 1.5B policy.
> The honest claim is "RS-SFT from a strong teacher vs DPO on that teacher's preferences", not
> "rejection sampling beats DPO" in general.

**3.6 Structured preferences (O6)** — per authored constraint prompt, generate one completion **with**
the constraint and one **without**; `chosen` = the constraint-following output. **The stored prompt
must include the constraint** — otherwise the model learns to add constraints unprompted. Train with
the existing `src/trainers/dpo.py`; evaluate on IFEval strict accuracy against `model_SFT`.

**3.7 Judge-bias report (O7)** — three measurements, two of which ride on data already collected:

| Bias | Method | New work |
|---|---|---|
| **Position** | disagreement rate between (A,B) and (B,A) orderings | none — JudgeArena pairwise already scores both; persist both orderings and report the rate |
| **Length** | regress `P(chosen)` on the length delta between completions; report the **slope** | none — `P3-3` / JudgeArena LC win-rate already collect lengths + verdicts |
| **Self-preference** | judge Qwen-generated vs SmolLM2/OLMo-2 completions of matched quality; report how often it prefers its own family | **this is the gap** — needs non-Qwen completions on a shared prompt slice |

**A cheap third opinion: average log-probability as a scorer.** Alongside the 32B judge, score a slice
with the policy's own **average answer log-probability** — it needs no second model and no judge time.
Treat it as a *diagnostic*, not a replacement: published results show logprob scoring works as a
tiebreaker (44.8% vs 43.2% untied) but is actively worse than a heuristic when used as the primary
selector (48.4% vs 57.8%), because it measures how *expected* an answer looks to the model rather than
how *good* it is. Where logprob and judge rankings diverge sharply, that divergence is itself a signal
about the judge — and it costs nothing to compute from generations already on disk.

### Stages S8–S10 — the KL frontier

*The proxy RM is `rm_1.5B` from S2 and the gold signal is the S-series judge; neither is retrained.
Keeping the RM at **1 epoch** (§3.2) is deliberate — it must stay hackable for the right reason.*

**3.8 Best-of-N sweep (O8/O9/O12)** — BoN over ~1.5K held-out prompts the RM never trained on
(UF `test_prefs` + a Tulu prompt slice) at 8 values of N. Per N record **proxy score, gold score and
measured KL**; KL from the reference grows ≈ `log N`, which is what makes BoN a clean dial with no RL
instability.

> **BoN is chosen as an *optimisation dial*, not as the best selector — say so in the write-up.**
> At matched n, majority voting (self-consistency) outperforms best-of-N selection in published
> results (43.2% vs 40.6% with a heuristic scorer). BoN is used here anyway because its KL from the
> reference is *predictable* (≈ `log N`), which is exactly what an over-optimization sweep needs and
> what majority voting does not give. A reader who knows the selection literature will otherwise assume
> the choice was uninformed. Overlay the S4 PPO run on the same gold-vs-KL axes — that overlay *is* O9. Report every
gold curve **raw and length-controlled**, reusing §3.3's length control (O12).

**3.9 Mitigation arms (O10/O12)** — four short PPO reruns against the S4 baseline: (a) β=0.1;
(b) reward = **min over a 3-RM ensemble** (two extra 1.5B RMs on different subsamples/seeds);
(c) early-stop at a fixed KL budget; (d) reward whitening. Metric: gold peak height and the KL at
which it peaks.

**3.10 Qualitative degradation + steering (O11)** — symptom curves (mean length, stock-phrase rate,
sycophancy answer-flip rate) at each BoN-N and each PPO checkpoint. Extract a **sycophancy vector** by
contrastive activation analysis on the over-optimized model: contrastive means at the **last prompt
token** of the residual stream, sweeping ℓ over **middle layers**, `v_ℓ = mean(a | S⁺) − mean(a | S⁻)`.
Steer `h ← h + α·v` over **α ∈ {−2,…,+2}** and test whether negative-α steering restores gold score
without retraining. Also test the **activation-capping** variant, which clamps rather than translates
and tends to cost less capability:

```
h′ = h − v · min(⟨h, v⟩ − τ, 0)      τ = 25th percentile of training projections
```

**Flip-rate definition:** the fraction of *correct* answers abandoned after user pushback — ×3 at
temp 0.7, mean ± std. Watch for a **non-monotone / U-shaped** response in α; assuming monotonicity is
the most likely way to misread this. Guardrail: MMLU + IFEval, steered vs unsteered.

**3.11 Steering vs character training (O13, gated)** —
- **Steering arm:** the best (ℓ\*, α\*) from §3.10, including the capping variant.
- **Training arm:** the 5-principle constitution → the model critiques and revises **its own** SFT
  outputs (no human labels, no teacher) → DPO at **lr 5e-7, β 0.1, 1 epoch**, reusing `src/trainers/dpo.py`.
- **Compare on:** flip-rate, MMLU/IFEval cost, judged coherence, and **multi-turn durability** — does
  the fix decay by turn 8+? Summary metric: **capability damage per point of sycophancy fixed.**
- Expected: steering is free but decays multi-turn; training is durable but costs measurable
  MMLU/IFEval. Either direction is shippable.

**3.12 Preference-pair screening (O14, gated)** — projection-difference metric per DPO training pair
onto `v_syc`; rank and read the top 50 by eye. Optional lite ablation: **drop the top 2% and re-DPO**,
then re-measure flip-rate.

---

## 4. Experiments & metrics

| Exp | Compares | Primary metric | Secondary |
|---|---|---|---|
| E1 (O1) | base vs SFT vs DPO | judged win-rate (pos-swapped) | IFEval, MMLU, format compliance |
| E2 (O2) | DPO β=0.05 vs β=0.1 | KL vs win-rate | chosen/rejected log-prob curves (displacement) |
| E3 (O3) | DPO judged ± length control | Δwin-rate after length correction | mean length, markdown rate |
| E4 (O4) | DPO vs PPO, same prompts | judged win-rate vs SFT | KL spent, wall-clock |
| E5 (O5) | RS-SFT vs DPO on identical prompts | judged win-rate vs base, with CIs | mean length, IFEval |
| E6 (O6) | with vs without structured prefs | **IFEval strict accuracy** | judged win-rate, format compliance |
| E7 (O7) | judge behaviour itself | length-bias slope; self-preference rate | position-bias disagreement rate |
| E8 (O8) | proxy vs gold across BoN-N | gold score vs `log N` (inverted-U?) | proxy score, measured KL, length |
| E9 (O9) | BoN curve vs PPO curve | gold score at equal measured KL | wall-clock per curve, `clip_frac` |
| E10 (O10/O12) | PPO base vs β×2 vs ensemble vs early-stop vs whitening | gold peak height & KL at peak | MMLU/IFEval regression |
| E11 (O11) | KL vs symptoms; steering on/off | flip-rate vs KL; gold recovered by steering | steering-response shape, guardrails |
| E12 (O12) | raw vs length-controlled gold | share of decline explained by length | — |
| E13 (O13) *gated* | steering vs character-DPO | flip-rate fixed per MMLU point lost | multi-turn durability, judged coherence |
| E14 (O14) *gated* | top-projection pairs vs random | human-read sycophancy rate of flagged pairs | retrain-lite Δflip-rate |

All judged evals ×3 runs, mean ± std. An MMLU drop > 5 pts is a broken-run signal.

**E7 gates the headline.** Report the judge-bias numbers *alongside* the O4 DPO-vs-PPO verdict, not in
an appendix — a verdict produced by a judge with unquantified self-preference is exactly the fragile
leaderboard number this programme exists to critique.

Pre-train baselines for every init model are required before the corresponding train run (see
**Evaluate-before-train** above); E1–E4 build on those baselines rather than replacing them.

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
2. **Do the CUDA-only paths work?** vLLM generation and JudgeArena judging are only ever unit-tested
   against injected fakes on the dev machine, because vLLM has no macOS build.
3. **What is the real tokens/sec?** Every hour in §5 is an estimate. One measurement at 0.5B replaces
   them; scaling 0.5B → 1.5B is roughly **3× compute** (approximate — attention does not scale
   linearly with parameters).

**This project's Rung-1 checklist:**

| Check | Retires |
|---|---|
| PPO ~20 steps on `Qwen2.5-0.5B` | The TRL 0.19.1 `PPOTrainer` API surface `src/trainers/ppo.py` assumes, and the **weakest number in the budget** — PPO's hours are a guess |
| `src/eval/generate.py` under real vLLM | Incremental writes + skip-completed-on-restart |
| lm-eval with `model_backend: vllm` | That `batch_size="auto"` behaves on the `VLLM` class, and `_default_simple_evaluate` works against the real library |
| JudgeArena wrapper + 1.5B judge standing in for 32B | Pairwise protocol; judgments/sec to size the 32B line — the largest single eval cost |
| SFT → RM → DPO end-to-end at 0.5B | The Phase-5 smokes, ~11× cheaper per hour |

**Set `total_episodes` from the measured sec/step — NOT `max_steps`.** ⚠️ Corrected after Phase R:
`max_steps` does **not** bound a TRL 0.19 PPO run. `PPOTrainer` overwrites it
(`state.max_steps = num_total_batches`, `ppo_trainer.py:386`) and the loop iterates
`num_total_batches`, which derives from `total_episodes` alone. PPO was also never *unbounded* — TRL
falls back to `num_train_epochs × train_dataset_len`, so it stopped at 46 updates — but that bound was
emergent and silently coupled to `num_eval_prompts`. `configs/ppo.yaml` now sets `total_episodes: 1472`
explicitly. Separately, `save_steps: 50` exceeded the 46-update run and wrote **zero** checkpoints,
falsifying the resume guarantee below; it is now 10. Convert it to a
declared wall-clock budget once Rung 1 gives a real rate: with `save_steps: 50` and resume already
configured, a truncated PPO still yields a usable checkpoint and a reportable comparison.

---

## 5. Compute & cost

**Re-forecast from Phase-R measurements — see `REFORECAST.md`.** The four training lines were
measured at 0.5B on an RTX 3090 and are **~3× over-provisioned**; the *Meas.* column projects them to
1.5B/bf16/A100 (net ×0.5, range ×0.35–1.0). Two lines below changed **hardware**, not just hours: PPO
cannot run on a 24 GB card at any model size (TRL stacks `batch × response_length × vocab × 4` =
9.27 GiB of logits, independent of model size), and the 32B judge needs 80 GB of weights.

| Stage | GPU | ~Hours | Meas. |
|---|---|---|---|
| SFT (25K × 1 epoch, 1.5B) | 1× A100 80 GB | 3 | **2.0** |
| RM (20K pairs, 1 epoch) | 1× A100 80 GB | 3 | **1.9** |
| DPO ×2 β arms (10K pairs) | 1× A100 80 GB | 5 | **0.9** |
| PPO | 1× A100 80 GB | 8 | **1.3** |
| Evals (RewardBench, judge, lm-eval) | 1× RTX 4090 24 GB | 3 |
| *(S5/O5)* Teacher N=8 generation over the 10K DPO prompts | 1× A100 80 GB (vLLM) | 3 |
| *(S5/O5)* Judge selection pass — **14B**, tournament to top-1 | 1× A100 80 GB (vLLM) | 3 |
| *(S5/O5)* RS-SFT train + head-to-head vs DPO | 1× A100 80 GB | 2 |
| *(S6/O6)* Structured pairs: gen → DPO → IFEval | 1× RTX 4090 24 GB | 3 |
| *(S7/O7)* Self-preference: non-Qwen completions + judging | 1× RTX 4090 24 GB | 2 |
| *(S8)* BoN sweep — 8 N-values × 1.5K held-out prompts | 1× RTX 4090 24 GB (vLLM) | 7 |
| *(S8)* Gold scoring of all BoN selections (32B judge) | 1× A100 80 GB (vLLM) | 4 |
| *(S9)* 2 ensemble RMs + 4 short PPO mitigation arms | ~~1× RTX 4090 24 GB~~ **1× A100 80 GB** | 12 |
| *(S10/O11)* Symptom curves, vector extraction, steering sweep, guardrails | 1× RTX 4090 24 GB | 4 |
| *(gated O13)* Character data → character-DPO → multi-turn evals | 1× A100 + 1× 4090 | 6 |
| *(gated O14)* Projection ranking (near-pure inference) | 1× RTX 4090 24 GB | 2 |

- **Total:** ~22 GPU-hrs (S1–S4) **+ ~13** (S5–S7) **+ ~27** (S8–S10) = **~62**, plus ~8 gated.
  **Picks:** A100 80 GB for training and 32B/14B serving, RTX 4090 for generation and eval.
- **Estimated cost: ≈ £65 + £25 + £55 = ≈ £145**, plus ≈ £8 if the gated arms run.
  **Wall-clock: ~4–5 days.**
- **The proxy RM and the PPO baseline are not retrained for S8–S10** — S2's `rm_1.5B` is the proxy and
  S4's PPO run is the reference curve. That reuse is what makes the KL frontier cost ~27 hrs instead
  of ~37 standalone.
- ⚠️ **These rates are stale.** The table assumes A100 ≈ £1/hr and RTX 4090 ≈ £0.45/hr; live RunPod
  *secure-cloud* rates are **A100 SXM £1.25 / A100 PCIe £1.09 / 4090 £0.58**, ~25–30% higher, and
  network volumes cannot use community pricing. Repriced, core ≈ **£72** (a bottom-up estimate for the
  full P5–P8 run came to **$92 ≈ £72**, within 11% of the £65 here). Also **no volume-capable
  datacenter currently stocks both 4090 and A100**, so this may collapse to one A100-class pod. See
  [`../PORTFOLIO.md`](../PORTFOLIO.md) → **Budget reality**.
- **S5 is the expensive one** (~8 of the 13 hrs) — judging N=8 is ~7 comparisons/prompt.
  Cut order if budget tightens: halve the RS prompt pool (10K → 5K) before dropping S6 or S7. **Never
  drop S7** — it is only ~2 hrs and it underwrites the O4 verdict.

---

## 6. Repo layout

```
tulu-postraining-pipeline/
├── README.md                      # findings summary
├── pyproject.toml                 # package metadata + pinned deps
├── requirements.txt               # trl, vllm, lm-eval, judgearena, reward-bench, wandb, datasets
├── .env.example                   # WANDB_API_KEY, HF_HOME=/workspace/hf, HF_XET_HIGH_PERFORMANCE=1
├── .gitignore                     # data/, results/checkpoints/, .env, .git-credentials, *.log
├── configs/                       # one YAML per stage
│   ├── sft.yaml
│   ├── rm.yaml
│   ├── dpo.yaml
│   ├── ppo.yaml
│   └── eval.yaml
├── src/                           # importable packages/modules (pythonpath)
│   ├── hub.py
│   ├── resume.py
│   ├── trainers/                  # sft.py / rm.py / dpo.py / ppo.py
│   ├── prepare/                   # sft.py / rm.py / dpo.py / ppo.py + decontam
│   ├── data_tools/
│   ├── eval/                      # vllm gen, judge, skills, rewardbench-chat gate
│   └── analysis/                  # stage attribution, length, KL, verdict
├── scripts/
│   ├── setup_env.sh               # wandb login + HF_HOME exports
│   ├── prepare/                   # sft.py / rm.py / dpo.py / ppo.py
│   ├── train/                     # sft.py / rm.py / dpo.py / ppo.py
│   ├── eval/                      # skills.py / rb_gate.py / head_to_head.py / style.py
│   └── analysis/                  # attribution / beta_plots / displacement / chattiness / verdict
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
└── tests/
    ├── eval/                      # gen, judge, skills, style, rb-chat gate
    ├── analysis/                  # attribution, beta plots, chattiness, verdict
    └── train/                     # sft / rm / dpo / ppo smoke
```

---

## 7. Deliverables & acceptance criteria

- **Deliverables:** the SFT / RM / DPO(×β) checkpoints + a stage-attribution report — (1) what SFT vs
  DPO each added (format/skills/style table), (2) β-sensitivity + displacement curves, (3) chattiness
  report (raw vs length-controlled win-rate), (4) DPO-vs-PPO verdict.
- **Data-engine deliverables** — (5) the **released decontaminated
  Tülu-mini SFT set + preference set** on HF: these are the existing `P1-4`/`P1-5`/`P1-6` prep outputs,
  so this is a *publish* step, not a rebuild; (6) **RS-SFT-vs-DPO verdict** with CIs on identical
  prompts; (7) **structured-preference IFEval gain**; (8) **judge-bias report** (length slope /
  self-preference rate / position disagreement).
- **Acceptance:** every train stage has a pre-train baseline eval on disk/W&B for its init model;
  RM ≥ 65–70% on RewardBench chat; DPO beats SFT > 55% under the judge; a measured
  length effect reported both raw and length-controlled.
- **Acceptance (O5–O7):** a signed O5 verdict with CIs; structured prefs improve IFEval by a visible
  margin (a null is a valid finding, an *unmeasured* one is not); a **non-zero judge length-bias slope
  and a stated self-preference rate, published next to the O4 verdict.**
- **KL-frontier deliverables** — (9) proxy/gold vs `log N` **inverted-U figure**, (10) BoN-vs-PPO
  gold-per-KL overlay, (11) mitigation table, (12) symptom-vs-KL curves + steering-recovery result,
  (13) raw vs length-controlled gold curves, (14) a **"hacking specimen gallery"** of N=128 outputs,
  (15) the released **sycophancy steering vector**, and (16) a reusable **"how far can I push this RL
  run" KL-frontier utility** — the thing every other bet adopts to bound its RL runs.
- **Acceptance (O8–O12):** proxy climbs monotonically while gold plateaus or declines; PPO shown to
  spend KL less efficiently than BoN; ≥1 mitigation measurably shifts the peak; sycophancy flip-rate
  grows with KL and negative-α steering reduces it without collapsing MMLU. **At 1.5B the O8 result may
  be a plateau rather than a decline — pre-register that reading** so a shallow curve is reported as a
  finding, not massaged into one.
- **Gated (O13/O14):** the steering-vs-character-DPO head-to-head and the pair ranking. **Neither is
  required for acceptance** — a documented "steering does not work at 1.5B" from O11 closes the line.

Optional extensions (native): an on-policy DPO variant (regenerate pairs from `model_SFT`); a
margin-loss RM ablation.

---

## 8. Risks & mitigations

- **RM: 1 epoch, no exceptions** — an overfit RM is hackable for the wrong reasons and poisons DPO.
- Keep the RM and DPO pair sets **disjoint**, or DPO evaluation is leaky.
- Chat-template bugs are silent — render and eyeball 5 examples before any run.
- Judge win-rates without length control flatter DPO — always report both.
