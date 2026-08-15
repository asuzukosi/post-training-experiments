# RLVR on Math with GRPO

Engineering spec for a self-contained experiment: reinforcement learning with **verifiable** rewards
on math — no reward model, just a checker — using GRPO. Reproduce the central data lever (difficulty
filtering), test dropping the KL penalty, and run the essential control: RL with random rewards.

---

## Bet & success target

Serves **Bet 2 — SOTA-for-size reasoning, proven clean.**
- **Thesis:** cheap RLVR gives `Qwen2.5-1.5B-base` a *real* GSM8K/MATH pass@1 lift — verified clean by a random-reward control.
- **Headline target:** **GSM8K pass@1 +≥10 pts over the strongest inference-time-scaled base at matched inference compute** (aim +10–20), **pass@8 ~flat**, MATH-500 lift reported, and the **random-reward run ≈ 0** (Δ within noise). All numbers with ±std over 3 seeds.
- **The baseline is the claim.** Beating *greedy, no-CoT* decoding is not a result — published MATH-500 numbers show chain-of-thought prompting alone lifts a small base model **15.2% → 40.6%**, and self-consistency (n=10) + top-p + CoT reaches **52.0%**, *above* a GRPO-trained model at 47.4%. Any RL gain measured against naive greedy decoding is measuring a handicapped opponent. See **O10**.
- **Artifact to ship:** the RL-tuned model on HF (`qwen2.5-1.5b_grpo-...`) + a documented contamination-control recipe.
- **Win / lose-but-ship:** clean ≥+10 pts with a null random-reward control = win. If random reward *also* moves GSM8K, that's a publishable **contamination finding** on Qwen bases — still a shipped result.

---

## 0. Overview

In verifiable domains (math, code) the reward is a deterministic checker, so no learned reward model
is needed. The practical lever for RLVR is **data curation**, not the optimizer: prompts a model
solves 0% or 100% of the time yield zero policy gradient, so the useful signal lives in the ~20–80%
pass-rate band. A known confound is that some base models gain benchmark points from **random** RL
rewards — pure contamination signal — so no RLVR result on such a base is credible without a
random-reward control. This project runs the recipe and its audit together.

---

## 1. Objectives

1. **O1 — New capability or concentrated mass?** Does GRPO with a binary verifier raise GSM8K
   **pass@1** while **pass@8** stays flat?
2. **O2 — Difficulty filtering:** Does training only on the 20–80% pass-rate band beat unfiltered
   training at equal compute?
3. **O3 — KL off vs on:** β=0 vs β=0.01 — does removing the KL penalty buy exploration on longer runs?
4. **O4 — The control:** Does a coin-flip reward also "improve" GSM8K on this base? If so, that's
   contamination, not learning.

**The trusted-number harness (O5–O9)** — this project also builds the measurement layer the whole
programme reports through. A headline "+15 pts" is meaningless without knowing the benchmark's own
run-to-run noise, how much it moves on prompt format alone, and whether the training data leaked.

5. **O5 — Variance:** Measure run-to-run benchmark variance at 1.5B with temperature > 0; produce a
   per-benchmark std-dev table and a stability ranking. Extend with **`avg@k`** on a tiny AIME-style
   set, since reasoning-style evals carry the largest sampling variance. **Also sweep eval-set size**
   — published MATH-500 numbers move 30% → 34% → 27% → **15.3%** at n = 10 / 50 / 100 / 500 for the
   same base model. Small slices are wildly optimistic *and* non-monotone, so any `--limit` result is
   a smoke test, never a number.
6. **O6 — Format fragility:** Quantify how much scores move on prompting format alone — few-shot vs
   zero-shot vs CoT, answer-extraction suffix, chat-template choice.
7. **O7 — Contamination scan:** n-gram / substring overlap between training corpora and eval sets;
   report hit counts with example matches. CPU work, no GPU.
8. **O8 — Perturbation probe:** accuracy on original vs **number-swapped** GSM8K. Treat a large delta
   as *suggestive* of memorised formats — not proof.
9. **O9 — Harness deltas:** score the **same checkpoint and tasks** through **two** harnesses —
   `lm-eval` (primary, published-comparable) and **Inspect AI** (second opinion). Quantify the
   disagreement and log every config difference that explains it. No LLM judge; both use
   exact/math extraction, not a pairwise scorer. **Drop LightEval** — a third HF-style harness
   adds another default-config soup without a new question.
9b. **O9b — Verifier audit:** hand-implement a second verifier (boxed extraction → normalisation →
   equivalence → grading) and run it **differentially** against `math_verify` on real model outputs.
   Report the disagreement rate and, specifically, the **false-negative rate** — correct answers scored
   0. CPU-only, no GPU.

> **In RLVR the verifier *is* the reward function.** There is no reward model, so a verifier bug is a
> corrupted gradient, not a mis-reported score: a false negative actively teaches the policy that a
> correct answer was wrong, and it does so silently. O9 already applies exactly this logic one level up
> — **lm-eval vs Inspect** on the same model/tasks — and the verifier sits *below* both while receiving
> none of that scrutiny. `math_verify` stays the production reward (maintained, used by `lm-eval`, and
> what keeps our numbers comparable to published work); the hand-rolled version exists to measure it.
> It also makes **O8** interpretable: a number-swap delta cannot be read as memorisation if part of it
> is normalisation behaviour.

> **Validation loss and task accuracy can move in opposite directions.** In published distillation
> runs, dropping `<think>` tokens *lowered* validation loss while *lowering* MATH-500 accuracy in
> almost every case. Never select a checkpoint on loss alone — every checkpoint decision in this
> project is made on the task metric.

> **O4 and the spurious-rewards probe are the same experiment.** The random-reward control is not a
> footnote to the GRPO result — it is what makes the headline believable, and it doubles as the
> contamination signal for the harness. **Never cut it**, whatever else goes.

**Inference-time scaling (O10–O11)** — the counterfactual the headline stands or falls on. Training is
not the only way to buy reasoning accuracy, and at small scale it is often not the cheapest.

10. **O10 — The honest baseline:** build the **strongest** base-model baseline before claiming any RL
    gain — greedy → chain-of-thought prompting → temperature + top-p → **self-consistency (majority
    vote) at n ∈ {3, 5, 10}** → CoT + self-consistency combined. Report GRPO against *that*, not
    against greedy decoding.
11. **O11 — Accuracy per unit of inference compute:** self-consistency at n=10 costs roughly **85×**
    the inference time of greedy decoding, so raw accuracy comparisons are not like-for-like. Report
    every method as **accuracy vs measured inference cost**, and state the crossover: below some
    inference budget prompting wins, above it the trained model wins. **That crossover is the real
    finding** — more useful than either number alone.

> **Why this is not optional.** Published MATH-500 results for a small base model: greedy **15.2%**,
> CoT prompting alone **40.6%**, self-consistency(10) + top-p + CoT **52.0%** — against **47.4%** for a
> GRPO-trained model and 48.2% for a reference reasoning model. **Inference-time scaling beat the
> trained model.** If our RL result is reported against greedy decoding, the first reviewer to ask
> "did you try prompting?" ends the conversation. O10 is pure inference — no training, cheap — and it
> converts the weakest point of Bet 2 into one of its strongest.

**Verifier fidelity (O12)** — how good does the checker actually have to be?

12. **O12 — Verifier fidelity sweep:** wrap the real verifier and corrupt its output at a controlled
    rate, then sweep **verifier accuracy p ∈ {100, 95, 90, 80, 65, 50}%** and measure GSM8K pass@1 at
    each. **This subsumes O4:** a binary verifier at p = 50% *is* the coin-flip reward, so the
    random-reward control becomes the endpoint of a curve rather than an isolated point.

    Sweep the two fault modes **separately**, because they are not symmetric:

    | Mode | Corruption | Effect on the policy |
    |---|---|---|
    | **False negative** | correct answer scored **0** | removes signal and punishes correct behaviour — closer to label noise |
    | **False positive** | wrong answer scored **1** | actively rewards wrong behaviour — the reward-hacking direction |

    Expect false positives to do more damage than false negatives at equal rate. Report the
    **degradation curve and the knee** — the accuracy below which RLVR stops working at this scale.

> **The GRPO-specific trap this exposes.** Advantages are group-mean-centred, so a group where every
> rollout scores the same contributes **zero gradient**. A faulty verifier *breaks those ties*: groups
> that were uniformly-correct or uniformly-wrong become mixed, producing **more** non-zero advantages
> and **more** apparent gradient — while the gradient points at noise. A degraded verifier can
> therefore make a run look *healthier* on training curves than a correct one. Track **% zero-advantage
> groups** at every p alongside accuracy; if that fraction falls as p falls, this effect is live.

> **Why this pairs with O9b.** The audit measures the false-negative rate our *real* verifier actually
> has; the sweep measures how much we can tolerate. Neither answers "is our verifier good enough?"
> alone — together they do, and the answer is a number rather than a judgement call.

**Expected outcomes:** pass@1 +10–20 pts with pass@8 ~flat; filtered ≈ unfiltered at roughly half the
data (a compute win); β=0 slightly better with stable format rewards; random-reward gains small but
possibly non-zero on the base — any gain triggers the contamination reading. Reasoning and
instruction-following evals show the largest variance and knowledge MCQ the smallest; **format swings
exceed run-to-run variance**; the n-gram scan finds real overlaps in public instruction data.

---

## 2. Environment & prerequisites

**Compute:** RunPod. Network volume **`posttraining-data`** — **deleted 2026-08-12; it does not
exist.** Recreate it (250 GB, US-CA-2) *only* when starting a full-scale run — see
[`../VOLUME.md`](../VOLUME.md) for the exact create + repopulate + re-secrets recipe. **Rung 1 (§4b)
needs no volume.** Once created: cache at `/workspace/hf`.

**Launch a pod** (A100 80 GB for GRPO training; RTX 4090 for filtering and pass@k evals):
```bash
runpodctl pod create --name rlvr-grpo \
  --gpu-id "NVIDIA A100-SXM4-80GB" --gpu-count 1 \
  --data-center-ids US-CA-2 --cloud-type SECURE \
  --network-volume-id <NEW_VOLUME_ID> --volume-mount-path /workspace \
  --container-disk-in-gb 30 --ports "22/tcp" --ssh \
  --image "pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel" \
  --terminate-after "$(date -u -v+8H +%Y-%m-%dT%H:%M:%SZ)"
```
On every pod: `export HF_HOME=/workspace/hf HF_XET_HIGH_PERFORMANCE=1`.

> **Python 3.12 is required** — `judgearena` needs `>=3.12`, so the `py3.11` RunPod image will not
> install the stack. Create the env first: `conda create -y -n py312 python=3.12`, then use
> `/opt/conda/envs/py312/bin/python`. Verified on 3.12.13 / CUDA 12.4.

**Constraint:** open-weights only — no gated models.

**Models & reward:**

| Role | Choice | Notes |
|---|---|---|
| Policy | `Qwen/Qwen2.5-1.5B` (base) | cached; enough latent capability for reasoning to emerge |
| Verifier | `math_verify` (pip), audited against a hand-written second implementation | reward = 1 if correct else 0 |
| Format reward | +0.1 for well-formed `<think>…</think>` | encourages the reasoning structure |

**Rollout budget is a first-class hyperparameter, not a footnote.** Published GRPO results at fixed
step count: **43.2%** (256 max tokens, 4 rollouts) → **45.6%** (512, 4) → **47.4%** (512, 8). A 4-point
swing from the generation budget alone, larger than most ablations in this spec. Set the training
rollout token cap explicitly and record it with every result; if the cap truncates before the boxed
answer, the rollout earns no reward and the signal is silently lost. Expect response length to *grow*
during training, so a tight cap becomes tighter as the run proceeds.

**Reward shaping is a bigger lever than the KL ablation.** Strict binary reward scores **47.4%** where
a partial-credit variant scores **37.8%** — a 9.6-point drop. This spec uses strict 1/0 by design;
do not quietly relax it to "help the model learn".

**Data (cached):** `openai/gsm8k` train (~7.5K problems) + test; `HuggingFaceH4/MATH-500` for an
optional harder OOD check. Difficulty filter: sample N=16/prompt at temp 1.0, keep pass-rate ∈ [0.2, 0.8].

**Notes:** `math_verify` installs via pip on the pod. Volumes attach only to secure-cloud pods in
US-CA-2; use the `hf` CLI (not `huggingface-cli`); delete the pod when done.

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

**Output naming.** Every trained checkpoint (and its W&B run) is named
`<base_name>_<task>_<dateandtime>` — `<base_name>` = short base id (`qwen2.5-1.5b`), `<task>` = the
run/arm, `<dateandtime>` = filesystem-safe UTC stamp `YYYYMMDDThhmmZ`. Save under
`results/checkpoints/`. Examples: `qwen2.5-1.5b_grpo_20260809T1730Z`,
`qwen2.5-1.5b_grpo-unfiltered_...`, `qwen2.5-1.5b_grpo-nokl_...`, `qwen2.5-1.5b_grpo-random_...`.

**Checkpointing & resume.** GRPO runs are long, so checkpoint frequently to the volume — a failed or
killed node loses nothing: launch a new pod, re-mount the volume, resume. In the trainer config:
`output_dir=results/checkpoints/<run>`, `save_strategy="steps"`, `save_steps=25–50` (RL steps are
expensive — save often), `save_total_limit=3`, launch with `resume_from_checkpoint=True` (restores
policy + optimizer + step). Fix the W&B run id and set `resume="allow"` so reward/length/KL curves
continue on the same run. The difficulty-filtering pass writes per-prompt pass-rates **incrementally**
and **skips already-sampled prompts** on restart. Because `output_dir` is on the network volume
(persists across pods), resume survives full node loss — so cheaper interruptible/spot GPUs are viable.

**Conventions.** Pin a known-good dependency set in `requirements.txt` (TRL ↔ transformers ↔ vLLM are
version-sensitive, especially the vLLM-backed GRPO path) and install from it — never `pip install -U`
mid-project. **Smoke-test** the GRPO loop on a few dozen prompts before the full run. **Evaluation
tooling:** **primary** = `lm-eval` (`gsm8k` strict-match; `minerva_math` optional) + `math-verify`
for pass@1 / pass@8 / maj@8. **O9 second harness** = **Inspect AI** on the same checkpoint, same
splits, local vLLM/HF (no API): GSM8K + MATH via Inspect tasks/scorers (`match` / math extraction,
not `model_graded_*`). Log few-shot, extraction regex, chat template, stop tokens for both.
**No LLM judge in this project.** Headlines are always the lm-eval number; Inspect is the delta.

---

## 3. Approach

```
GSM8K train (7.5K)
   │  S1: difficulty filter — N=16 samples/prompt, keep pass-rate ∈ [0.2, 0.8]  (~40–60% kept)
   ▼
filtered prompt set ──► also keep a random equal-size UNFILTERED control set (O2)
   │
   ├─► S2: GRPO main run — binary verifier + format reward, β_KL = 0.01
   ├─► S3: GRPO unfiltered-control run (equal steps/compute)                       (O2)
   ├─► S4: GRPO β_KL = 0 run                                                       (O3)
   └─► S5: GRPO random-reward run (~200 steps)                                     (O4)
              │
              ▼
        eval: pass@1 vs pass@8 vs maj@8 (vLLM), temp 0.7, ×3 runs
```

**Tooling:** `TRL GRPOTrainer` with `vLLM` generation (`use_vllm=True`, colocate mode);
`lm-evaluation-harness` (primary GSM8K/MATH); **Inspect AI** (O9 second harness, local vLLM/HF);
`math-verify`; `W&B` (reward, completion-length, format-compliance, KL curves).

**3.1 Difficulty filtering** — vLLM batch generation, N=16, temp 1.0, max 768 tokens. Persist
per-prompt pass-rates; the histogram is a deliverable (where does a 1.5B base sit?).

**3.2 GRPO main run** — 8–16 generations/prompt, group-relative advantage, lr 1e-6, clip 0.2, max
completion 768, ~300–500 steps. Watch the reward curve, **completion-length growth** (the
inference-time-scaling signature), and format compliance → ~100%.

**3.3 Ablations** — unfiltered control (identical config, random prompts, equal compute); β_KL=0
(watch for entropy collapse vs extra exploration); random-reward run (reward = fair coin, ~200 steps,
eval before/after).

**3.4 Optional cold-start loop** — rejection-sample the RL checkpoint (keep only correct chains) → SFT
one epoch → re-eval: one turn of the R1-style loop.

**3.5 Trusted-number harness (O5–O9)** — built alongside the RL runs and applied to every number this
project reports.

- **Variance (O5)** — 3 runs per benchmark at temp 0.7, plus 1 run at temp 0 for reference; vary one
  infra knob (vLLM tensor-parallel / batch size) to check numeric sensitivity. Add `avg@k` on a small
  AIME-style set. Output: benchmark, mean, std-dev, stability category.
- **Format sensitivity (O6)** — MMLU under {0-shot, 5-shot, 0-shot-CoT, 5-shot-CoT with an
  "answer is (X)" suffix, alternate chat template}; GSM8K under strict vs flexible extraction. Report
  the full per-benchmark swing.
- **Contamination scan (O7)** — 8-gram word overlap + 50-char substring matching between each training
  corpus and each eval's prompts; report hit counts plus example matches. CPU-only.
- **Perturbation probe (O8)** — accuracy on original vs number-swapped GSM8K.
- **Harness bake-off (O9)** — identical checkpoint + GSM8K (and MATH-500 if scored) through
  **`lm-eval` then Inspect**. Inspect run: `inspect eval` with a local model
  (`vllm/Qwen/Qwen2.5-1.5B` or the GRPO ckpt; `base_url` only if a vLLM server is already up —
  prefer in-process / the same backend as generation). **Do not** use Inspect `model_graded_qa`.
  Log every config delta (few-shot count, extraction regex, system prompt, stop tokens) and deliver
  the two-harness delta table *with the reasons*, not just the numbers. If the gap exceeds O5
  run-to-run std, the headline must name which harness and why.
- **The protocol** — codify "every sampled eval → ×3 runs, report mean ± std" as a checklist, then
  re-score candidate checkpoints and report **which pairwise 'wins' survive once error bars are
  attached**. That checklist is the reusable artifact other bets adopt.

**3.6 Inference-time scaling ladder (O10/O11)** — pure inference on the **base** model, run before the
GRPO result is written up. Each rung is a strictly stronger opponent than the last:

| Rung | Method | Purpose |
|---|---|---|
| 0 | greedy decoding | the naive baseline most papers quote |
| 1 | + chain-of-thought prompting | the single largest free gain; expect a very large jump |
| 2 | + temperature & top-p sampling | enables the sampling the next rung needs |
| 3 | + **self-consistency** (majority vote) at n ∈ {3, 5, 10} | the compute-scaling axis |
| 4 | CoT + top-p + self-consistency combined | the strongest cheap baseline |

**Record wall-clock and total generated tokens for every rung**, not just accuracy — O11 needs the
cost axis, and self-consistency gains are bought with a large multiple of inference compute.

**Two traps this ladder exposes.** Self-consistency is **not monotone in n** — published numbers go
29.6% (n=3) → 27.8% (n=5) → 31.6% (n=10), so a single n can mislead in either direction; run all
three. And **majority voting beats best-of-N scoring** at matched n (43.2% vs 40.6% with a heuristic
scorer), which is worth knowing before spending judge time on selection.

**Tie-breaking is a real knob:** with self-consistency at n=3, no tiebreak gives 43.2%, a heuristic
tiebreak 43.4%, and average-logprob tiebreak 44.8%. Small, but free — pick one and state it.

---

## 4. Experiments & metrics

| Exp | Compares | Primary metric | Secondary |
|---|---|---|---|
| E1 (O1) | base vs GRPO | GSM8K pass@1 | pass@8, maj@8, completion length |
| E2 (O2) | filtered vs unfiltered, equal compute | pass@1 per GPU-hour | % non-zero-advantage groups |
| E3 (O3) | β=0.01 vs β=0 | pass@1, KL-from-base | entropy, format stability |
| E4 (O4) | real vs random reward | ΔGSM8K under noise | verdict: learning vs leakage |
| E5 (O5) | runs × benchmarks | per-benchmark std-dev @ temp 0.7 | temp-0 reference, infra-knob delta, `avg@k` |
| E6 (O6) | prompt variants | max score swing per benchmark | which variant each harness assumes |
| E7 (O7/O8) | train vs eval corpora; original vs number-swapped | n-gram hit rate | perturbation delta |
| E8 (O9) | lm-eval vs Inspect (same ckpt, same tasks) | absolute score deltas per task | config-diff log |
| E9 | candidate checkpoints under the protocol | fraction of "wins" surviving CIs | the checklist itself |
| E10 (O10) | greedy vs CoT vs top-p vs self-consistency n∈{3,5,10} vs combined, all on **base** | pass@1 per rung | non-monotonicity in n; tie-break choice |
| E11 (O11) | **GRPO vs the best rung, at matched inference compute** | accuracy vs wall-clock and generated tokens | **the crossover budget** — where training starts to win |
| E12 (O9b) | hand-rolled verifier vs `math_verify` on real outputs | disagreement rate; **false-negative rate** | which normalisation step causes each disagreement |
| E13 (O12) | GRPO at verifier accuracy p ∈ {100,95,90,80,65,50}%, FN and FP modes swept separately | pass@1 vs p — **the degradation knee** | **% zero-advantage groups vs p**; reward-hacking signatures at low p |

All evals ×3 at temp 0.7; report mean ± std.

**E5 is the denominator for E1.** The headline "+10–20 pts pass@1" is only meaningful against GSM8K's
measured run-to-run std-dev — report them together, and if the format swing in E6 exceeds the training
gain, say so plainly.

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
2. **Do the CUDA-only paths work?** vLLM generation, lm-eval, and Inspect are only ever unit-tested
   against injected fakes on the dev machine, because vLLM has no macOS build.
3. **What is the real tokens/sec?** Every hour in §5 is an estimate. One measurement at 0.5B replaces
   them; scaling 0.5B → 1.5B is roughly **3× compute** (approximate — attention does not scale
   linearly with parameters).

**This project's Rung-1 checklist:**

| Check | Retires |
|---|---|
| GRPO ~20 steps on `Qwen2.5-0.5B` with a rule-based verifier | The TRL 0.19.1 `GRPOTrainer` API and group-relative advantage wiring; real sec/step for the 10-hour main run |
| The **random-reward control** at toy scale | What "no signal" actually looks like on this stack — run it before the real one so a null is recognisable rather than assumed |
| Difficulty-filtering generation (N=16) under vLLM | Throughput for the 7.5K-prompt pass-rate sweep |
| lm-eval ×3 at temp 0.7 on a `--limit` slice | The variance protocol itself — the harness must work before it certifies anything |
| Inspect AI on the **same** `--limit` slice + ckpt | O9 second harness imports; produce a stub delta vs lm-eval and a config-diff log |

**The variance protocol is testable at 0.5B and worth testing there.** O5's whole claim is that
benchmarks are noisier than reported; you can observe that on a small model in minutes, and it
calibrates how many runs the real sweep actually needs.

---

## 5. Compute & cost

| Stage | GPU | ~Hours |
|---|---|---|
| Difficulty filtering (N=16 × 7.5K prompts) | 1× RTX 4090 24 GB (vLLM) | 4 |
| GRPO main run | 1× A100 80 GB | 10 |
| Unfiltered control | 1× A100 80 GB | 8 |
| β=0 ablation | 1× A100 80 GB | 10 |
| Random-reward control | 1× A100 80 GB | 3 |
| pass@k / maj@k evals | 1× RTX 4090 24 GB | 3 |
| Variance + format sweeps (benchmarks × variants × 3 runs) | 1× RTX 4090 24 GB | 6 |
| Perturbation probe + harness bake-off + `avg@k` | 1× RTX 4090 24 GB | 4 |
| Contamination n-gram scan | CPU / laptop | 2 |
| Verifier differential audit (O9b) | CPU / laptop | 2 |
| Verifier fidelity sweep (O12) — short runs, 0.5B in Rung 1 then 2–3 points at 1.5B | 1× cheap GPU + 1× A100 | 9 |
| Inference-time scaling ladder on base (5 rungs, self-consistency n=3/5/10) | 1× RTX 4090 24 GB (vLLM) | 5 |

- **Total:** ~62 GPU-hrs (~33–40 of it training, depending on ablations kept) + ~2 CPU-hrs.
  **Picks:** A100 80 GB for GRPO, RTX 4090 for generation and the harness sweeps.
- **Estimated cost: ≈ £140** incl. one rerun. **Wall-clock: ~3–4 days.**
- **Run the O12 sweep at 0.5B on a cheap GPU first.** The *shape* of the degradation curve is the
  finding, and shape is far cheaper to establish at 0.5B than at 1.5B; only the knee needs confirming
  at full scale (2–3 points). The full six-point sweep at 1.5B is not worth its cost.
- **The inference-time ladder (O10) is the cheapest insurance in the programme** — ~5 GPU-hrs of pure
  inference, no training, and without it the headline is contestable in one sentence. Self-consistency
  dominates that budget: n=10 costs ~85× greedy per prompt, so run the ladder on a **fixed subset** and
  state the subset size (see the eval-size warning in §4).
- ⚠️ **These rates are stale.** The table assumes A100 ≈ £1/hr and RTX 4090 ≈ £0.45/hr; live RunPod
  *secure-cloud* rates are **A100 SXM £1.25 / A100 PCIe £1.09 / 4090 £0.58**, ~25–30% higher, and
  network volumes cannot use community pricing. See [`../PORTFOLIO.md`](../PORTFOLIO.md) → **Budget**.
- **The random-reward control is not a separate line item** — the O4 control *is* the spurious-rewards
  contamination probe. It is costed once, at 3 hrs, and serves both.
- **Cut order if budget tightens:** the unfiltered control first, then the β=0 ablation, then the
  harness bake-off (O9). **Never the random-reward control** — it is what makes E1 believable — and
  never the ×3 variance runs, which are what make every other number readable.

---

## 6. Repo layout

```
rlvr-grpo-math/
├── README.md                      # findings summary
├── pyproject.toml                 # package metadata + pinned deps
├── requirements.txt               # trl (GRPOTrainer), vllm, math-verify, lm-eval, inspect-ai, wandb
├── .env.example                   # WANDB_API_KEY, HF_HOME=/workspace/hf, HF_XET_HIGH_PERFORMANCE=1
├── .gitignore                     # data/, results/checkpoints/, .env, .git-credentials, *.log
├── configs/                       # one YAML per run/arm
│   ├── filter.yaml
│   ├── grpo.yaml
│   ├── grpo_nokl.yaml
│   ├── grpo_random.yaml
│   └── eval.yaml
├── src/
│   └── rlvr/                      # importable package
│       ├── __init__.py
│       ├── difficulty_filter.py   # N=16 pass-rate band
│       ├── verifier.py            # boxed extraction + format reward
│       ├── grpo.py                # main + control + ablation arms
│       ├── eval.py                # pass@1 / pass@8 / maj@8 (lm-eval primary)
│       ├── inspect_eval.py        # O9: Inspect AI wrapper, same ckpt/tasks, local vLLM/HF
│       └── coldstart.py           # rejection-sample → SFT loop (optional)
├── scripts/                       # thin CLI entry points
│   ├── setup_env.sh               # wandb login + HF_HOME exports
│   ├── run_filter.py
│   ├── train_grpo.py              # --arm {main,unfiltered,nokl,random}
│   ├── run_eval.py
│   └── run_coldstart.py
├── data/                          # gitignored: raw/ + processed/
├── results/
│   ├── checkpoints/               # <base>_<task>_<datetime>
│   ├── metrics/
│   └── plots/
└── tests/                         # smoke tests
```

---

## 7. Deliverables & acceptance criteria

- **Deliverables:** (1) pass-rate histogram & filtered dataset, (2) reward / length / format curves,
  (3) pass@1-vs-pass@8 table (taught vs concentrated), (4) difficulty-filtering compute verdict,
  (5) **random-reward verdict**, (6) KL ablation note.
- **Verifier-fidelity deliverable:** (16) the **degradation curve** — pass@1 vs verifier accuracy for
  both fault modes, with the knee named and the zero-advantage fraction overlaid. Read together with
  (15), this states whether the production verifier is comfortably inside tolerance or uncomfortably
  near the knee.
- **Verifier deliverable:** (15) the **verifier audit** — disagreement and false-negative rates against
  `math_verify`, with the normalisation step responsible for each class of disagreement. A false-negative
  rate above a few percent is a training-signal defect, not an eval nitpick, and must be fixed before
  the GRPO runs rather than reported after them.
- **Inference-time deliverables:** (13) the **scaling ladder table** — accuracy and cost for greedy /
  CoT / top-p / self-consistency n∈{3,5,10} / combined, on the base model; (14) the **GRPO-vs-best-rung
  comparison at matched inference compute**, naming the crossover budget.
- **Harness deliverables:** (7) the **released trusted-number harness** — ×3 variance protocol,
  8-gram decontamination script, random-reward probe, format-sensitivity sweep — packaged as a small
  tool the other bets adopt; (8) a per-benchmark **variance + stability table** with ≥1 "high-variance"
  and ≥1 "very-stable" benchmark named; (9) the format-swing table; (10) the contamination hit report
  with example matches; (11) the **lm-eval vs Inspect** delta table **with the config reasons**; (12) the
  trusted-number checklist and which candidate "wins" survive error bars.
- **Inference-time acceptance:** the ladder is run **before** the GRPO write-up, and the headline is
  stated against the **best rung**, not greedy. If GRPO does not beat the best cheap baseline, that is
  a publishable finding about small-model RL — report it, do not requote against greedy.
- **Acceptance:** +10–20 pts pass@1 over the best inference-time-scaled baseline with pass@8 ~flat; format compliance ≈ 100%; an
  unambiguous yes/no on noise-reward gains; all numbers with std-devs.
- **Harness acceptance:** the variance table distinguishes stable from fragile benchmarks; the format
  swing is quantified against that variance; the n-gram scan reports real overlaps or a credible null;
  and **every headline number in this project is reported through the harness it ships** — the harness
  failing its own project is the one outcome that voids it.

Optional extensions (native): DAPO clip-higher + dynamic sampling (drop all-0/all-1 groups); the
cold-start loop; a MATH-500 hard-subset transfer check.

---

## 8. Risks & mitigations

- **Truncated answers corrupt reward** — penalize/filter overlong generations; score only
  EOS-terminated completions.
- All-correct or all-wrong groups give **zero GRPO advantage** — that's why the filter exists; log
  the % of degenerate groups per batch.
- Verifier brittleness: extraction-regex failures masquerade as wrong answers — audit 20 failures by
  hand before trusting the reward curve.
- A random-reward gain indicts the **base model's data**, not the trainer — report it as a
  contamination finding and caveat E1 accordingly.
- Length growth ≠ reasoning growth — only report length alongside pass@k, never alone.
