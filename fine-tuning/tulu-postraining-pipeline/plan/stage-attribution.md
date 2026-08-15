# Stage attribution

**Question.** What does each training stage add — format, skills, or judged quality?

Compares base vs SFT vs DPO on the same three axes so a gain can be attributed to a stage rather
than to the pipeline as a whole. This is the root experiment: SFT is trained here, and every other
experiment starts from that checkpoint.

```
Qwen2.5-1.5B base ──────► baseline eval (IFEval / MMLU / judged)
   │  SFT — 25K samples, 1 epoch, packing, prompt masking, lr 1e-5, max_len 4096
   ▼
model_SFT ──────────────► post-SFT battery; the init for RM, DPO, PPO and RS-SFT
   │  DPO — 10K disjoint pairs (trained in beta-sensitivity.md)
   ▼
model_DPO ──────────────► judged win-rate vs SFT
```

**Evaluate before train (hard rule).** No full train starts until the init checkpoint for that stage
has metrics on disk. Without a baseline there is nothing to subtract.

## The three axes

| Axis | Benchmark | Size | What it answers |
|---|---|---|---|
| Format | IFEval strict accuracy | all 541 questions | does it follow explicit instructions? |
| Skills | MMLU | 25 questions × 57 subjects = 1,425 | which way did knowledge move, and how far? |
| Quality | judged win-rate vs previous stage | 500 held-out prompts | do people prefer the output? |

"Questions" here means individual evaluation items — one MMLU multiple-choice question, one IFEval
instruction. (lm-eval calls these *documents*; the word means nothing else in this context.)

### What MMLU tells us

The measurement is a **signed delta**: training moves MMLU up or down, and both directions are
findings. Instruction tuning can improve answer selection or cause forgetting — we report which
happened and by how much, not whether damage occurred.

Separately, a drop beyond **5 points** is an alarm that something broke, not a measurement.

MMLU is scored by comparing log-probabilities of the four options, so it barely exercises
instruction-following — that is IFEval's job. If MMLU moves a lot while IFEval does not, that points
at representation drift rather than a capability change.

### Why 1,425 questions and not fewer

To trust a 5-point alarm, measurement noise has to be well under 5 points — about 1.8. Noise shrinks
as you measure more questions:

| Questions per subject | Total | Noise (±) | Can it detect a 5-point drop? |
|---|---|---|---|
| 5 | 285 | 3.0 pts | no |
| 10 | 570 | 2.1 pts | no |
| **25** | **1,425** | **1.3 pts** | **yes** |
| all | 14,042 | 0.4 pts | yes, at 10× the cost |

25 per subject is the cheapest setting that can tell a broken run from a noisy measurement.

## The base model has no judged score

A judged win-rate is pairwise by definition — there is no such thing as a standalone judged number
for one model. So the baseline pass measures only what a single model can be measured on: IFEval and
MMLU.

The base-vs-SFT judged comparison happens **after SFT exists**, and is reported as its own figure
rather than as a table column. The attribution table's judged column is defined as "versus SFT", so
neither base nor SFT can carry one; it exists to answer what preference optimisation added on top of
SFT.

## When we evaluate during training

**Every training run gets exactly 4 mid-training evaluations plus a final one**, regardless of how
long it is. The interval is derived from the run length (`total_steps / 5`), not fixed in steps — a
fixed step interval would give SFT dozens of evaluations and PPO none, since the stages differ in
length by about 30×.

| Point | Sample | Purpose |
|---|---|---|
| 4 evenly spaced during the run | 285 questions (5/subject) | tripwire — catch a collapse early |
| end of the run | 1,425 questions (25/subject) | the number that goes in the table |

Five evaluations per stage, four stages: 20 checks, ~1 hour total.

The mid-run checks use a small sample on purpose. Their job is noticing a 20-point collapse, and
±3 points is ample for that; paying for ±1.3 points three times per stage buys precision nothing
reads. The end-of-run number is the one that needs to resolve a 5-point alarm, so it pays for the
full sample.

No kept experiment requires a mid-training curve — these are insurance against discovering a broken
run only at the end, when the GPU time is already spent.

## The judged win-rate

**Prompts.** 500 prompts sampled from `HuggingFaceH4/ultrafeedback_binarized` `test_prefs` — the
held-out split, disjoint from everything SFT, RM and DPO trained on. The same 500 for every
comparison, so base-vs-SFT and SFT-vs-DPO are measured on identical ground.

**Why 500.** The win-rate's uncertainty is `sqrt(p(1−p)/N)`. The acceptance target is "DPO beats SFT
above 55%", and that only clears 50% with confidence at N ≥ 500:

| N | 95% interval on a 55% win-rate | verdict |
|---|---|---|
| 100 | 45.2 – 64.8% | unusable |
| 300 | 49.3 – 60.7% | still touches 50% |
| **500** | **50.6 – 59.4%** | **clears 50%** |

**The question the judge is asked.** JudgeArena owns the prompt — we do not reimplement it. It shows
the judge one instruction and two responses and asks which better answers it, returning a preference
score that maps to A / B / tie. Each pair is judged **twice with the positions swapped**, and the
two passes must agree or the result is recorded as a tie.

Position swapping is not optional: on a rung-1 check where both sides were byte-identical, the judge
still disagreed with itself on **17 of 32 pairs** based purely on which slot the text occupied.
Without swapping, those would have been recorded as wins.

**One run, not three.** Generation and judging both run at temperature 0, so repeating the identical
comparison produces the identical number — three runs would report a standard deviation of zero and
manufacture false confidence. The uncertainty that matters is over *prompts*, and that comes from
the binomial interval above, on one pass over 500 prompts. This costs a third as much as three
repeats and is a tighter bound.

## The attribution calculation

Three inputs per stage, then one subtraction:

```
1. skills json      per stage:  {ifeval_prompt_strict, mmlu_acc}
2. style report     per stage:  win_rate_b from the judged head-to-head, that stage as B
3. assemble         one row per stage: base, sft, dpo-b0.05, dpo-b0.1, ppo

4. deltas:
     sft_vs_base   = sft.metric   − base.metric
     <stage>_vs_sft = stage.metric − sft.metric      for every preference stage
```

Two rules that decide whether the table means anything:

- **Preference stages are measured against SFT, not base.** Measuring DPO against base would credit
  it with everything SFT already did.
- **A missing metric stays `None`, never 0.0.** Zero would read as "this stage changed nothing",
  which is a different and false claim. The table build blocks if a generative stage is missing a
  metric rather than reporting a partial result.

## Code

- `src/trainers/sft.py` — packing, prompt masking, chat template
- `src/analysis/attribution.py` — the table and the stage deltas
- `src/eval/lm_eval_skills.py` — IFEval + MMLU
- `src/eval/head_to_head.py`, `src/eval/judge.py` — judged win-rate
- `scripts/analysis/attribution.py` — CLI

## Environment

- **Python 3.12 required** — judgearena needs ≥3.12. On a pod the training env is
  `/opt/conda/envs/py312`, not `/opt/conda/bin/python`.
- `export HF_HOME=/workspace/hf HF_XET_HIGH_PERFORMANCE=1`
- Pinned: torch 2.5.1 · vllm 0.7.3 · trl 0.19.1 · transformers 4.51.0 · lm-eval 0.4.8 ·
  judgearena 0.1.0
- Base `Qwen/Qwen2.5-1.5B`; judge `Qwen/Qwen2.5-32B-Instruct` (needs an 80 GB card)

## Steps

### CPU, before any GPU is rented

- [x] SFT trainer, attribution table and stage deltas implemented and tested
- [x] lm-eval IFEval/MMLU path validated against the real vLLM backend
- [ ] Build and decontaminate the SFT 25K, RM 20K, DPO 10K and 1.5K held-out sets locally
- [ ] Sample the fixed 500-prompt judging set from `test_prefs`; freeze it as an artifact
- [ ] Publish the decontaminated SFT + preference sets

### GPU

- [ ] Create the network volume (250 GB, US-CA-2) and upload the prepared data.
      Never run `setup_secrets.sh` on a rented marketplace box — it logs in a write token.
- [ ] SFT smoke before the full run
- [ ] Baseline eval on `Qwen2.5-1.5B` — IFEval and MMLU only (see below)
- [ ] Run full SFT (resumable, step checkpoints, hub push, W&B resume), with the
      285-question MMLU tripwire at 25/50/75% of the run
- [ ] Eval `model_SFT` on the same battery — **required before RM, DPO, PPO and RS-SFT start**
- [ ] Judged head-to-head base vs SFT on the 500 prompts — reported on its own, not in the table

### After the DPO arms exist

DPO is trained in [beta-sensitivity.md](beta-sensitivity.md); these steps wait on it.

- [ ] lm-eval IFEval + MMLU on each DPO arm
- [ ] Judged head-to-head SFT vs DPO on the same 500 prompts
- [ ] Build the attribution table; it blocks if any generative stage is missing a metric
- [ ] Report which stage moved which axis, including any axis that moved **down**

> Pull `results/` off the volume before any teardown. Nothing on the box persists.
