# Stage attribution

**Question.** What does each training stage add — format, skills, or judged quality?

Compares base vs SFT vs DPO on the same three axes so a change can be attributed to a stage rather
than to the pipeline as a whole. Which way each axis moves is the finding, in either direction. This is the root experiment: SFT is trained here, and every other
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
| Quality | judged win-rate vs previous stage | 481 held-out prompts | do people prefer the output? |

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
neither base nor SFT can carry one; it exists to answer what preference optimisation changed on top
of SFT, including changing it for the worse.

## When we evaluate during training

**Every training run gets exactly 4 mid-training evaluations plus a final one**, regardless of how
long it is. The interval is derived from the run length (`total_steps / 5`), not fixed in steps — a
fixed step interval would give SFT dozens of evaluations and PPO none, since the stages differ in
length by about 30×.

| Point | Sample | Backend | Purpose |
|---|---|---|---|
| 4 evenly spaced during the run | 285 questions (5/subject) | **hf**, in-memory | tripwire — stop a collapsed run |
| end of the run | 1,425 questions (25/subject) | **vllm** | the number that goes in the table |

The tripwire runs lm-eval against the model already in memory, because vllm would reserve
~90% of the card that training is using. It **aborts the run** when mmlu falls more than
5 points from its first reading — a tripwire that only logs is a slower way of finding
out at the end. Step checkpoints are already on disk, so an aborted run is inspectable.

Its numbers are not comparable with the table's: different backend, twentieth of the
sample. It answers "has this collapsed", nothing finer.

Configured per stage with `tripwire_evals` (0 turns it off); implemented in
`src/trainers/tripwire.py`. Smokes disable it — a 2-step smoke should not stop to run
285 questions.

**Verified on a GPU** with `RUN_TRIPWIRE_SMOKE=1`: `HFLM` accepts the in-memory model,
measuring does not exhaust the card training is already using, and training continues
afterwards and still writes a trained checkpoint. The unit tests stub lm-eval, so that
smoke is the only thing covering the real path — re-run it after any change to the
callback or to the lm-eval pin.

The mid-run checks use a small sample on purpose. Their job is noticing a 20-point collapse, and ±3 points is ample for that; paying for ±1.3 points three times per stage buys precision nothing
reads. The end-of-run number is the one that needs to resolve a 5-point alarm, so it pays for the full sample.

No kept experiment requires a mid-training curve — these are insurance against discovering a broken run only at the end, when the GPU time is already spent.

## The judged win-rate

**Prompts.** `data/processed/eval_prompts.jsonl`, built by `scripts/prepare/judge.py` and frozen.
The same file for every comparison, so base-vs-SFT and SFT-vs-DPO are measured on identical ground.

**Why 481 and not a round number.** The size is decided by what is left, not chosen. PPO trains on
`test_prefs` too, and judging PPO on prompts it was RL-trained on would inflate its win-rate — so the
judging set is the remainder, and `prepare_judge_prompts` asserts the disjointness rather than
assuming it:

```
test_prefs                          2,000
  non-empty + decontaminated        1,981
    ppo_1.5k (PPO trains here)      1,500
    judging set                       481
```

**What 481 resolves.** The win-rate's uncertainty is `sqrt(p(1−p)/N)`. The question is whether the
interval clears parity **in either direction**, and a 5-point margin needs N ≥ 381:

| N | 95% interval on a 5-point margin | verdict |
|---|---|---|
| 100 | 45.2 – 64.8% | unusable |
| 300 | 49.3 – 60.7% | still touches 50% |
| 381 | 50.0 – 60.0% | the strict minimum |
| **481** | **50.5 – 59.5%** | **what we have** |

**N is decisive pairs, not prompts.** Ties leave the denominator — `WinRate.decisive` excludes them,
and a position-swap disagreement is recorded as a tie — so 481 prompts is 481 only if nothing ties.
At a 20% tie rate it is 385, right at the minimum; past ~21% a 5-point margin stops resolving. The
tie rate has never been measured, and the one datapoint points the wrong way: on a rung-1 check the
judge disagreed with itself on 17 of 32 byte-identical pairs. **Report the interval from the observed
decisive count** — `assess_head_to_head` already computes it — rather than promising ±4.5 in advance.

The table shows the margin as 55% for legibility; 45% is the same interval mirrored, and
`assess_head_to_head` calls SFT the winner when the interval sits wholly below parity. **DPO not
beating SFT is a result, not a failed run** — the same rule the MMLU axis already applies, where a
signed delta in either direction is the finding. The size tells the three outcomes apart; it is not
a bar to clear.

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
the binomial interval above, on one pass over the 481 prompts. This costs a third as much as three
repeats and is a tighter bound. **Enforced in code**: `run_head_to_head` has no `runs` parameter and
`configs/eval.yaml` no `default_runs`, so the lever cannot be turned back on by accident.

**Generations are cached per model, not per comparison.** SFT appears in four head-to-heads (vs base,
vs both DPO arms, vs PPO), and generation is deterministic at temperature 0 over a frozen prompt set,
so it runs once. `cached_generations` writes to a shared `generations/` directory keyed by model name
and `generate_incremental` skips what is already there — 12 generation passes become 6, along with 6
vLLM engine inits. The one hazard, two checkpoints sharing a directory basename, raises instead of
silently serving one model's completions as another's.

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
- `src/prepare/{sft,rm,dpo,ppo}.py` — the four sets; each builds the decontam bank and
  passes it to the builder in `src/data_tools/`
- `src/prepare/judge.py`, `scripts/prepare/judge.py` — freeze the judging prompt set,
  held back from PPO's pool
- `scripts/prepare/push.py` — upload `data/processed/` to the hub, card and all
- `src/analysis/attribution.py` — the table and the stage deltas
- `src/eval/lm_eval_skills.py` — IFEval + MMLU
- `src/eval/head_to_head.py`, `src/eval/judge.py` — judged win-rate
- `scripts/analysis/attribution.py` — CLI

## Prepared data — build once, download thereafter

The four sets live on the hub at
[`kosiasuzu/tulu-postraining-data`](https://huggingface.co/datasets/kosiasuzu/tulu-postraining-data)
(public, 262 MB). Rebuilding them on a rented box means downloading all of Tulu-3 and
UltraFeedback and re-running the decontam scan — GPU time spent on CPU work. Pull instead:

```sh
hf download kosiasuzu/tulu-postraining-data --repo-type dataset --local-dir data/processed
```

The upload is the `save_to_disk` layout unchanged, so the trainers' `load_from_disk` works
with no code change. Re-push after any prep change with `scripts/prepare/push.py --public`;
it regenerates the dataset card from what is actually on disk.

**Decontamination.** One 8-gram bank built from the benchmarks we score on — RewardBench,
IFEval, MMLU (`prepare/decontam.py`), 547,015 grams from 19,384 texts. Applied **before**
sampling, so a contaminated row is replaced rather than leaving the subset short. SFT and the
preference pairs scan the prompt **and** the responses, since all of it is trained on; the PPO
pool scans the prompt only, because its stored responses are discarded and the policy
generates its own. `--skip-decontam` on every prep CLI turns it off, for debugging only.

**The CLI is `hf`, not `huggingface-cli`.** The Homebrew `huggingface-cli` on this machine
predates the `hf` command; the current one is a uv tool at `~/.local/bin/hf`, installed with
`uv tool install huggingface_hub`. It carries its own `huggingface_hub`, independent of the
version the project venv pins.

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
- [x] **Build and decontaminate all four sets** — `sft_25k` 24,634 · `rm_20k` 20,000 ·
      `dpo_10k` 10,000 · `ppo_1.5k` 1,500. Decontam had been wired into SFT prep only;
      the preference builders now take an `ngram_bank` too. It is not cosmetic: **802
      rows of `train_prefs` (1.3%) and 19 of `test_prefs` overlap the eval bank**, so
      roughly 260 of the 20,000 RM pairs would have carried eval text — against a bank
      that includes RewardBench, the benchmark the RM gate is scored on.
- [x] **Published** to [`kosiasuzu/tulu-postraining-data`](https://huggingface.co/datasets/kosiasuzu/tulu-postraining-data)
      (public). See *Prepared data* above for the pull command.
- [x] **Judging set frozen** — `data/processed/eval_prompts.jsonl`, **481 prompts**, built by
      `scripts/prepare/judge.py`. Not 500: PPO trains on `test_prefs` as well and takes 1,500
      of the 1,981 clean prompts, so the judging set is what it leaves. The alternative was
      shrinking PPO's pool to 1,481 — above its 1,472 episodes, so no prompt reuse — for a
      round 500; not worth 0.1 points of resolution and 9 prompts of slack.

### GPU

- [x] **Network volume + data pulled.** `zdhbaj21wa`, **200 GB, EU-RO-1** — not US-CA-2, which
      was `Low` on A100 and H100 at creation time, and volumes cannot move between regions.
      Base model (2.9 GB) and all five data artifacts pulled from the Hub; nothing rebuilt.
      Two corrections to the recipe, both in `VOLUME.md`: the image must be `runpod/*` (Docker
      Hub's `pytorch/pytorch` has no sshd, so SSH is refused), and the venv goes on **container
      disk, not the volume** — installing to `/workspace` ran at 116 MB/min and took ~20 min
      against 2 s on local disk, and the import tax repeats on every process.
- [x] SFT smoke before the full run — **passed in 20 s** on the A100 80 GB PCIe
- [x] Tripwire smoke (`RUN_TRIPWIRE_SMOKE=1`) — **passed in 297.7 s** on the A100 80 GB PCIe.
      Proves all three at once: `HFLM` accepts the in-memory model, measuring does not exhaust
      the card training is already using, and training resumes afterwards and still writes a
      trained checkpoint (`assert_saved_model` and `assert_trained` both hold).
- [x] **Baseline eval on `Qwen2.5-1.5B`** — IFEval **21.07%** prompt-strict (all 541),
      MMLU **62.46%** (25/subject = 1,425). `results/metrics/skills_Qwen2.5-1.5B.json`.
      MMLU matches Qwen's own reported ~60%, so the harness is measuring rather than
      producing a plausible artefact; the low IFEval is the base model's, and is the
      thing SFT is meant to move.

      **Took 5 minutes, not the ~25 forecast.** REFORECAST extrapolated eval cost from
      lm-eval's hf backend on a 3090; vLLM on an A100 is ~5x that. The ~10 remaining
      skills evals are therefore under an hour in total, and the full 14,042-question
      MMLU would be ~30-40 min rather than prohibitive — still not worth +/-0.4 over
      +/-1.3, but no longer the 10x the plan assumed.

      Two bugs surfaced getting here, both now fixed and tested: lm-eval's `limit` is
      global across tasks, so per-task depths (`skills_task_limits`) were added rather
      than truncating IFEval to 25; and `run_skills_eval` resolved the model with
      `resolve_path`, which glued the hub id onto the repo root — **the baseline path
      had never worked**. `model_ref` now lives in `prepare/paths.py` and is used by
      every model-taking caller.
- [x] **Full SFT run** — `qwen2.5-1.5b_sft_20260816T1938Z`, **33 min 27 s** for 260 steps
      over 24,634 rows (17.0M tokens), 1 epoch. Final loss 0.918, token accuracy 77.3%.
      Pushed private to `kosiasuzu/qwen2.5-1.5b_sft_20260816T1938Z`.

      **The tripwire ran four times on a real run and never fired**: mmlu 0.6491 (step 52,
      its baseline) -> 0.6421 -> 0.6456 -> 0.6491, worst excursion **0.70 points** against
      the 5-point abort. SFT taught format without eroding knowledge. Its 0.6491 also sits
      within noise of the vllm baseline's 0.6246 — different backend and a twentieth of
      the sample, so not strictly comparable, but agreement at that level says neither
      measurement is broken.

      **REFORECAST is systematically pessimistic and should be re-baselined.** It projected
      2-4 h; the run took 0.56 h. With the eval coming in 5x fast too, the real 0.5B/3090 ->
      1.5B/bf16/A100 hop is landing near x0.15, not the x0.35-x1.0 band. Every remaining
      cost estimate in the programme derives from that band.
- [x] **Eval `model_SFT`** — IFEval **17.93%** prompt-strict, MMLU **62.88%**
      (`mmlu_diff=+0.42`, well inside the 5-point alarm).
      `results/metrics/skills_qwen2.5-1.5b_sft_20260816T1938Z.json`.

      **IFEval fell 3.14 points, and the measurement is not clean.** Confirmed against the
      lm-eval v0.4.8 source: `ifeval.yaml` is `doc_to_text: prompt` — the bare instruction —
      and `simple_evaluate(apply_chat_template=False)` by default, which our wrapper never
      overrides. So the checkpoint was prompted with no `<|im_start|>` markers and no
      generation prompt, after being trained on that format exclusively with
      `assistant_only_loss=True`. MMLU is unaffected because log-prob scoring over fixed
      options barely depends on formatting; IFEval is generative, which is where it bites.

      **The base and SFT numbers are therefore not comparable** — raw prompting is neutral
      for a base model and adversarial for an instruct-tuned one, so the delta is a
      diagonal across two protocols rather than a measurement of what SFT changed.

- [ ] **Settle the IFEval prompting protocol before reading the format axis.** Two runs,
      not four — the raw pair is already on disk, the templated pair is missing:

      |        | raw prompt | templated |
      |--------|-----------|-----------|
      | base   | 21.07% (have) | needed |
      | SFT    | 17.93% (have) | needed |

      Report the templated pair as the headline, since it matches how the models are used
      and what `head_to_head.py` already does for the judged axis; keep raw as secondary.
      ~10 min and ~$0.25 of GPU.

- [x] **Added `--apply-chat-template`** to `scripts/eval/skills.py`. The plumbing existed —
      `run_skills_eval(**extra_kwargs)` already reaches `simple_evaluate` — so this is a
      flag, not a mechanism. **Do not set `system_instruction`**: the template itself emits
      `<|im_start|>system\nYou are a helpful assistant.` when a row has no system turn, which
      is what nearly every Tulu row did in training. Adding one on top would diverge from
      training conditions rather than reproduce them.

      The protocol is now recorded in the metrics json (`apply_chat_template`) and in the
      filename (`skills_<model>_chat.json`), so a templated run cannot silently overwrite
      an untemplated one — the two answer different questions and a file that does not say
      which is not comparable to anything.

- [ ] **Re-pull `skills_Qwen2.5-1.5B.json` from the volume.** The local copy was destroyed
      by a test that wrote fixture values through `DEFAULT_METRICS_DIR`; the test now
      redirects it to a tmp dir. The real file is intact on the volume at
      `results/metrics/`, and the headline numbers (IFEval 21.07%, MMLU 62.46%) are above —
      what is lost locally is the 61 subject slices and the other three IFEval accuracies.
      Costs nothing beyond mounting the volume, which the next session does anyway.

- [ ] **Ship the inference template with the checkpoint, not the training one.** `sft.py:128`
      installs the `{% generation %}`-marked template for `assistant_only_loss` and
      `sft.py:190` saves that tokenizer into the artifact. Verified locally that the markers
      render to nothing — base and training templates produce byte-identical prompts — so
      this is not a correctness bug. It is a portability one: anyone rendering our published
      checkpoint with a jinja environment lacking the extension gets `unknown tag
      'generation'`. Restore the base template before `save_pretrained`.
- [ ] Judged head-to-head base vs SFT on the judging set — reported on its own, not in the table

### After the DPO arms exist

DPO is trained in [beta-sensitivity.md](beta-sensitivity.md); these steps wait on it.

- [ ] lm-eval IFEval + MMLU on each DPO arm
- [ ] Judged head-to-head SFT vs DPO on the same judging set
- [ ] Build the attribution table; it blocks if any generative stage is missing a metric
- [ ] Report which stage moved which axis, including any axis that moved **down**

> Pull `results/` off the volume before any teardown. Nothing on the box persists.
