# Rung 1 report — Phase R

**What it was:** run every stage of the pipeline end-to-end at 0.5B on a cheap rented GPU
(Vast.ai RTX 3090, $0.160/hr) before committing to 1.5B on A100-class hardware.

**What it cost:** **$1.196**, across TWO instances — 47706876 ($0.468, earlier session) and 47725995
($0.728, 4.42 h, destroyed 2026-08-14). Both figures are from Vast's charge ledger, not from
`duration x dph`: the latter gave $0.71 for the second instance and misses the storage and bandwidth
billed on top, which is the same ~12% gap the old spec already flagged in its rate notes. Most of the
wall-clock was the box idling between commands rather than compute.

**What it found:** 20 defects, 8 of which fail *silently* — they produce plausible-looking output
while doing nothing, or the wrong thing. Those are the ones that justify the rung: a crash on
expensive hardware costs a re-run, but a silent failure costs the *conclusion*, and you don't find
out until the write-up doesn't make sense.

---

## 1. Scoreboard

| Task | Result |
|---|---|
| R0 pod / R1 install | Python 3.12 required — `judgearena` needs ≥3.12 and the py3.11 image cannot resolve the stack at all |
| R2 decontam / R3 disjointness | CPU-only; guards confirmed to fire when deliberately violated |
| R4 SFT smoke | **1.700 steps/s** — caught a silent null training run |
| R5 RM smoke | **1.479 steps/s** — caught a TRL preference-schema mismatch |
| R6 DPO smoke | **3.278 steps/s** — init loss exactly `ln(2)`, proving the reference model is frozen |
| R7 PPO API | **17,194 MiB peak**; KL penalty arithmetically exact; 3 bugs |
| R8 vLLM generation | **~1,430 tok/s**; engine-per-batch bug; torn-tail resume gap |
| R9 lm-eval | 2 bugs, 5 attempts; `batch_size="auto"` **does not work** for loglikelihood |
| R10 judge | **~10 judgments/s**; 3 bugs; **53% position bias** measured |
| R11 PPO bound | 4 config problems; PPO **cannot run on 24 GB at any model size** |
| R12 re-forecast | Training lines **~3× over-provisioned**; 2 lines change hardware |

Test suite over the phase: **90 → 102 passing**, every addition anchored to a real failure.

---

## 2. The silent failures

These are the findings that would not have surfaced as errors.

1. **SFT trained on nothing and the test passed.** The smoke used `max_length: 256`. Real Tulu rows
   are median 554 tokens, and 11 of 32 have their assistant span starting past token 256 — so
   truncation removed every assistant token, the label mask was all-zero, loss and grad_norm were
   0.0, and `assert_saved_model` happily confirmed a checkpoint existed. **Fix:** deleted the
   `max_length` override so smokes use production values, and added an `assert_trained` fixture that
   reads `trainer_state.json`. Verified both ways — passes at 4096, fails at 256.
2. **The judge would have sampled at temperature 0.6.** JudgeArena 0.1.0 hardcodes
   `SamplingParams(temperature=0.6, top_p=0.95)`. The spec requires temp 0. This one is corrosive
   rather than fatal: `aggregate_winner` returns a **tie** whenever the two position-swapped passes
   disagree, so judge noise does not average out — it converts real wins into ties and flattens the
   DPO-vs-PPO signal the comparison exists to measure.
3. **PPO wrote zero checkpoints.** `save_steps: 50` exceeded the run's 46 total updates. The plan's
   own assurance that "a truncated PPO still yields a usable checkpoint" was false — a pod kill at
   update 45 lost everything.
4. **`max_steps` does not bound a TRL 0.19 PPO run.** `PPOTrainer` overwrites it
   (`state.max_steps = num_total_batches`, `ppo_trainer.py:386`) and iterates `num_total_batches`.
   the old spec's PPO guidance recommended exactly this. Setting it would have been silently ignored.
5. **PPO's run length was set by the eval split.** `total_episodes` falls back to
   `num_train_epochs × train_dataset_len`, and `train_dataset_len` is the pool minus
   `num_eval_prompts` — so changing the eval split silently changed how long PPO trained.
6. **A torn JSONL line made every resumable job unrecoverable.** A real `SIGKILL` leaves a
   half-written final line; `load_completed_ids` raised on it. Worse, skipping it on read alone is a
   trap — the next append runs onto that unterminated line, merging two records and moving the damage
   mid-file where it is indistinguishable from corruption. The earlier resume tests used `head -n`,
   which always cuts on a line boundary and never exercised this.
7. **Judge and generation engines were rebuilt per batch.** Each reserves ~90% of the card. Invisible
   to tests because every stub replaced the function that builds them.
8. **`run_head_to_head` never released the generation engine before loading the judge** — despite its
   docstring claiming "sequential on one gpu".

---

## 3. The crashes that fire *after* training starts

Cheap to fix, expensive to hit — all of these detonate deep into a run.

- **PPO ×3.** TRL's `PPOTrainer.train()` calls `generate_completions()` whenever
  `num_sample_generations > 0` (default 10), and that path iterates `eval_dataset`. (a) it was never
  passed → `NoneType has no len()`; (b) the eval dataloader uses `drop_last=True`, so a partial batch
  is dropped whole and iteration yields `None`; (c) a pool too small to spare a batch reproduces (a).
  SFT/RM/DPO all treat `eval_dataset` as optional, so nothing upstream could catch any of them.
- **Judge.** `make_model(id, temperature=0)` routes temperature into vLLM's engine kwargs, which
  reject it outright.
- **RM/DPO schema.** TRL's `apply_chat_template` validates an exact key set. Our artifact keeps 7
  columns; TRL raised `KeyError` inside a `.map()`, surfacing much later as a misleading
  "text input must be of type str". Fixed with `to_trl_preference_columns()` — the on-disk artifact
  stays rich, only the trainer's view is reduced.

---

## 4. Memory — the recurring theme

The same arithmetic caused three separate failures: **`tokens × vocab × 4 bytes`**, allocated
*outside* whatever the framework reserved.

| Where | Tensor | Size |
|---|---|---|
| lm-eval MMLU | logits at every prompt position (loglikelihood) | 16,384 × 151,936 × 4 = **9.27 GiB** |
| TRL PPO | `torch.stack(output.scores)` over the whole rollout | 8 × 4 × 512 × 151,936 × 4 = **9.27 GiB** |
| vLLM sampler | fp32 log-softmax + gather + int64 ranks | ~3 copies, one at 8 bytes/element |

**Qwen2.5's vocab is 151,936 at both 0.5B and 1.5B, so none of this shrinks with the model.** The
knobs are `max_num_batched_tokens` and `response_length`, not model size — which is the opposite of
the usual intuition and the reason three attempts at lowering `gpu_memory_utilization` failed: that
buys a *fixed* headroom against a peak, while the token budget is a *multiplier* on every tensor at
once.

**Consequence:** `DEFAULT_VLLM_ARGS = "gpu_memory_utilization=0.45,max_num_batched_tokens=2048,max_model_len=2048"`
for lm-eval, and PPO cannot run on a 24 GB card at any model size.

---

## 5. Measurements

All at 0.5B on RTX 3090. SFT/RM/DPO ran at batch 1 in fp32 (smoke overrides), so they **understate**
production, which uses microbatch 2–4 with accumulation 8–16 in bf16.

| Stage | Rate | Peak memory |
|---|---|---|
| SFT | 1.700 steps/s | — |
| RM | 1.479 steps/s | — |
| DPO | 3.278 steps/s | — |
| PPO | 6.24 s/episode | 17,194 MiB @ batch 1; 22.67 GiB @ 8 episodes (OOM) |
| vLLM generation | ~1,430 tok/s + 46 s init | 20,746 MiB |
| Pairwise judge (1.5B) | ~10 judgments/s + 64 s init | 20,430 MiB |

Correctness confirmations worth as much as the rates:

- **DPO init loss = `ln(2)` = 0.6931 exactly** — the policy equals the frozen reference at init, so
  log-ratio 0 → sigmoid(0) = 0.5 → −log(0.5). A drifting or mismatched reference would not land there.
- **PPO KL penalty is arithmetically exact** — `rlhf_reward = scores − 0.05·kl` held on every logged
  step, proving `kl_coef` reaches the reward computation rather than merely being accepted by the
  constructor.
- **`clipfrac ≈ 0.0015`**, as the spec requires.
- **Judge position bias: 53%.** Both checkpoints emitted byte-identical completions, making this an
  accidental clean probe. The judge returned 32/32 ties (correct) but disagreed with itself on
  **17/32** pairs based purely on slot order, at temperature 0. Position swapping is load-bearing, not
  a refinement — without it, ~17 of 32 would have been recorded as spurious wins.

---

## 6. What changes for the main run

**Config changes now in the repo**

- `configs/ppo.yaml`: `total_episodes: 1472` (explicit bound), `save_steps: 10` (was 50),
  `num_sample_generations` / `num_eval_prompts` / `per_device_eval_batch_size` added.
- `src/eval/lm_eval_skills.py`: `DEFAULT_VLLM_ARGS` memory guard, `build_model_args()` merging by key.
- `requirements.txt`: `langdetect`, `immutabledict` pinned (lm-eval hides them behind an `ifeval`
  extra, so a plain install imports fine and dies at scoring time).

**Spec corrections**

- the old spec's PPO guidance — bound PPO with `total_episodes`, **not** `max_steps`.
- Compute table gained a measured column; training lines are ~3× over-provisioned.
- **S9's four PPO mitigation arms move off the RTX 4090 line onto A100** (+£8). They cannot run on
  24 GB.
- The 32B judge needs ~64 GB of bf16 weights and cannot sit on the 4090 line either.

**Operational**

- Eval subset sizing is now derivable rather than guessed: lm-eval reports `acc_stderr` directly, so
  the minimum credible `--limit` follows from the precision required. To detect the spec's 5-point
  MMLU drop at 2σ needs stderr < 1.77pp → **MMLU limit=25** (1,425 docs, ~1.32pp). **IFEval should run
  full** (541 docs) — at limit=10 its stderr is 13.3pp on a 20% score.
- The training env on any pod is `/opt/conda/envs/py312`, **not** `/opt/conda/bin/python` (the base
  image, torch 2.4.0, no vLLM). Probing the wrong one gives false answers — it cost a wrong
  recommendation during R8.
- Every `run_skills_eval` call re-pays ~46 s engine init *and* full MMLU doc-building (57 subjects,
  single-threaded CPU, GPU at 0%). Batch tasks into one call where possible.

---

## 7. What Rung 1 did **not** verify

Stated plainly, because the re-forecast rests on it:

- **Anything at 1.5B.** Every production number crosses a model-size hop from 0.5B.
- **Anything on an A100.** Every production number crosses a hardware hop from a 3090.
- **PPO at its production 32-episode rollout** — it OOMs on 24 GB, so it has never run at all.
- **The 32B judge**, the largest single eval line in the programme. The ~1.2 judgments/s figure is a
  projection across both a 21× model hop and a hardware hop.
- **Judge and BoN torn-tail resume** have unit coverage only; only the generation path was verified
  against a real crash artifact on the box.

**Recommendation:** the first hour of A100 time should re-measure SFT and PPO rather than trust the
projections. The *relative shape* of the budget is now evidence-based; the absolute numbers are not.

---

## 8. The honest summary

Rung 1 cost **$1.20** and returned:

- 8 silent failures that would have produced confident, wrong results
- 6 crashes that fire deep into expensive runs
- 4 memory limits that change *hardware allocation*, not just hours
- a budget re-forecast showing training is ~3× over-provisioned
- 12 new regression tests, each anchored to a real failure

The pattern worth carrying into the main run: **every one of the silent failures was invisible to the
existing tests because the tests stubbed the exact thing that was broken.** The engine-rebuild bugs
were hidden by a stub that replaced the function building engines; the temperature bug by a stub that
returned a bare `object()` swallowing every kwarg; the null training run by an assertion that checked
a file existed rather than that learning happened. When a test double is more permissive than the real
dependency, it does not just fail to catch bugs — it actively certifies them.
