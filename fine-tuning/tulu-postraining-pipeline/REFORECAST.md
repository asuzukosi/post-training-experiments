# R12 — budget re-forecast from measured throughput

Every number in §1 was measured on the Phase-R rung-1 box (Vast.ai RTX 3090, $0.160/hr).
Everything in §3 is a projection from those numbers and is labelled as such.

## 1. What was actually measured

| Stage | Rate | Conditions |
|---|---|---|
| SFT | **1.700 steps/s** | 0.5B, fp32, batch 1, `max_length` 4096 |
| RM | **1.479 steps/s** | 0.5B, fp32, batch 1, `max_length` 2048 |
| DPO | **3.278 steps/s** | 0.5B, fp32, batch 1, `max_length` 2048 |
| PPO | **6.24 s/episode** | 0.5B, fp32, 8 episodes/update, `response_length` 512 |
| vLLM generation | **~1,430 completion tok/s** + ~46 s engine init | 0.5B, bf16 |
| Pairwise judge | **~10 judgments/s** + ~64 s engine init | 1.5B judge, bf16, 2 passes per judgment |
| lm-eval MMLU | 285 docs in 234.7 s | includes ~46 s init + single-threaded doc prep |
| lm-eval IFEval | 10 docs in 79.2 s | init-dominated |

**Peak memory:** PPO 17,194 MiB at batch 1 (R7); 22.67 GiB at 8 episodes/update before OOM (R11).

### Conditions that bias these numbers

- **SFT/RM/DPO ran at batch 1** (`BASE_OVERRIDES`), while production uses microbatch 2–4 with
  accumulation 8–16. Larger batches raise utilisation, so these rates **understate** production.
- **The smokes force `bf16: False`**; production is bf16 throughout. Another understatement.
- The generation figure was taken with `DEFAULT_BATCH_SIZE = 8` write batches, which R8 flagged as
  leaving throughput on the table. It is a **lower bound**, not a steady-state maximum.

## 2. The scaling hops

Production differs from the measurement in three ways at once, so no single multiplier is honest:

| Hop | Direction | Rough factor |
|---|---|---|
| 0.5B → 1.5B | slower | ×3 (approximate; attention does not scale linearly) |
| fp32 → bf16 | faster | ÷~2 on Ampere tensor cores |
| RTX 3090 → A100 80 GB | faster | ÷~2–3.5 |

Net ≈ **×0.35 to ×1.0**, centred on **×0.5** — production is plausibly *faster* than the 0.5B
measurement despite tripling the model. Batching headroom is deliberately excluded, so the range
leans conservative.

## 3. Re-forecast — S1–S4 training

| Stage | Measured (0.5B/3090) | ×0.35 | **×0.5** | ×1.0 | Spec |
|---|---|---|---|---|---|
| SFT (25K × 1 epoch) | 4.08 h | 1.43 | **2.04** | 4.08 | 3 |
| RM (20K pairs) | 3.76 h | 1.31 | **1.88** | 3.76 | 3 |
| DPO (10K × 2 β arms) | 1.69 h | 0.59 | **0.85** | 1.69 | 5 |
| PPO (1,472 episodes) | 2.55 h | 0.89 | **1.28** | 2.55 | 8 |
| **Total** | **12.09 h** | **4.23** | **6.04** | **12.09** | **19** |

**The training lines are ~3× generous.** Even the pessimistic ×1.0 column lands under the spec's 19 h.
SFT and RM are close to right; DPO and PPO are the overestimates.

## 4. Findings that change the plan, not just the numbers

These came out of Phase R and are not absorbed by re-pricing.

1. **PPO cannot run on a 24 GB card at any model size.** TRL stacks full-vocab logits for every
   generated position in fp32: `batch × response_length × vocab × 4` = 8 × 4 × 512 × 151,936 × 4 =
   **9.27 GiB**, on top of four resident models. Vocab is 151,936 at both 0.5B and 1.5B, so this term
   **does not shrink with the model**. → **S9's four PPO mitigation arms are budgeted on an RTX 4090
   24 GB and cannot run there.** They need A100 time, a smaller rollout batch, or a shorter
   `response_length`. This is a cost-line change: 12 h × (£1.25 − £0.58) ≈ **+£8**.
2. **The 32B judge needs an 80 GB card** (~64 GB of bf16 weights) and cannot sit on the 4090 line.
   Extrapolating 1.5B → 32B across a hardware hop gives ~1.2 judgments/s, so ~4,500 head-to-head
   judgments ≈ 1 h — but this is a projection through two hops and should be measured before it is
   quoted.
3. **`max_steps` does not bound a TRL 0.19 PPO run** — `spec.md:466` recommends it and is wrong.
   `total_episodes` is the only lever (now set explicitly to 1,472).
4. **`whiten_rewards` asserts `local_mini_batch_size >= 8`**; ours is 2. The planned E10/S9 whitening
   arm would assert-fail on the current config.
5. **Teacher generation (S5, N=8 over 10K prompts) is the least-supported line.** 41M tokens at the
   measured lower-bound rate exceeds the spec's 3 h — but that rate was throttled by our write-batch
   size, so it may be fine. **Measure before trusting either direction.**

## 5. What is still unmeasured

- Anything at **1.5B** — every production number crosses a model-size hop from 0.5B.
- Anything on an **A100** — every production number crosses a hardware hop from a 3090.
- PPO at the **production rollout batch** (32 episodes), which OOMs on 24 GB and so has never run.
- The **32B judge**, the largest single eval line in the programme.

The honest summary: the *relative* shape of the budget is now evidence-based, and the training lines
are comfortably over-provisioned. The *absolute* production numbers still rest on two untested hops,
and the first hour on an A100 should be spent re-measuring SFT and PPO rather than assuming these.
