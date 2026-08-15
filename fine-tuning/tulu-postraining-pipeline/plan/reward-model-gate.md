# Reward model gate

**Question.** Is the reward model good enough to train against?

A quality gate, not a comparison. PPO and rejection-sampling both optimise against `rm_1.5B`, so a
weak reward model silently degrades every result downstream of it. It runs once, after RM training,
before anything consumes it.

Starts from the SFT checkpoint — see [stage-attribution.md](stage-attribution.md) for the box, the
environment and the evaluate-before-train rule.

## Threshold

**RewardBench-chat ≥ 65–70%.** Below that the reward signal is not worth optimising against, and the
PPO and RS-SFT arms should not start.

## Data and training constraints

- 20K pairs from `HuggingFaceH4/ultrafeedback_binarized` `train_prefs`, decontaminated
- **Disjoint from the DPO 10K** on `prompt_id`, asserted — and the guard is tested by deliberately
  violating it, because a guard that never fires is indistinguishable from a broken one
- Linear head on the last non-pad token, Bradley-Terry log-sigmoid loss
- **1 epoch only** — overfitting risk; the trainer forces it regardless of what the config says
- Effective batch 64 pairs, max_len 2048

## Code

- `src/trainers/rm.py` — scalar head, BT loss, forced single epoch
- `src/data_tools/ultrafeedback.py` — the disjoint RM/DPO split
- `src/eval/rb_gate_chat.py` — the RewardBench-chat gate
- `scripts/eval/rb_gate_chat.py` — CLI

## Steps

- [x] Scalar head, BT loss and the forced single epoch validated against the real trainer
- [x] Disjointness guard tested by deliberately violating it
- [ ] Prepare RM 20K **locally on CPU**, before renting a GPU
- [ ] Confirm the RM and DPO prompt sets are disjoint on `prompt_id`
- [ ] RM smoke on GPU before the full run
- [ ] Train the RM from the SFT checkpoint, 1 epoch, logging pair accuracy
- [ ] Confirm the scalar head replaced the LM head and the epoch cap held
- [ ] Run the RewardBench-chat gate
- [ ] Record the score against the threshold — **if it fails, stop before PPO and RS-SFT**
