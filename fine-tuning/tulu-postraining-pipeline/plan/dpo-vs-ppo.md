# DPO vs PPO

**Question.** At equal prompts and equal data, does PPO beat DPO at 1.5B?

Both arms start from the same SFT checkpoint and see the same prompt set, so the comparison isolates
the optimisation method rather than the data. Needs a reward model that passed
[reward-model-gate.md](reward-model-gate.md), and the DPO arms from
[beta-sensitivity.md](beta-sensitivity.md).

## Method

Each arm is judged against SFT on the same 481 held-out prompts — held back from PPO's own
training pool, so PPO is not judged on prompts it was rl-trained on — and the two win-rates are
compared with a confidence interval on the **difference**. A winner is declared only when that interval excludes
zero; otherwise it is a tie.

The interval comes from the number of prompts judged, not from repeating the run: generation and
judging are both temperature 0, so a repeat produces the identical number. The verdict code refuses
to declare a winner when it is handed a single value per arm with no spread.

Secondary: KL spent and wall-clock per arm — PPO is expected to cost far more for whatever it buys.

## PPO configuration

- 1.5K held-out prompts from `test_prefs` — prompts the RM never trained on
- KL β=0.05 against the SFT reference, clip 0.2, 1 gradient step per batch, `clip_frac ≈ 0`
- Score only EOS-terminated completions
- `total_episodes` set **explicitly**. `max_steps` does nothing — TRL overwrites it and drives the
  loop from `num_total_batches`, which comes from `total_episodes` alone.
- `save_steps` must stay well under the total update count, or the run writes **zero** checkpoints
  and resume is silently dead.

**PPO needs an 80 GB card.** TRL generates the whole rollout at once and stacks full-vocab logits for
every generated position: `batch × response_length × vocab × 4` = 9.27 GiB on top of four resident
models. Vocab is 151,936 at both 0.5B and 1.5B, so that term does **not** shrink with model size —
cut `response_length` or the rollout batch instead.

## Code

- `src/trainers/ppo/` — policy, frozen reference, reward and value models
- `src/trainers/dpo.py` — the DPO arm
- `src/analysis/verdict/dpo_ppo.py` — the comparison and the CI
- `src/analysis/verdict/arms.py` — loads an arm from a head-to-head summary
- `scripts/analysis/dpo_ppo_verdict.py` — CLI

## Steps

- [x] PPO validated against the real TRL API; KL penalty verified arithmetically
- [x] Run bounded with `total_episodes`; `save_steps` brought under the update count
- [x] Verdict CI implemented; a single run per arm now refuses to declare a winner
- [x] Prepare the held-out prompt pool from `test_prefs` **locally on CPU** —
      `data/processed/ppo_1.5k`, 1,500 prompts, decontaminated. The 481 prompts it leaves
      are the judging set, so PPO is never judged on prompts it trained on
- [ ] PPO smoke on GPU before the full run
- [ ] Train PPO from the SFT checkpoint against `rm_1.5B`; watch `clip_frac ≈ 0`
- [ ] Confirm `rlhf_reward = scores − kl_coef · kl` holds on the real run
- [ ] Judged head-to-head, DPO and PPO each vs SFT on the same judging set, one pass
- [ ] Record KL spent and wall-clock per arm
- [ ] Build the verdict; report the delta and its CI even when the answer is a tie
