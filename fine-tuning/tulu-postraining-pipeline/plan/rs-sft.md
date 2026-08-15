# Rejection-sampling SFT

**Question.** On identical prompts, does rejection-sampling SFT match or beat preference
optimisation — against **both** DPO and PPO?

RS-SFT is the simpler recipe: generate n candidates per prompt, keep the best one, SFT on it. No
preference pairs, no reference model, no reward model in the training loop. If it matches DPO and
PPO, a large amount of pipeline complexity is unnecessary at this scale.

```
DPO prompt set ──► generate n=8 per prompt (temp 0.8, top-p 0.95)
                        │
                   judged tournament (single-elim) ──► top-1 ──► SFT ──► model_RS
```

Needs the SFT checkpoint, and the DPO and PPO arms to compare against.

## Method

Two head-to-heads, both with RS as model B so the win-rate is already rs-vs-opponent and chance is
0.5 by construction: RS vs DPO, and RS vs PPO, both on the fixed 500-prompt set.

RS wins only when the whole 95% interval clears 0.5 — a mean above 0.5 with a wide spread is a tie,
not a result. Either comparison can land on its own, so the DPO verdict does not block on the slower
PPO arm.

Use the smaller judge for the high-volume selection pass and reserve the 32B for the final
head-to-heads. **The 32B judge needs an 80 GB card** (~64 GB of bf16 weights).

## Code

- `src/eval/bon/candidates.py` — group generations per prompt
- `src/eval/bon/proxy.py` — reward-model scoring; nested pools per n
- `src/eval/bon/tournament.py` — single-elim judged selection to top-1
- `src/eval/bon/select.py` — writes the RS-SFT training rows
- `src/analysis/verdict/rs_sft.py` — the verdict against both opponents
- `scripts/train/rs_sft.py`, `scripts/analysis/rs_verdict.py` — CLIs

## Steps

- [x] Candidate grouping, nested pools, proxy selection and the judged tournament implemented
- [x] Selection maths tested — pools exactly n, nested across the ladder, ties deterministic
- [x] Verdict rebuilt to compare against **both** DPO and PPO, either alone
- [ ] Generate n=8 candidates per prompt over the same 10K prompt set DPO used
- [ ] Run the judged tournament to top-1 per prompt with the smaller judge
- [ ] Train the RS arm on the top-1 completions, prompt-masked, from the same SFT checkpoint
- [ ] Judged head-to-head RS vs DPO on the 500-prompt set
- [ ] Judged head-to-head RS vs PPO on the same 500 prompts
- [ ] Build the combined verdict; report both comparisons with CIs
