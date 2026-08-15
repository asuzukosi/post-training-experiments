# Sycophancy steering

**Question.** Can a contrastive activation vector reduce sycophancy at inference time, without
retraining, and without destroying capability?

Runs against the model that shows the symptom — the PPO checkpoint from
[dpo-vs-ppo.md](dpo-vs-ppo.md). See [stage-attribution.md](stage-attribution.md) for the box and
environment.

Sycophancy here is measured concretely: the **flip rate**, the fraction of *correct* answers the
model abandons after the user pushes back. Extract a direction that separates sycophantic from
non-sycophantic behaviour, add it to the residual stream during generation, and see whether the flip
rate falls while MMLU holds.

## Method

Extract at the **last prompt token** of the residual stream, sweeping over middle layers:

```
v_l = mean(h_l | s+) − mean(h_l | s−),  then unit-normalised
```

`s+` / `s−` are trait-eliciting vs trait-suppressing prompt pairs. Then steer:

```
h ← h + α·v            α ∈ {−2, …, +2}
```

Also test the **capping** variant, which clamps rather than translates and tends to cost less
capability:

```
h′ = h − v · min(⟨h, v⟩ − τ, 0)        τ = 25th percentile of training projections
```

Flip rate is measured ×3 at temp 0.7, mean ± std. **Watch for a non-monotone / U-shaped response in
α** — assuming monotonicity is how a real effect gets missed.

## Two failure modes that look identical to "steering doesn't work"

1. **A hook that never fires** produces exactly zero steering effect. The hook must be verified to
   fire, and α=0 must be a genuine no-op against unsteered generation.
2. **A sign-flipped vector** steers deeper into sycophancy while every other part looks correct.
   `v` must point from negative toward positive, and the maths is asserted offline.

## Code

The package is split along the seam between pure maths and model access:

- `src/eval/steer/vectors.py` — the vector itself: contrastive maths, τ, trait pairs, save/load.
  **Touches no model**; fully covered by CPU tests.
- `src/eval/steer/extract.py` — the only forward passes: last-token hiddens out of a real model
- `src/eval/steer/apply.py` — residual hook, α sweep, steered generation
- `src/eval/steer/flip_rate.py` — probes, pushback turn, flip scoring

## Steps

- [x] Package split along the maths/model seam so the maths is covered without a GPU
- [x] Contrastive vector, τ and layer selection tested — sign, unit norm, degenerate input
- [ ] Author trait pairs (`prompt_pos` / `prompt_neg`) and pushback probes with known answers —
      **CPU/authoring work, do it before renting a GPU**
- [ ] Extract last-prompt-token hiddens across middle layers from the PPO checkpoint
- [ ] Build the vector per layer; confirm unit norm, correct sign, non-degenerate separation
- [ ] Fit τ as the 25th percentile of training projections
- [ ] **Verify the residual hook actually fires**, that α=0 is a true no-op against unsteered
      generation, and that the hook is removed afterwards
- [ ] Sweep α ∈ {−2, …, +2}; measure flip rate ×3 at temp 0.7
- [ ] Test the capping variant alongside the additive one
- [ ] Measure MMLU and IFEval at each α — restoring quality by damaging capability is not a result
- [ ] Report the α-response shape; do not assume it is monotone
- [ ] Publish the vector if it works — a documented "steering does not work at 1.5B" closes the
      line just as well

> **Pull `results/` off the volume before any teardown.** Nothing on the box persists.
