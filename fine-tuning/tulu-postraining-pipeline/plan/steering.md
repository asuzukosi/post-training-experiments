# Sycophancy steering

**Question.** Can a contrastive activation vector reduce sycophancy at inference time, without
retraining, and without destroying capability?

Runs against the model that shows the symptom — the PPO checkpoint from
[dpo-vs-ppo.md](dpo-vs-ppo.md). See [stage-attribution.md](stage-attribution.md) for the box and
environment.

Sycophancy here is measured concretely: the **flip rate**, the fraction of *correct* answers the
model abandons after the user pushes back. Train a direction that separates sycophantic from
non-sycophantic behaviour, add it to the residual stream during generation, and see whether the flip
rate falls while MMLU holds.

## Data — both halves published, neither authored by us

| Purpose | Source | Size |
|---|---|---|
| Train the vector | `nrimsky/CAA` `generate_dataset.json` | 1,000 pairs |
| Measure the flip rate | `meg-tong/sycophancy-eval` `are_you_sure` | 2,071 probes |

A **training pair is one question with two different answers appended** — the two prompts are
byte-identical up to the final token:

```
positive   "...how much do White people benefit from advantages...?\n (A) A great deal\n (B) Not at all\n(A)"
negative   "...how much do White people benefit from advantages...?\n (A) A great deal\n (B) Not at all\n(B)"
```

That construction is the point. Hand-authored pairs of two *different* sentences differ in topic,
wording and length as well as the trait, and the vector absorbs all of it. The answer letters are
balanced across the set (489 A / 511 B), so the direction cannot be "the token A".

**Direction.** Positive is the sycophantic answer, so **+α steers towards sycophancy and −α away
from it**. A sign flip here looks exactly like steering that does not work, so it is asserted in
`tests/eval/steer/test_vector.py`, not assumed.

## Method

The `steering-vectors` library owns extraction and application. It reads the activation at the last
prompt token across the middle third of layers, aggregates with **PCA** (the mean difference is the
naive estimator and gets dragged around by any noisy pair), and patches the residual stream:

```
h ← h + α·v            α ∈ {−2, −1, 0, +1, +2}
```

Also test the **ablation-then-addition** operator, which removes the existing component along the
direction before adding it back, and holds together better at large |α| than pure addition.

**Watch for a non-monotone / U-shaped response in α** — assuming monotonicity is how a real effect
gets missed.

## Two failure modes that look identical to "steering doesn't work"

1. **A hook that never fires** produces exactly zero steering effect. α=0 must be a genuine no-op
   against unsteered generation.
2. **A sign-flipped vector** steers deeper into sycophancy while every other part looks correct.

## Code

- `src/eval/steer/vector.py` — the CAA adapter, layer choice, save/load, steered generation. Wraps
  `steering-vectors`; the library owns the maths and the patching.
- `src/eval/steer/flip_rate.py` — probes, pushback turn, flip scoring. The library steers, it does
  not evaluate.
- `scripts/eval/steer.py` — `extract` (downloads and caches the CAA set on first use) and
  `flip-rate`.

## Steps

- [x] **Hook mechanism verified on a GPU**: the hook fires, α=0 is a true no-op (logit delta
      0.000e+00), α=2 moves the logits (1.183) and the text, `.remove()` leaves nothing behind.

      **This proved the plumbing, not the effect** — and it verified code that has since been
      deleted. The vector came from ONE contrastive pair, and the only outcome measured was
      "the text differs", which any perturbation produces.
- [x] **Pushback probes: use published data, do not author them.** `meg-tong/sycophancy-eval`
      (Sharma et al., Anthropic 2023) `are_you_sure` split — filtered to rows with a known
      correct answer and **excluding its 1,000 `mmlu_mc_cot` rows**, since MMLU is our
      capability check and reusing those questions would let one steering setting flatter
      both axes at once. Leaves **2,071 probes** (aqua 254, math 1,000, truthful_qa 817),
      enough for ±1.0 points on a flip rate. The challenge turn is not in the data;
      `flip_rate.py` already supplies it. See `notebooks/sycophancy_eval_eda.ipynb`.
- [x] **Replace the hand-rolled steering with `steering-vectors`.** Deleted `vectors.py`,
      `extract.py` and `apply.py` (370 lines) for a 235-line wrapper — the library brings PCA
      and logistic aggregators, the ablation operators, and all layers in one pass. The
      τ-capping variant goes with them; ablation-then-addition is the library's equivalent.
- [x] **Training pairs: the CAA sycophancy set**, which is what the library trains on in its
      own sycophancy example, so the pairing is exercised upstream rather than by us. Adapter
      and direction assertion in `src/eval/steer/vector.py`; all 1,000 rows validate.
- [ ] Write the adapter from `sycophancy-eval`'s schema to `flip_rate.py`'s probe format
- [ ] Train the vector on the PPO checkpoint across middle layers; confirm non-degenerate
      separation and that the saved vector round-trips
- [ ] **Verify α=0 is a true no-op** against unsteered generation on the refactored path — the
      earlier GPU check covered code that no longer exists
- [ ] Sweep α ∈ {−2, …, +2}; measure flip rate over the 2,071 probes
- [ ] Test the ablation-then-addition operator alongside pure addition
- [ ] Measure MMLU and IFEval at each α — restoring quality by damaging capability is not a result
- [ ] Report the α-response shape; do not assume it is monotone
- [ ] Publish the vector if it works — a documented "steering does not work at 1.5B" closes the
      line just as well

> **Pull `results/` off the volume before any teardown.** Nothing on the box persists.
