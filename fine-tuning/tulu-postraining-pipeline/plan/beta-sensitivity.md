# Beta sensitivity and preference displacement

**Question.** How does DPO's β move the KL/quality trade-off, and does training push the policy away
from *both* sides of the preference pair?

Two DPO arms at β ∈ {0.05, 0.1} on identical data. β controls how far the policy may drift from the
reference; the expectation is that the lower β drifts further, spends more KL, and shifts style more.

Starts from the SFT checkpoint — see [stage-attribution.md](stage-attribution.md) for the box and
environment.

## Preference displacement

The second half of this experiment is a specific pathology: **both chosen and rejected log-probs
falling**. That means the policy moved away from the whole pair rather than learning to prefer one
side — not the same as the ordinary case where rejected falls and chosen holds.

Detecting it requires logging chosen and rejected log-probs **separately** during training. The
detector compares end vs start, so a dip that recovers is not a fall.

A useful sanity check at init: DPO's step-1 loss should be exactly `ln(2)` = 0.6931, because the
policy still equals the frozen reference. A drifting or mismatched reference does not land there.

## Data and training constraints

- 10K pairs from `train_prefs`, decontaminated, **disjoint from the RM 20K** on `prompt_id`
- lr 5e-7, 1 epoch, max_len 2048, cached reference log-probs (~50% peak-memory saving)

## Code

- `src/trainers/dpo.py` — β from config, cached reference log-probs
- `src/analysis/beta_plots.py` — β-vs-KL/win-rate plot, displacement detector and curves
- `scripts/analysis/beta_plots.py` — CLI

## Steps

- [x] β wired from config; cached reference log-probs confirmed working
- [x] Displacement detector implemented and tested against all four cases
- [x] DPO smoke on real pairs — init loss landed on `ln(2)` exactly
- [ ] Prepare DPO 10K **locally on CPU**, asserting disjointness from RM, before renting a GPU
- [ ] DPO smoke on GPU before the full arms
- [ ] Train DPO β=0.05 from the SFT checkpoint (resumable, hub, W&B)
- [ ] Train DPO β=0.1 from the same checkpoint and the same 10K pairs
- [ ] Confirm both arms logged `logps/chosen` and `logps/rejected` per step
- [ ] Measure KL from the reference for each arm
- [ ] Judged win-rate vs SFT for each arm on the fixed 500-prompt set, one pass
- [ ] Plot β vs KL and β vs win-rate
- [ ] Run the displacement detector on both arms; report whether it fires and on which
