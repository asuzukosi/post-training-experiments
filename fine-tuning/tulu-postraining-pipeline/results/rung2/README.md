Artifacts from the second cheap-GPU validation run (RTX 3090, $0.172/hr).

Verified two things that unit tests cannot reach:

**The mid-training MMLU tripwire.** lm-eval's `HFLM` accepts a model already in memory,
so no second copy and no engine handover is needed. Measuring peaked at 5.6 GB of 24,
and training continued afterwards with loss still falling (0.342 -> 0.281 -> 0.129).
The tripwire itself cost ~3 minutes at 1 question per subject.

**The steering hooks.** The residual hook fires, alpha=0 is a true no-op (logit delta
0.000e+00), alpha=2 changes the logits (1.183) and the generated text, the hook is gone
after `.remove()`, and `cap_hidden` clamps the projection to exactly tau.

| File | What it is |
|---|---|
| `tripwire_mmlu_smoke.jsonl` | the tripwire's own record, written next to the checkpoint |
| `tripwire_smoke_trainer_state.json` | the training log proving the run continued after the check |

Both from a 0.5B smoke, not results.
