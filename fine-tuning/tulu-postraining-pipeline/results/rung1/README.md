Artifacts from the cheap-GPU validation run, kept because the box was destroyed and they
cannot be regenerated without renting another one.

All of it is 0.5B smoke output, not results. The real 1.5B outputs land in `metrics/`,
`plots/` and `checkpoints/`.

| File | What it shows |
|---|---|
| `generations_sft_smoke.jsonl` | 32 completions from the SFT smoke checkpoint |
| `generations_sft_smoke_second_checkpoint.jsonl` | 32 from a second checkpoint — byte-identical to the first |
| `judgments_position_bias.jsonl` | the same 32 pairs judged. both sides identical, so every non-tie is pure position bias: the judge disagreed with itself on 17 of 32 based only on which slot the text occupied |
| `mmlu_285_questions.json` | `acc_stderr` 0.029 at 285 questions — the measurement that set the MMLU sample size |
| `ifeval_10_questions.json` | `acc_stderr` 0.133 at 10 questions — why IFEval runs its full set |

Findings are written up in `RUNGREPORT.md`.
