# dataset eda findings

- `tulu3_sft_mixture_eda.ipynb` — `allenai/tulu-3-sft-mixture`
- `ultrafeedback_eda.ipynb` — `HuggingFaceH4/ultrafeedback_binarized`
- `eval_sets_eda.ipynb` — reward-bench, alpaca_eval, IFEval, MMLU

---

#### 1. tulu-3 sft mixture → 25k subset

**schema:** `{id, messages, source}` chat turns. ~939k rows, 19 sources.

**stratify by `source`.** mixture is skewed (largest source ~16%); do not sample uniformly over sources.

**source families (full split):**

| family | ~pct |
|---|---|
| math | 35.6% |
| general_ift | 26.8% |
| code | 15.1% |
| multilingual | 10.6% |
| safety | 6.5% |
| preference_ish | 5.3% |

safety / preference-style content is already inside this IFT mix — not pure task SFT. proportional stratify inherits that; downweight only if Phase 1 explicitly decides to.

math and code have the highest joint distribution at about 51% of the entire dataset, when splitting for the training run, we need to explicitly select highly for math and coding as we are trying to maximize behaviour on verifiable tasks

**filters before sampling:**

1. drop **716 empty-assistant** rows (all in `tulu_v3.9_synthetic_finalresp_wildguardmixtrain_decontaminated_50k`).
2. candidate drop **weak assistant** (`assistant_chars < 50`) — ~8.6% / 80.9k rows; confirm policy in P1-4 (drop vs keep).
3. length tail: on 2k templated Qwen tokens, `frac_gt_4096 ≈ 0.0115` (p99≈4934, max≈15467) — truncate or filter; do not assume all fit `max_len=4096`.

**turns:** mostly single exchange (`frac_multi_turn≈0.046`, `frac_last_assistant=1.0`, system almost never). consecutive double-user: 0.

**template:** spot-check Qwen `apply_chat_template` renders each role once.

**Phase 1 action:** empty-assistant drop → optional weak drop → 8-gram decontam → stratified 25k by `source` → handle `>4096`.

---

#### 2. ultrafeedback → rm / dpo / ppo

**splits:**

| split | n | use |
|---|---|---|
| `train_prefs` | 61,135 | RM 20k + DPO 10k (disjoint) |
| `test_prefs` | 2,000 | PPO 1.5k prompts only |

budget OK: 20k+10k ≤ 61k; 1.5k ≤ 2k. train/test split already separates preference train from PPO prompts.

**schema:** `{prompt, prompt_id, chosen, rejected, messages, score_chosen, score_rejected}`. chosen/rejected are chat message lists — use last assistant turn for length/scores.

**scores (train_prefs):** mean accept ≈7.83, reject ≈5.95, margin ≈1.87 (test similar). chosen tends longer than rejected (p50 chars 981 vs 797) — chattiness confound for later O3.

**filters:**

- sample RM then DPO by unique `prompt_id`; assert empty intersection.
- drop empty chosen/rejected sides before sampling.
- pairs with `score_margin ≤ 0` are weak BT/DPO signal — consider drop (margin min is 0 in train summary).

**PPO:** sample 1500 prompts from `test_prefs`; ignore preference labels at train time.

**Phase 1 action:** id-disjoint RM/DPO from `train_prefs`; PPO prompts from `test_prefs`.

---

#### 3. eval sets → decontam + gate design

analyze **per `split_id`** — never merge MMLU splits for stats or conclusions.

**roles / sizes:**

| split_id | n | role | decontam field |
|---|---|---|---|
| `reward_bench_filtered` | ~2985 | RM gate (prefer over `raw`) | `prompt` |
| `reward_bench_raw` | ~5123 | diagnostic only | `prompt` |
| `alpaca_eval_eval` | 805 | judged H2H prompts | `instruction` |
| `ifeval_train` | 541 | IF / format guardrail | `prompt` |
| `mmlu_test` | ~14k | skills / broken-run | `question` |
| `mmlu_validation` / `mmlu_dev` | ~1.5k / ~285 | skills | `question` |
| `mmlu_auxiliary_train` | ~100k | **exclude** from default bank | — |

**reward-bench:** gate on **chat** subsets (≥65–70%); score `prompt+chosen` vs `prompt+rejected` at eval time (no score column). prompt “dups” (~13% of filtered rows) are intentional — same prompt across `alpacaeval-*` / `mt-bench-*` with different pairs; do not collapse before scoring. exact triple copies ≈0.

**alpaca_eval:** columns `instruction`, `output`, `generator`, `dataset`. only `instruction` is the eval prompt; `output` is davinci reference, not a gold label. H2H = our models + local judge, not the `alpaca-eval` package.

**IFEval:** columns `key`, `prompt`, `instruction_id_list`, `kwargs`. 25 verifiers / 9 categories; non-null `kwargs[i]` → args for that verifier. **do not reimplement** — use `lm-eval` task `ifeval` (`lm_eval.tasks.ifeval`).

**MMLU:** `{question, subject, choices, answer}` — exact/MCQ correctness, no instruction ids. gold is A–D (`ClassLabel`).

**cross-prompt overlap (exact normalized):** `reward_bench_filtered` ∩ `alpaca_eval_eval` ≈102 prompts (expected; RB alpacaeval subsets). IFEval ∩ others ≈0. tiny MMLU internal overlap.

**8-gram bank:** whitespace 8-grams on decontam text fields. `ngram_instances` = sliding windows counted with multiplicity; `unique_8grams` = set size (bank proxy). default bank = all split_ids except `mmlu_auxiliary_train`.
---

#### 4.checklist (from eda)

1. SFT: drop empty assistants → optional weak-assistant filter → decontam → stratify 25k by `source` → truncate/filter `>4096`.
2. RM 20k + DPO 10k: disjoint `prompt_id`s from `train_prefs`; drop empty sides; optional margin filter.
3. PPO 1.5k: prompts from `test_prefs` only.
4. Decontam bank: RB filtered prompts + alpaca instructions + IFEval prompts + MMLU test/val/dev questions (not auxiliary_train).
5. Eval wiring later: RewardBench-chat gate; IFEval via lm-eval; MMLU via lm-eval; Alpaca instructions for local judged H2H.
