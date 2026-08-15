# Portfolio & Roadmap — goal-oriented bets

Each bet has a **thesis**, a single **headline target metric** on a named target, a **releasable
artifact**, and an explicit **win / lose-but-ship** condition. Pedagogy is a byproduct, not the goal.

**Four project specs, four bets.** The credibility layer is not a separate project — it is built into
the two projects that need it most, so every headline number is produced by the harness that measures
it.

---

## Bet 2 — SOTA-for-size reasoning, proven clean  *(do first: cheap, credible, momentum)*

- **Thesis:** cheap RLVR gives `Qwen2.5-1.5B-base` a *real* GSM8K/MATH pass@1 lift — and it's clean,
  because a random-reward control rules out the contamination that inflates most small-model RL claims.
- **Target metric:** GSM8K pass@1 (+ MATH-500) for the 1.5B **base**, headline "+X pts over base,
  pass@8 ~flat," with the random-reward control ≈ 0.
- **Artifact:** the RL-tuned model (HF) + the contamination-control recipe + **the released
  trusted-number harness** (×3 variance protocol, 8-gram decontamination, random-reward probe,
  format-sensitivity sweep) that the other bets report through.
- **Win:** statistically-clean +10–20 pts pass@1 **over the strongest inference-time-scaled baseline**,
  random-reward null. **Lose-but-ship:** a *positive* random-reward result is itself a valuable
  contamination finding on Qwen bases — and "cheap prompting beats RL at 1.5B" is equally publishable.
- **The baseline is the claim.** Published MATH-500 figures for a small base model: greedy 15.2%,
  chain-of-thought prompting alone **40.6%**, self-consistency(10)+CoT **52.0%** — versus **47.4%** for
  a GRPO-trained model. Inference-time scaling *beat* training. Bet 2 therefore reports against the
  best cheap baseline at matched inference compute, never against greedy decoding.
- **Feeds from:** `rlvr-grpo-math`.
- **Cost:** ~£178. **Sequence: 1st after the base recipe.**

## Bet 1 — Agentic tool-use unlock, packaged as a Harbor benchmark  *(flagship)*

- **Thesis:** RL turns a 1.5B model that *fails* at multi-turn tool-use into one that succeeds where
  prompting alone can't — and we release a small benchmark to prove it.
- **Target metric:** task-success lift over the prompted baseline. De-risk on **ALFWorld** (cheap,
  rule-based) → transfer to **τ-bench-mini** (~30–50 curated tasks) → package the win as a
  **10–20-task Harbor micro-benchmark** with a leaderboard (frontier vs small vs our RL-tuned small).
- **Harbor is scoped to *benchmarking/eval only* — never RL training.** Run it via Harbor's
  **Local Docker** backend on the Mac (no cloud provider = no sandbox fee; only model-API/served-model
  cost). **All RL training stays on the Docker-free envs** (ALFWorld, τ-bench-mini in-process), which
  also avoids the Docker-in-a-pod friction. A Harbor custom-agent/model adapter points at the pod's
  vLLM OpenAI endpoint to eval our 1.5B.
- **Artifact:** the RL-tuned agent model **+ the released Harbor benchmark + leaderboard**.
- **Win:** measurable success lift on a held-out agentic eval + a usable benchmark. **Lose-but-ship:**
  the benchmark itself + a documented "1.5B ceilings at X on agentic tool-use" negative result.
- **Feeds from:** `agentic-rl-alfworld` + `verifiers`/GRPO (training) + **Harbor** (eval only).
- **Cost:** ~£32 — ALFWorld's environment and reward are rule-based, so there is no auxiliary model to
  serve; **Harbor eval ≈ free** (Local Docker on the Mac). Cheapest bet in the programme.
  **Sequence: 2nd (flagship).**

## Bet 3 — Does on-policy/self distillation actually beat offline at 1.5B?  *(method finding)*

- **Thesis:** OPD/MOPD/OPSD beat offline sequence-KD at 1.5B on reasoning — or a clean null; either is
  a publishable small-scale result few people have.
- **Target metric:** GSM8K pass@1 across offline-KD / OPD / MOPD / OPSD, controlled **per token and
  per prompt**, with a signed CI-backed verdict. The two views disagreeing is itself a result.
- **Artifact:** the comparison + a released distilled 1.5B reasoner.
- **Win:** a decisive verdict either direction with error bars. **Lose-but-ship:** the clean null is
  still a contribution.
- **Feeds from:** `distillation-bakeoff` — **four arms, not eight**: one offline baseline (A) against
  single-teacher on-policy (C), multi-teacher (E) and self-distillation (F). OPSD needs no teacher at
  all, making it the cheapest arm and the most novel claim.
- **Cost:** ~£89. The largest remaining line and the most cuttable — a method finding, not a released
  capability. **Sequence: 3rd, and the natural stopping point if budget runs out.**

## Credibility — built in, not bolted on

Every headline number ships with error bars, a contamination check, and a KL-vs-quality context, so
our own results aren't the fragile leaderboard numbers we're critiquing. This lives inside the work:

- **`rlvr-grpo-math`** owns the **trusted-number harness** — variance/stability table, format-swing
  sweep, n-gram decontamination, perturbation probe, **lm-eval vs Inspect AI** bake-off, and the
  ×3-runs-with-CIs checklist. Its random-reward control *is* the spurious-rewards contamination
  probe: one experiment, costed once, serving both.
- **`tulu-postraining-pipeline`** owns the **KL frontier** — the Goodhart inverted-U, BoN-vs-PPO at
  equal KL, mitigation arms, and the released **sycophancy steering vector** plus a reusable "how far
  can I push this RL run" utility that bounds every RL run in Bets 1–3. It also owns the **judge-bias
  report**, without which every judged number in the programme carries an unquantified confound.

---

## Supporting input (artifact, not a standalone bet)

- `tulu-postraining-pipeline` → the **base recipe, the data engine, and the KL frontier**. Produces the
  SFT/RM/DPO/PPO checkpoints the other bets start from, the released decontaminated **Tülu-mini SFT +
  preference set**, the stage-attribution table, the DPO-vs-PPO and RS-SFT-vs-DPO verdicts, the
  judge-bias report, and the over-optimization utility. **Cost: ~£184** (+~£10 for two gated arms).

> **This project is a single point of failure for three bets** — it owns the RM, the PPO loop, the 32B
> judge and the datasets. That concentration is deliberate (the machinery is genuinely shared) but it
> means sequencing it **first** and shipping its checkpoints early, before anything downstream blocks.

---

## Budget

| Project | GPU-hrs | Spec £ | At current rates |
|---|---:|---:|---:|
| `tulu-postraining-pipeline` — recipe + data engine + KL frontier | ~62 | £145 | **~£184** ⚠️ |
| `rlvr-grpo-math` — Bet 2 + trusted-number harness + inference-time ladder + verifier fidelity | ~62 | £140 | **~£178** |
| `distillation-bakeoff` — Bet 3, four arms | ~26 | £70 | **~£89** |
| `agentic-rl-alfworld` — Bet 1 (flagship) | ~21 | £25 | **~£32** |
| **Total** | **~171** | **£380** | **~£483** |
| Volume storage over the programme | | | ~£30 |
| **Budget** | | | **£400** |

**⚠️ `tulu-postraining-pipeline` is re-forecast from measured throughput** — see
[`tulu-postraining-pipeline/REFORECAST.md`](tulu-postraining-pipeline/REFORECAST.md). Phase R measured
every stage at 0.5B on a $0.16/hr RTX 3090. Two corrections, pulling opposite ways:

- **The four S1–S4 training lines are ~3x over-provisioned** — 19 spec-hours re-forecast to ~6
  (range 4.2–12.1). Worth roughly **-£16** at A100 rates.
- **Two lines change hardware, not hours.** PPO cannot run on a 24 GB card *at any model size*: TRL
  stacks `batch x response_length x vocab x 4` = 9.27 GiB of logits on top of four resident models,
  and vocab is 151,936 at both 0.5B and 1.5B. So **S9's 4 PPO mitigation arms (12 h) move off the
  RTX 4090 line onto A100**, about **+£8**. The 32B judge likewise needs 80 GB.

Net effect is a modest saving, but the *shape* matters more: training is cheaper than costed and some
24 GB lines are infeasible. Every production figure still crosses two untested hops (0.5B->1.5B,
3090->A100); the first hour on an A100 should re-measure SFT and PPO rather than trust these.

**Spec cost tables are priced low.** They assume **RTX 4090 ≈ £0.45/hr** and **A100 80 GB ≈ £1/hr**;
live RunPod *secure-cloud* rates are **4090 £0.58 / A100 SXM £1.25 / A100 PCIe £1.09 / H100 £2.59** —
~25–30% higher. Network volumes require `--cloud-type SECURE`, so community pricing is unreachable.
The 300 GB volume bills **~£17/mo** whether or not a pod runs — delete it between bets.

Also note **no volume-capable datacenter currently stocks both RTX 4090 and A100**, so the two-tier
cheap/expensive plan the specs assume may collapse to a single A100-class pod.

### The programme is ~£76 over, and consolidation cannot close it

Merging saved less than expected. The duplication between projects was mostly **infrastructure** —
pod setup, data prep, judge serving, eval scaffolding — which is a handful of GPU-hours, not the bulk
of the cost. The actual compute (BoN sweeps, variance runs, mitigation arms, teacher generation) is
real work that costs the same wherever it lives:

| Action taken | Expected saving | Actual |
|---|---:|---:|
| Fold the over-optimization work into tulu | ~£35 | **~£13** |
| Fold the eval-variance and contamination work into rlvr | ~£25 | **~£9** |
| Trim distillation from 8 arms to 4 | ~£50 | **~£40** |

**Scope cuts, not structural ones, are what remain.** In order of least damage:

1. **Stop after Bet 1** — the first three lines total ~£357 and deliver the base recipe, the
   trusted-number harness, a clean RLVR result and the flagship benchmark. Bet 3 is a method finding
   and is already sequenced last; treating it as budget-contingent is the honest plan.
2. Drop tulu's gated O13/O14 (~£10) — already conditional on O11 yielding a usable vector.
3. Take rlvr's own cut order — the unfiltered control first, then the β=0 ablation (~£29).

**Never cut, at any budget:** rlvr's random-reward control (the basis of Bet 2's "proven clean"
claim), tulu's per-stage evals (the attribution table *is* the deliverable), the judge-bias report,
and ×3 variance with CIs everywhere.

---

## Where each workload runs

Three platforms, chosen by **storage and credential needs — not by hourly rate.**

| Phase / workload | Platform | Why |
|---|---|---|
| **Phase R (Rung 1)** — install check, PPO/GRPO API, vLLM, judge, smokes, fidelity sweep | **Vast.ai** | Stateless, ≤0.5B models, hours not days, **no credentials needed**. Cheap tier is 2–4× under RunPod |
| Verifier audit, n-gram contamination scan, data prep | **Local** | CPU-only — do not rent a GPU to run string matching |
| **Training runs that must resume** — SFT, RM, DPO, PPO, GRPO arms | **RunPod** | The **detachable network volume is the resumability design**. Also the only place the HF **write** token belongs |
| **Judge / gold scoring / eval fan-out** — 32B judging, BoN scoring, head-to-heads | **Vast** or **Modal** | Vast: ≥90 GB cards well under first-party pricing. Modal: no idle cost, `.map()` fan-out. Generations write incrementally, so interruption is cheap |
| Capacity overflow when a GPU is out of stock | **Vast** | A marketplace nearly always has an A100 when a single provider does not |

**The rule that decides it:** *if losing the machine means losing the run, use RunPod.* A
Vast volume is bound to one physical host — the failure mode is "lose the machine, lose
everything since the last push", not "lose the pod, keep the volume".

**The rule that overrides it:** *never put a write-scoped HF token, GitHub PAT, or reusable
API key on a Vast box.* The host physically owns the hardware and Docker is not a security
boundary against them. Every model and dataset here is open, so the IP risk is nil — the
exposure is entirely credentials. Note `setup_secrets.sh` performs `hf auth login` with a
**write** token: that script is RunPod-only.

Setup, commands and gotchas for all three: the global `gpu-cloud-platforms` skill.

---

## Sequence (build order)

1. **`tulu-postraining-pipeline`** — base recipe, checkpoints and datasets three bets depend on;
   then its data engine and KL frontier.
2. **Bet 2** — cheap, credible RLVR reasoning number, built on the trusted-number harness. Momentum.
3. **Bet 1 (flagship)** — ALFWorld harness → τ-bench-mini capability → package as a Harbor benchmark.
4. **Bet 3** — the on-policy/self-distillation verdict. Budget-contingent.

**Definition of done for the whole program:** ≥1 released model per bet, ≥1 released dataset/benchmark,
a public trusted-number harness, and a short write-up per bet stating the claim, the number (with error
bars), and whether we won or lost. Shipped bets > tutorials.
