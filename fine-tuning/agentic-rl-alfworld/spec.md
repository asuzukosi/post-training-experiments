# Agentic RL in a Text-Game Environment (ALFWorld)

Engineering spec for a self-contained experiment: train a small model with **multi-turn** reinforcement
learning inside a stateful, interactive **text-game environment** (ALFWorld). Unlike single-turn
verifiable-reward RL (one prompt → one answer → checker), here the agent takes a sequence of actions
in a world that changes in response, and is rewarded on task completion. A tiny hand-built TextWorld
env is included first to validate the harness end-to-end.

---

## Bet & success target

Serves **Bet 1 — agentic tool-use unlock, packaged as a Harbor benchmark *(flagship)*.**
- **Thesis:** RL turns a 1.5B model that *fails* at multi-turn tool-use into one that succeeds where prompting can't — and we release a small benchmark to prove it.
- **Headline target:** **ALFWorld task-success +≥15 pts over the prompted baseline** (±std over held-out seeds), transfer measured on **τ-bench-mini**, then a **released 10–20-task Harbor micro-benchmark + leaderboard** (frontier vs small vs our RL-tuned small).
- **Artifact to ship:** the RL-tuned agent model (HF) **+ the released Harbor benchmark** (run locally via Harbor's free Local Docker backend).
- **Scope:** RL training stays on Docker-free envs (ALFWorld / τ-bench-mini); **Harbor is eval/benchmark only.**
- **Win / lose-but-ship:** a clear success lift + a usable benchmark = win. If 1.5B ceilings out, the **benchmark itself + a documented "small models cap at X on agentic tool-use"** is still a shipped contribution.

---

## 0. Overview

An **agentic RL environment** is a multi-step MDP: `reset()` gives an initial observation, the agent
emits an action, `step(action)` returns a new observation + reward + done, and this repeats until the
task is solved or a step limit is hit. **ALFWorld** is an ideal small target: it's a **rule-based**
text-game engine (built on TextWorld) with household tasks (find/clean/heat/cool/place objects), short
text observations, a list of admissible actions per step, and a **programmatic goal-completion
reward** — so the environment needs no auxiliary LLM (unlike LLM-user-simulator benchmarks), which
keeps it cheap. The novelty here versus the other training projects is that we build the
**environment/harness + reward loop**, not just a training loop; the optimizer (GRPO) is standard.

---

## 1. Objectives

1. **O1 — Does multi-turn RL work here?** Does GRPO raise ALFWorld **task success rate** over the
   base prompted agent at 1.5B?
2. **O2 — Reward design:** sparse **terminal** reward (goal reached) vs light **subgoal shaping** —
   which trains better / is more sample-efficient at this scale?
3. **O3 — Generalization:** train on the ALFWorld training games, evaluate on the **held-out /
   out-of-distribution** eval games — how much does RL transfer to unseen layouts?
4. **O4 — Harness validation (toy env):** build a minimal hand-written TextWorld task and verify the
   full `reset`/`step`/reward + rollout-masking loop and GRPO training on it before ALFWorld.

5. **O5 — Group degeneracy (the thing most likely to silently kill this run):** track the **fraction
   of rollout groups with zero advantage** at every step, per arm.

> **Why O5 is not optional here.** GRPO computes advantages by subtracting the group mean. **If every
> rollout in a group earns the same reward, all advantages are exactly zero and the step contributes
> no gradient at all.** A sparse *terminal* reward on a 1.5B agent that starts out failing nearly every
> ALFWorld task is precisely that case: most groups will be all-zero-reward, training will look like it
> is running — loss curves, wall-clock, GPU utilisation all normal — and almost nothing will be
> learning. This is the mechanistic reason O2 exists: **subgoal shaping's real job is to break group
> ties**, not merely to "help". Report the zero-advantage fraction next to the success rate for both
> arms; if Arm A sits near 100%, the sparse arm has failed for a structural reason rather than because
> RL does not work here, and that distinction is the finding.

**Expected outcomes:** RL lifts success rate meaningfully over the prompted base; sparse reward works
but subgoal shaping improves sample efficiency **mainly by reducing zero-advantage groups**; partial
transfer to unseen games (some drop OOD); the toy env confirms the harness and catches masking/format
bugs cheaply.

---

## 2. Environment & prerequisites

**Compute:** RunPod. Network volume **`posttraining-data`** — **deleted 2026-08-12; it does not
exist.** Recreate it (250 GB, US-CA-2) *only* when starting a full-scale run — see
[`../VOLUME.md`](../VOLUME.md) for the exact create + repopulate + re-secrets recipe. **Rung 1 (§4b)
needs no volume.** Once created: model cache at `/workspace/hf`.

**Launch a pod** (A100 80 GB for GRPO training + vLLM rollouts; the text env itself runs on CPU):
```bash
runpodctl pod create --name agentic-rl \
  --gpu-id "NVIDIA A100-SXM4-80GB" --gpu-count 1 \
  --data-center-ids US-CA-2 --cloud-type SECURE \
  --network-volume-id <NEW_VOLUME_ID> --volume-mount-path /workspace \
  --container-disk-in-gb 40 --ports "22/tcp" --ssh \
  --image "pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel" \
  --terminate-after "$(date -u -v+6H +%Y-%m-%dT%H:%M:%SZ)"
```
On every pod: `export HF_HOME=/workspace/hf HF_XET_HIGH_PERFORMANCE=1`.

> **Python 3.12 is required** — `judgearena` needs `>=3.12`, so the `py3.11` RunPod image will not
> install the stack. Create the env first: `conda create -y -n py312 python=3.12`, then use
> `/opt/conda/envs/py312/bin/python`. Verified on 3.12.13 / CUDA 12.4.

**Constraint:** open-weights only — no gated models.

**Model (cached):**

| Role | Model | Notes |
|---|---|---|
| Policy / agent | `Qwen/Qwen2.5-1.5B-Instruct` | chat-formatted; needs to follow the action/tool format |
| Reference (KL) | frozen copy of the policy | standard GRPO KL anchor |

**Environment (not on the volume — installed at runtime; both fully open, no LLM inside):**
```bash
pip install alfworld textworld verifiers
export ALFWORLD_DATA=/workspace/alfworld_data   # persist game data on the volume
alfworld-download                                # one-time; downloads game files to $ALFWORLD_DATA
```
ALFWorld provides `reset()`/`step()`, text observations, per-step **admissible commands**, and a
goal-completion reward. The tiny toy env (O4) is a hand-written TextWorld game generated with the
`textworld` package (no download).

*(Standard secrets, checkpointing, conventions, and naming blocks below — identical policy to the
other projects.)*

**Secrets, code & artifacts** — set up once; everything persists on the volume.

*Keys (one-time, on a pod with the volume mounted):*
```bash
export HF_HOME=/workspace/hf
hf auth login --token hf_XXXX          # write scope; token persists at /workspace/hf/token
cat > /workspace/.env <<'EOF'
export HF_HOME=/workspace/hf
export HF_XET_HIGH_PERFORMANCE=1
export ALFWORLD_DATA=/workspace/alfworld_data
export WANDB_API_KEY=XXXX               # Weights & Biases (only key not persisted by a login)
EOF
chmod 600 /workspace/.env
```
*Every pod (one line; wrapped by `scripts/setup_env.sh`):* `set -a; source /workspace/.env; set +a`
— authenticates `hf` + `wandb`, points HF at the cache, and sets the ALFWorld data path.

*Code — clone the repo onto the volume so it persists; later pods just pull:*
```bash
cd /workspace && git clone https://github.com/<you>/<repo>.git
git config --global credential.helper 'store --file /workspace/.git-credentials'   # PAT saved on volume
# later pods: cd /workspace/<repo> && git pull
```

*Artifacts out —* models → HF Hub (`push_to_hub=True`, `hub_model_id="<you>/<base>_<task>_<datetime>"`,
`hub_private_repo=True`, or `hf upload <you>/<repo> results/checkpoints/<run>`); metrics/plots → W&B;
files → `scp -i ~/.runpod/ssh/runpodctl-ssh-key -P <PORT> -r root@<IP>:/workspace/<repo>/results ./`
(ip/port from `runpodctl ssh info`) or `runpodctl send`/`receive`. `results/` persists on the volume regardless.

> Plaintext secrets on a shared volume: `chmod 600 /workspace/.env`, never commit it (commit `.env.example`).

**Checkpointing & resume.** GRPO runs are long, so checkpoint frequently to the volume — a failed or
killed node loses nothing: launch a new pod, re-mount the volume, resume. In the trainer config:
`output_dir=results/checkpoints/<run>`, `save_strategy="steps"`, `save_steps=25–50`,
`save_total_limit=3`, launch with `resume_from_checkpoint=True` (restores policy + optimizer + step).
Fix the W&B run id and set `resume="allow"` so reward/success curves continue on the same run. The
rollout collector writes completed episodes **incrementally** to the volume and **skips already-run
task seeds** on restart. Because `output_dir` is on the network volume (persists across pods), resume
survives full node loss — so cheaper interruptible/spot GPUs are viable.

**Conventions.** Pin a known-good dependency set in `requirements.txt` (TRL ↔ transformers ↔ vLLM ↔
verifiers ↔ alfworld are version-sensitive) and install from it — never `pip install -U` mid-project.
**Smoke-test** on the toy TextWorld env (O4) and a few dozen ALFWorld seeds before the full run.
**Evaluation:** the metric is **task success rate** from the environment itself (programmatic) — no
LLM judge and no `lm-eval` needed in this project.

**Output naming.** Every trained checkpoint (and its W&B run) is named `<base_name>_<task>_<dateandtime>`
— `<base_name>`=`qwen2.5-1.5b-instruct`, `<task>` the run/arm, `<dateandtime>` a filesystem-safe UTC
stamp `YYYYMMDDThhmmZ`. Save under `results/checkpoints/`. Examples:
`qwen2.5-1.5b-instruct_grpo-alfworld_20260809T1730Z`, `..._grpo-alfworld-shaped_...`,
`..._grpo-toy_...`.

---

## 3. Approach

```
Toy TextWorld env (hand-built) ──► validate reset/step/reward + rollout masking + GRPO loop   (O4)
        │  (once green)
        ▼
ALFWorld (alfworld pkg, TextWorld backend)
   │  reset() → obs + admissible commands;  step(action) → obs', reward, done
   ├─► Arm A: GRPO, sparse terminal reward (success=1)                                 (O1)
   ├─► Arm B: GRPO, + subgoal shaping (e.g. correct object picked up)                  (O2)
   └─► eval: success rate on TRAIN games vs held-out OOD games                         (O3)
```

**Tooling:** `verifiers` (environment API + multi-turn rollout that wraps ALFWorld and plugs into
GRPO), `TRL GRPOTrainer` as the optimizer, `vLLM` for fast rollout generation, `alfworld` + `textworld`
(the environments), `W&B` (reward, success rate, episode length, invalid-action rate).

**3.1 Rollout loop (per episode).** `obs, valid = env.reset(seed)`; loop: build the prompt (task +
current observation + admissible commands + short history), agent generates the next command
(optionally a short `<think>` then the command), parse it, `obs, reward, done = env.step(cmd)`, append,
repeat until `done` or `max_steps` (e.g. 40). **Mask environment/observation tokens from the loss** —
train only on the agent's action tokens. Truncate old history to cap context growth.

**3.2 GRPO training (Arm A, O1).** Sample **G=8** trajectories per task seed (temp 1.0);
trajectory reward = terminal success (broadcast across the episode's action tokens); group-relative
advantage; KL to the frozen reference; lr 1e-6; ~300–500 steps.

**Track these from step 0, not as an afterthought** (they are what make a stalled run
diagnosable rather than merely disappointing): **% zero-advantage groups** (O5), **policy entropy**
(collapse warning), the **advantage-magnitude distribution**, and **clip fraction** — a clip fraction
drifting away from ~0 means the policy is moving faster than the trust region allows.

**Set the rollout token budget explicitly and log it with every result.** Published single-turn GRPO
results move 43.2% → 45.6% → 47.4% purely from raising the per-rollout token cap 256 → 512 and
rollouts 4 → 8 at a fixed step count — a larger swing than most ablations here. Multi-turn episodes
concatenate many observations and actions, so the cap binds *harder* than in single-turn RLVR: if it
truncates before the episode terminates, the rollout earns no reward and the group degenerates (O5).

Track success rate, mean episode
length, and invalid-action rate over training.

**3.3 Reward shaping (Arm B, O2).** Add small dense subgoal rewards (e.g. +0.2 for reaching a required
sub-state) on top of the terminal reward; compare success and **sample efficiency** (success vs steps)
against Arm A at equal compute.

**3.4 Generalization (O3).** Evaluate the trained policy on the ALFWorld **held-out / OOD** eval games
(unseen layouts) vs the training games; report the gap.

**3.5 Toy env (O4).** A minimal TextWorld game (e.g. "find the key, unlock the box, take the coin")
generated with the `textworld` package; use it to smoke-test the harness, masking, reward wiring, and
the GRPO loop before spending compute on ALFWorld.

---

## 4. Experiments & metrics

| Exp | Compares | Primary metric | Secondary |
|---|---|---|---|
| E0 (O4) | toy env before vs after GRPO | toy task success | harness sanity (masking, parse, reward) |
| E1 (O1) | base prompted vs GRPO (Arm A) | ALFWorld success rate (train games) | mean episode length, invalid-action rate |
| E2 (O2) | Arm A (sparse) vs Arm B (shaped) | success rate + success-vs-steps curve | reward-hacking / degenerate-action checks |
| E3 (O3) | trained policy: train vs OOD eval games | OOD success rate; train→OOD gap | per-task-type breakdown |
| E4 (O5) | Arm A vs Arm B, per step | **% zero-advantage groups** | reward entropy; advantage magnitude distribution |

All eval success rates over a fixed set of held-out seeds, ×3 runs, mean ± std.

---

## 4b. Rung-1 dry run — validate before committing

**No full-scale run starts until this passes.** Rung 1 is this project's own pipeline exercised at
**0.5B on a cheap CUDA box with no persistent volume** — run it on **Vast.ai**, where the 20–33 GB
tier is ~$0.07–0.19/hr against RunPod's ~$0.25–0.28:

```bash
vastai search offers 'num_gpus=1 gpu_ram>=20 gpu_ram<=33 rentable=true reliability>0.95' -o dph
```

Filter on `disk_bw` and `inet_down` as well as price — host quality on a marketplace varies by two
orders of magnitude, and a slow disk dominates model-download time. RunPod RTX 4000 Ada (~$0.28) is
the fallback. Budget: **a few hours, ~$1–3.**

> **Nothing secret goes on this box.** Vast hosts physically own their hardware, so no write-scoped HF
> token, no GitHub PAT, no `ssh -A`. Rung 1 needs none of them: models are public, and results are
> copied off at teardown. `setup_secrets.sh` is **RunPod-only** — it logs in a *write* token.
> Full platform guidance: the global `gpu-cloud-platforms` skill.

**Everything runs on the pod, not the laptop.** The dev machine has a different torch, no vLLM (no
macOS build), and MPS can change results — 90% on CPU and CUDA versus **80% on MPS** for the same
model and prompts. Running Rung 1 on a cheap CUDA box with the **exact pinned stack** removes every
one of those variables at ~$0.28/hr. The only work that stays local is pure-Python and CPU-only
(n-gram scans, the verifier audit), where no GPU library is involved.

It answers three things that estimates cannot:

1. **Does the pinned stack install?** `torch 2.5.1 + vllm 0.7.3 + trl 0.19.1 + transformers 4.51.0`
   has never been resolved together on any machine. It is the highest-probability failure in the plan
   and costs ~$0.10 to discover here instead of on an H100.
2. **Do the CUDA-only paths work?** vLLM generation and judging are only ever unit-tested against
   injected fakes on the dev machine, because vLLM has no macOS build.
3. **What is the real tokens/sec?** Every hour in §5 is an estimate. One measurement at 0.5B replaces
   them; scaling 0.5B → 1.5B is roughly **3× compute** (approximate — attention does not scale
   linearly with parameters).

**This project already has a Rung-1 stage built in.** Objective **O4 — harness validation** is
exactly this: a minimal hand-written TextWorld task verifying the loop before ALFWorld proper. Run it
on the cheap pod rather than the A100 it is currently costed against, and add:

| Check | Retires |
|---|---|
| O4 toy-env harness on `Qwen2.5-0.5B` | The multi-turn rollout loop, reward plumbing and episode accounting |
| **Zero-advantage rate on the toy env** | Whether groups degenerate *before* spending A100 hours discovering it on ALFWorld — the toy task is easy enough that groups should be mixed; if they are not, the harness or reward is wrong |
| GRPO ~20 steps against the toy reward | TRL 0.19.1 `GRPOTrainer` on multi-turn episodes; real sec/step for the 8–10 hour Arm A run |
| `alfworld-download` + env import | That the environment installs and its data fetches — pure CPU, but a hard blocker if it fails |

Because ALFWorld's reward is rule-based there is no auxiliary model to serve, so this project's Rung-1
run is the cheapest of the four — and it is the flagship bet, so validating its harness early has the
highest option value.

---

## 5. Compute & cost

| Stage | GPU | ~Hours |
|---|---|---|
| Toy env harness validation + smoke tests | 1× RTX 4090 24 GB | 2 |
| GRPO Arm A (sparse), ~300–500 steps | 1× A100 80 GB | 8–10 |
| GRPO Arm B (shaped) | 1× A100 80 GB | 6–8 |
| Eval (train vs OOD games, ×3) | 1× RTX 4090 24 GB | 3 |

- **Total:** ~19–23 GPU-hrs (env runs on CPU; GPU cost is policy generation + GRPO). **Picks:** A100
  80 GB (~£1/hr) for training, RTX 4090 (~£0.45/hr) for the toy env + eval.
- **Estimated cost: ≈ £25.** **Wall-clock: ~1.5 days.**
- Much cheaper than an LLM-user-simulator benchmark (e.g. τ-bench) because ALFWorld's environment and
  reward are rule-based — no auxiliary model to serve, short contexts.

---

## 6. Repo layout

```
agentic-rl-alfworld/
├── README.md                      # findings summary
├── pyproject.toml                 # package metadata + pinned deps
├── requirements.txt               # trl (GRPOTrainer), verifiers, vllm, alfworld, textworld, wandb
├── .env.example                   # WANDB_API_KEY, HF_HOME=/workspace/hf, ALFWORLD_DATA=/workspace/alfworld_data
├── .gitignore                     # data/, results/checkpoints/, .env, .git-credentials, *.log
├── configs/                       # one YAML per arm
│   ├── toy.yaml
│   ├── grpo_sparse.yaml
│   ├── grpo_shaped.yaml
│   └── eval.yaml
├── src/
│   └── agentic/                   # importable package
│       ├── __init__.py
│       ├── env_alfworld.py        # ALFWorld wrapper: reset/step, admissible-command formatting
│       ├── env_toy.py             # hand-built TextWorld toy task
│       ├── rollout.py             # multi-turn rollout collector + action-token masking
│       ├── reward.py              # terminal reward + optional subgoal shaping
│       ├── grpo.py                # GRPO training over collected trajectories
│       └── eval.py                # success rate on train vs OOD games
├── scripts/                       # thin CLI entry points
│   ├── setup_env.sh               # wandb login + HF/ALFWORLD env exports
│   ├── run_toy.py
│   ├── train_grpo.py              # --reward {sparse,shaped}
│   └── run_eval.py
├── data/                          # gitignored (alfworld data lives at $ALFWORLD_DATA on the volume)
├── results/
│   ├── checkpoints/               # <base>_<task>_<datetime>
│   ├── metrics/
│   └── plots/
└── tests/                         # smoke tests (toy env + parsing/masking unit tests)
```

---

## 7. Deliverables & acceptance criteria

- **Group-degeneracy deliverable:** the **zero-advantage fraction curve** for both reward arms, and a
  stated verdict on whether the sparse arm trained at all. A near-100% zero-advantage sparse arm is a
  publishable negative — "terminal-only reward does not train a 1.5B agent because groups degenerate"
  — and is a different claim from "RL does not help".
- **Deliverables:** (1) a working ALFWorld RL environment + multi-turn rollout harness (with a toy
  TextWorld env as the validated reference), (2) success-rate learning curves for GRPO,
  (3) sparse-vs-shaped reward comparison incl. a success-vs-steps efficiency plot, (4) train-vs-OOD
  generalization table with a per-task-type breakdown.
- **Acceptance:** the toy env trains to near-100% (harness proven); GRPO raises ALFWorld success rate
  measurably over the prompted base; a signed sparse-vs-shaped verdict; an OOD generalization number
  with error bars.

Optional extensions (native): a difficulty curriculum (easy→hard task types); ReAct-style
`think→act` formatting vs action-only; a second small policy for a size-transfer check.

---

## 8. Risks & mitigations

- **Reward too sparse → no gradient:** if the base agent almost never succeeds, GRPO gets no signal.
  Mitigate with subgoal shaping (Arm B), a difficulty curriculum starting on easy task types, and a
  prompted base strong enough to occasionally succeed.
- **Invalid / unparseable actions:** constrain to the env's **admissible commands** (show them in the
  prompt; reject/penalize off-list actions); audit the parser on 20 episodes early.
- **Context growth over long episodes:** truncate/summarize history; cap `max_steps`.
- **Masking bug:** train only on agent action tokens, not environment observations — verify on the toy
  env before ALFWorld (this is what E0 is for).
- **Reward hacking / degenerate loops:** penalize step-limit timeouts and repeated no-op actions;
  track invalid-action and repeat rates.
- **Dependency fragility:** verifiers ↔ TRL ↔ vLLM ↔ alfworld versions are sensitive — pin them and
  validate on the toy env first.
- **ALFWorld data:** `alfworld-download` writes to `$ALFWORLD_DATA` on the volume once; confirm it
  persists so pods don't re-download.
