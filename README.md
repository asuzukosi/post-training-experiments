# post-training experiments

experimental work on generative ai apps, agents, rag, and post-training.

## setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
cp .env.example .env
```

or:

```bash
pip install -r requirements.txt
```

optional groups:

```bash
pip install -r requirements/rag.txt
pip install -r requirements/search.txt
# see requirements/ for: prompting, langchain-extra, fine-tuning, niche, all
```

## layout

organized by purpose, not by framework or course source.

| folder | purpose |
|---|---|
| `apps/` | runnable demos and cli tools |
| `rag/` | retrieval, embeddings, vector stores |
| `agents/` | tool use, single/multi-agent, workflows, domain crews |
| `prompting/` | prompt design and optimization |
| `fine-tuning/` | supervised ft, rlhf, agent finetuning |
| `serving/` | inference and deployment |
| `generation/` | non-rag generative work (e.g. diffusion) |
| `llmops/` | data prep, pipelines, automation |
| `archive/` | frozen / unrelated material |

## conventions

- kebab-case folder names
- one experiment = one folder, with a clear entrypoint (`main.py` or a notebook)
- optional `README.md` and `.env.example` per experiment when needed
- new work goes into the matching purpose bucket; no new top-level dump folders
- remove `.gitkeep` when real files land in a folder
