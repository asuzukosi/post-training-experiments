"""react agent finetuning on uber 10q filings (from llama_index react_agent example).

pipeline:
  1) build-index          — vector indexes over march/june/sept 10qs
  2) gen-questions        — synthetic train/eval questions
  3) collect-trajectories — run gpt-4 react agent; log openai finetune events
  4) validate             — check jsonl chat format + token stats
  5) upload / create-job / retrieve — openai fine-tuning api
  6) compare              — query base gpt-3.5 agent vs finetuned model

requires: openai, llama-index, pypdf, tiktoken, numpy, python-dotenv
optional: wandb
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import tiktoken
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = ROOT / "data" / "10q"
DEFAULT_STORAGE = ROOT / "storage"
DEFAULT_TRAIN_Q = ROOT / "train_questions_10q.txt"
DEFAULT_EVAL_Q = ROOT / "eval_questions_10q.txt"
DEFAULT_EVENTS = ROOT / "finetuning_events_10q.jsonl"

PDFS = {
    "march": "uber_10q_march_2022.pdf",
    "june": "uber_10q_june_2022.pdf",
    "sept": "uber_10q_sept_2022.pdf",
}

VARY_QUESTION_TMPL = """\
You are a financial assistant. Given a question over a 2022 Uber 10Q filing, your goal
is to generate up to {num_vary} variations of that question that might span multiple 10Q's.

This can include compare/contrasting different 10Qs, replacing the current quarter with
another quarter, or generating questions that can only be answered over multiple quarters (be creative!)

You are given a valid set of 10Q filings. Please only generate question variations that can be
answered in that set.

Base Question: {base_question}
Valid 10Qs: {valid_10qs}
Question Variations:
"""


def _import_llama():
    """import llama_index pieces with current package layout."""
    try:
        from llama_index.core import (
            SimpleDirectoryReader,
            StorageContext,
            VectorStoreIndex,
            load_index_from_storage,
            Settings,
        )
        from llama_index.core.agent import ReActAgent
        from llama_index.core.evaluation import DatasetGenerator
        from llama_index.core.llms import ChatMessage
        from llama_index.core.prompts import PromptTemplate
        from llama_index.core.tools import QueryEngineTool
        from llama_index.llms.openai import OpenAI as LlamaOpenAI
        from llama_index.core.callbacks import CallbackManager

        try:
            from llama_index.core.callbacks import OpenAIFineTuningHandler
        except ImportError:
            from llama_index.callbacks import OpenAIFineTuningHandler  # type: ignore

        return {
            "SimpleDirectoryReader": SimpleDirectoryReader,
            "StorageContext": StorageContext,
            "VectorStoreIndex": VectorStoreIndex,
            "load_index_from_storage": load_index_from_storage,
            "Settings": Settings,
            "ReActAgent": ReActAgent,
            "DatasetGenerator": DatasetGenerator,
            "PromptTemplate": PromptTemplate,
            "QueryEngineTool": QueryEngineTool,
            "LlamaOpenAI": LlamaOpenAI,
            "CallbackManager": CallbackManager,
            "OpenAIFineTuningHandler": OpenAIFineTuningHandler,
            "ChatMessage": ChatMessage,
        }
    except ImportError as exc:
        raise SystemExit(
            "llama-index is required. install with: pip install -r requirements/rag.txt\n"
            f"import error: {exc}"
        ) from exc


def save_questions(questions: list[str], path: Path) -> None:
    path.write_text("\n".join(q.strip() for q in questions if q.strip()) + "\n")
    print(f"saved {len(questions)} questions -> {path}")


def load_questions(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(path)
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def build_indexes(data_dir: Path, storage_dir: Path, model: str) -> dict:
    li = _import_llama()
    llm = li["LlamaOpenAI"](temperature=0, model=model)
    li["Settings"].llm = llm

    indexes = {}
    for name, filename in PDFS.items():
        persist = storage_dir / name
        pdf_path = data_dir / filename
        try:
            storage_context = li["StorageContext"].from_defaults(persist_dir=str(persist))
            indexes[name] = li["load_index_from_storage"](storage_context)
            print(f"loaded index {name} from {persist}")
            continue
        except Exception:
            pass

        if not pdf_path.exists():
            raise FileNotFoundError(
                f"missing {pdf_path}. place uber 10q pdfs under {data_dir}"
            )
        docs = li["SimpleDirectoryReader"](input_files=[str(pdf_path)]).load_data()
        index = li["VectorStoreIndex"].from_documents(docs)
        persist.mkdir(parents=True, exist_ok=True)
        index.storage_context.persist(persist_dir=str(persist))
        indexes[name] = index
        print(f"built + persisted index {name} -> {persist}")
    return indexes


def make_query_tools(indexes: dict, model: str):
    li = _import_llama()
    llm = li["LlamaOpenAI"](temperature=0, model=model)
    li["Settings"].llm = llm
    tools = []
    descriptions = {
        "march": "Provides information about Uber quarterly financials ending March 2022",
        "june": "Provides information about Uber quarterly financials ending June 2022",
        "sept": "Provides information about Uber quarterly financials ending September 2022",
    }
    for name, index in indexes.items():
        engine = index.as_query_engine(similarity_top_k=3)
        tools.append(
            li["QueryEngineTool"].from_defaults(
                query_engine=engine,
                name=f"{name}_2022",
                description=descriptions[name],
            )
        )
    return tools


def make_agent(tools, model: str, verbose: bool = True, callback_manager=None):
    li = _import_llama()
    llm = li["LlamaOpenAI"](model=model, temperature=0)
    kwargs = {"tools": tools, "llm": llm, "verbose": verbose}
    if callback_manager is not None:
        kwargs["callback_manager"] = callback_manager
    return li["ReActAgent"].from_tools(**kwargs)


def cmd_build_index(args: argparse.Namespace) -> None:
    build_indexes(args.data_dir, args.storage_dir, args.model)
    print("indexes ready")


def cmd_gen_questions(args: argparse.Namespace) -> None:
    li = _import_llama()
    indexes = build_indexes(args.data_dir, args.storage_dir, args.model)
    march_docs = li["SimpleDirectoryReader"](
        input_files=[str(args.data_dir / PDFS["march"])]
    ).load_data()

    llm = li["LlamaOpenAI"](temperature=0, model=args.model)
    li["Settings"].llm = llm
    base_question_gen_query = (
        "You are a Teacher/ Professor. Your task is to setup a quiz/examination."
        " Using the provided context from the Uber March 10Q filing, formulate a"
        " single question that captures an important fact from the context."
        " Restrict the question to the context information provided."
    )
    dataset_generator = li["DatasetGenerator"].from_documents(
        march_docs,
        question_gen_query=base_question_gen_query,
    )
    questions = dataset_generator.generate_questions_from_nodes(num=args.num_base)
    print(f"generated {len(questions)} base questions")

    vary_llm = li["LlamaOpenAI"](model=args.vary_model)
    prompt_tmpl = li["PromptTemplate"](VARY_QUESTION_TMPL)
    valid_10qs = "[March 2022, June 2022, September 2022]"
    new_questions = []
    for idx, question in enumerate(questions):
        new_questions.append(question)
        response = vary_llm.complete(
            prompt_tmpl.format(
                num_vary=args.num_vary,
                base_question=question,
                valid_10qs=valid_10qs,
            )
        )
        lines = [line for line in str(response).splitlines() if line.strip()]
        print(f"[{idx}] {question} -> {lines}")
        new_questions.extend(lines)

    split = args.train_split
    train_questions, eval_questions = new_questions[:split], new_questions[split:]
    save_questions(train_questions, args.train_questions)
    save_questions(eval_questions, args.eval_questions)
    del indexes


def cmd_collect(args: argparse.Namespace) -> None:
    li = _import_llama()
    indexes = build_indexes(args.data_dir, args.storage_dir, args.index_model)
    tools = make_query_tools(indexes, args.index_model)

    finetuning_handler = li["OpenAIFineTuningHandler"]()
    callback_manager = li["CallbackManager"]([finetuning_handler])
    agent = make_agent(
        tools,
        model=args.teacher_model,
        verbose=True,
        callback_manager=callback_manager,
    )

    questions = load_questions(args.train_questions)
    for idx, question in enumerate(questions):
        print(f"[{idx}] question: {question}")
        response = agent.query(question)
        print(f"[{idx}] response: {response}")

    args.events_path.parent.mkdir(parents=True, exist_ok=True)
    finetuning_handler.save_finetuning_events(str(args.events_path))
    print(f"saved trajectories -> {args.events_path}")


def openai_validate_data(dataset_path: Path) -> None:
    with dataset_path.open() as handle:
        dataset = [json.loads(line) for line in handle if line.strip()]

    print("num examples:", len(dataset))
    if dataset:
        print("first example:")
        for message in dataset[0].get("messages", []):
            print(message)

    format_errors: dict[str, int] = defaultdict(int)
    for ex in dataset:
        if not isinstance(ex, dict):
            format_errors["data_type"] += 1
            continue
        messages = ex.get("messages")
        if not messages:
            format_errors["missing_messages_list"] += 1
            continue
        for message in messages:
            if "role" not in message or "content" not in message:
                format_errors["message_missing_key"] += 1
            if any(key not in ("role", "content", "name") for key in message):
                format_errors["message_unrecognized_key"] += 1
            if message.get("role") not in ("system", "user", "assistant", "tool", "function"):
                format_errors["unrecognized_role"] += 1
            content = message.get("content")
            if content is None or not isinstance(content, str):
                format_errors["missing_content"] += 1
        if not any(message.get("role") == "assistant" for message in messages):
            format_errors["example_missing_assistant_message"] += 1

    if format_errors:
        print("found errors:")
        for key, value in format_errors.items():
            print(f"{key}: {value}")
    else:
        print("no errors found")

    encoding = tiktoken.get_encoding("cl100k_base")

    def num_tokens_from_messages(messages, tokens_per_message=3, tokens_per_name=1):
        num_tokens = 0
        for message in messages:
            num_tokens += tokens_per_message
            for key, value in message.items():
                if isinstance(value, str):
                    num_tokens += len(encoding.encode(value))
                if key == "name":
                    num_tokens += tokens_per_name
        return num_tokens + 3

    convo_lens = [num_tokens_from_messages(ex["messages"]) for ex in dataset]
    print("\n#### distribution of num_total_tokens_per_example:")
    print(f"min / max: {min(convo_lens)}, {max(convo_lens)}")
    print(f"mean / median: {np.mean(convo_lens)}, {np.median(convo_lens)}")
    n_too_long = sum(length > 4096 for length in convo_lens)
    print(f"{n_too_long} examples may be over the 4096 token limit")


def cmd_validate(args: argparse.Namespace) -> None:
    openai_validate_data(args.events_path)


def cmd_upload(args: argparse.Namespace) -> None:
    client = OpenAI()
    with args.events_path.open("rb") as handle:
        info = client.files.create(file=handle, purpose="fine-tune")
    print(f"uploaded file_id={info.id}")


def cmd_create_job(args: argparse.Namespace) -> None:
    client = OpenAI()
    job = client.fine_tuning.jobs.create(
        training_file=args.file_id,
        model=args.model,
        hyperparameters={"n_epochs": args.n_epochs},
    )
    print(f"created job_id={job.id} status={job.status}")
    if args.wandb:
        try:
            from wandb.integration.openai.fine_tuning import WandbLogger

            WandbLogger.sync(fine_tune_job_id=job.id, openai_client=client)
            print("synced job to wandb")
        except Exception as exc:
            print(f"wandb sync skipped: {exc}")


def cmd_retrieve(args: argparse.Namespace) -> None:
    client = OpenAI()
    state = client.fine_tuning.jobs.retrieve(args.job_id)
    print(json.dumps(state.model_dump(), indent=2, default=str))


def cmd_compare(args: argparse.Namespace) -> None:
    indexes = build_indexes(args.data_dir, args.storage_dir, args.index_model)
    tools = make_query_tools(indexes, args.index_model)
    base_agent = make_agent(tools, model=args.base_model, verbose=True)
    ft_agent = make_agent(tools, model=args.ft_model, verbose=True)

    if args.question:
        questions = [args.question]
    else:
        questions = load_questions(args.eval_questions)
        questions = [questions[args.qidx]]

    for question in questions:
        print("question:", question)
        print("\n--- base ---")
        print(base_agent.query(question))
        print("\n--- finetuned ---")
        print(ft_agent.query(question))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="react agent finetune pipeline (uber 10q)")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--storage-dir", type=Path, default=DEFAULT_STORAGE)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("build-index", help="build or load vector indexes")
    p.add_argument("--model", default="gpt-3.5-turbo")

    p = sub.add_parser("gen-questions", help="generate train/eval questions")
    p.add_argument("--model", default="gpt-3.5-turbo")
    p.add_argument("--vary-model", default="gpt-4o")
    p.add_argument("--num-base", type=int, default=20)
    p.add_argument("--num-vary", type=int, default=3)
    p.add_argument("--train-split", type=int, default=60)
    p.add_argument("--train-questions", type=Path, default=DEFAULT_TRAIN_Q)
    p.add_argument("--eval-questions", type=Path, default=DEFAULT_EVAL_Q)

    p = sub.add_parser("collect-trajectories", help="log gpt-4 react trajectories for finetune")
    p.add_argument("--index-model", default="gpt-3.5-turbo")
    p.add_argument("--teacher-model", default="gpt-4o")
    p.add_argument("--train-questions", type=Path, default=DEFAULT_TRAIN_Q)
    p.add_argument("--events-path", type=Path, default=DEFAULT_EVENTS)

    p = sub.add_parser("validate", help="validate finetune jsonl")
    p.add_argument("--events-path", type=Path, default=DEFAULT_EVENTS)

    p = sub.add_parser("upload", help="upload jsonl to openai")
    p.add_argument("--events-path", type=Path, default=DEFAULT_EVENTS)

    p = sub.add_parser("create-job", help="create openai fine-tuning job")
    p.add_argument("--file-id", required=True)
    p.add_argument("--model", default="gpt-3.5-turbo")
    p.add_argument("--n-epochs", type=int, default=3)
    p.add_argument("--wandb", action="store_true")

    p = sub.add_parser("retrieve", help="retrieve fine-tuning job")
    p.add_argument("--job-id", required=True)

    p = sub.add_parser("compare", help="compare base vs finetuned react agents")
    p.add_argument("--index-model", default="gpt-3.5-turbo")
    p.add_argument("--base-model", default="gpt-3.5-turbo")
    p.add_argument("--ft-model", required=True, help="fine-tuned model id")
    p.add_argument("--eval-questions", type=Path, default=DEFAULT_EVAL_Q)
    p.add_argument("--qidx", type=int, default=0)
    p.add_argument("--question", default=None, help="optional one-off question")

    return parser


def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is required (set in env or .env)")

    parser = build_parser()
    args = parser.parse_args()
    commands = {
        "build-index": cmd_build_index,
        "gen-questions": cmd_gen_questions,
        "collect-trajectories": cmd_collect,
        "validate": cmd_validate,
        "upload": cmd_upload,
        "create-job": cmd_create_job,
        "retrieve": cmd_retrieve,
        "compare": cmd_compare,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
