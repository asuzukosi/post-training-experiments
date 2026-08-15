"""shared judgearena stubs for eval tests (no gpu)."""
from __future__ import annotations

import sys
import types

import pytest


@pytest.fixture(autouse=True)
def reset_live_engine():
    """clear the process-wide engine between tests.

    it is module state by design (only one engine may hold the gpu), so without this
    one test's cached engine satisfies the next test's lookup and hides a reload.
    """
    from eval import vllm_backend

    vllm_backend._LIVE = None
    yield
    vllm_backend._LIVE = None


def stub_judgearena(
    monkeypatch: pytest.MonkeyPatch,
    prefs_by_prompt: dict[str, tuple[float, float]],
) -> list[str]:
    """monkeypatch make_model, annotate_battles, and PairScore.

    prefs_by_prompt maps instruction text -> (pref_ab, pref_ba) in the original
    frame. the ba annotate pass still returns displayed P(B wins); the wrapper
    remaps with 1 - pref.
    """
    judged: list[str] = []
    prev: tuple[list[str], list[str], list[str]] | None = None

    class _ann:
        def __init__(self, text: str) -> None:
            self.judge_completion = text

    # a bare object() swallowed every kwarg and had no sampling_params, which is
    # exactly why the judge silently sampling at judgearena's hardcoded 0.6 was
    # invisible to these tests. mirror the real ChatVLLM surface instead.
    class _FakeSamplingParams:
        def __init__(self) -> None:
            self.max_tokens = 8192
            self.temperature = 0.6  # judgearena 0.1.0's hardcoded default
            self.top_p = 0.95

    class _FakeChatVLLM:
        def __init__(self) -> None:
            self.sampling_params = _FakeSamplingParams()

    def fake_make_model(*_args, **kwargs):
        # the real make_model forwards unknown kwargs to vllm's LLM(), which rejects
        # `temperature` outright — so passing it here must fail loudly, not pass.
        if "temperature" in kwargs:
            raise TypeError(
                "EngineArgs.__init__() got an unexpected keyword argument 'temperature'"
            )
        return _FakeChatVLLM()

    def fake_annotate_battles(
        *_args,
        instructions: list[str],
        completions_A: list[str],
        completions_B: list[str],
        **_kwargs,
    ):
        nonlocal prev
        inst = list(instructions)
        ca = list(completions_A)
        cb = list(completions_B)
        swapped = (
            prev is not None
            and prev[0] == inst
            and prev[1] == cb
            and prev[2] == ca
        )
        if not swapped:
            prev = (inst, ca, cb)
            judged.extend(inst)
        out = []
        for prompt in inst:
            pref_ab, pref_ba = prefs_by_prompt[prompt]
            displayed = pref_ab if not swapped else 1.0 - pref_ba
            out.append(_ann(str(displayed)))
        return out

    class fake_pair_score:
        def parse_model_raw(self, judge_completion: str) -> float:
            return float(judge_completion)

    try:
        import judgearena.evaluate as ja_eval
        import judgearena.utils as ja_utils

        monkeypatch.setattr(ja_eval, "annotate_battles", fake_annotate_battles)
        monkeypatch.setattr(ja_eval, "PairScore", fake_pair_score)
        monkeypatch.setattr(ja_utils, "make_model", fake_make_model)
    except ModuleNotFoundError:
        pkg = types.ModuleType("judgearena")
        evaluate = types.ModuleType("judgearena.evaluate")
        utils = types.ModuleType("judgearena.utils")
        evaluate.annotate_battles = fake_annotate_battles
        evaluate.PairScore = fake_pair_score
        utils.make_model = fake_make_model
        pkg.evaluate = evaluate
        pkg.utils = utils
        monkeypatch.setitem(sys.modules, "judgearena", pkg)
        monkeypatch.setitem(sys.modules, "judgearena.evaluate", evaluate)
        monkeypatch.setitem(sys.modules, "judgearena.utils", utils)
    return judged


@pytest.fixture
def install_judgearena_stub(monkeypatch: pytest.MonkeyPatch):
    def _install(prefs_by_prompt: dict[str, tuple[float, float]]) -> list[str]:
        return stub_judgearena(monkeypatch, prefs_by_prompt)

    return _install


def stub_vllm_generate(
    monkeypatch: pytest.MonkeyPatch,
    fn=None,
) -> list[list[str]]:
    """monkeypatch eval.generate.vllm_generate so tests never load vllm."""
    calls: list[list[str]] = []

    def fake_gen(prompts, *, model, max_tokens, temperature, top_p):
        batch = list(prompts)
        calls.append(batch)
        if fn is not None:
            return fn(batch)
        return [f"out:{p}" for p in batch]

    monkeypatch.setattr("eval.generate.vllm_generate", fake_gen)
    return calls


@pytest.fixture
def install_vllm_stub(monkeypatch: pytest.MonkeyPatch):
    def _install(fn=None) -> list[list[str]]:
        return stub_vllm_generate(monkeypatch, fn)

    return _install


def stub_chat_template(monkeypatch: pytest.MonkeyPatch) -> None:
    """skip tokenizer load; pass prompts through unchanged."""

    def identity(items, *, tokenizer_source):
        return [{"id": str(item["id"]), "prompt": str(item["prompt"])} for item in items]

    monkeypatch.setattr("eval.head_to_head.apply_chat_template_to_prompts", identity)


@pytest.fixture
def install_chat_template_stub(monkeypatch: pytest.MonkeyPatch):
    def _install() -> None:
        stub_chat_template(monkeypatch)

    return _install
