"""tests for qwen training chat template (assistant mask)."""
from __future__ import annotations

import pytest

from data_tools.chat import (
    assert_single_role_markers,
    ensure_assistant_generation_template,
    has_generation_tags,
    load_qwen_training_chat_template,
    render_chat,
)


@pytest.fixture
def qwen_tokenizer():
    pytest.importorskip("transformers")
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained(
            "Qwen/Qwen2.5-0.5B",
            trust_remote_code=True,
            local_files_only=True,
        )
    except Exception as exc:  # noqa: BLE001 — skip if cache missing
        pytest.skip(f"qwen tokenizer unavailable: {exc}")


def test_training_template_has_generation_tags() -> None:
    tmpl = load_qwen_training_chat_template()
    assert has_generation_tags(tmpl)
    assert "<|im_start|>assistant" in tmpl


def test_ensure_installs_when_missing(qwen_tokenizer) -> None:
    stock = qwen_tokenizer.chat_template
    assert not has_generation_tags(stock)
    ensure_assistant_generation_template(qwen_tokenizer)
    assert has_generation_tags(qwen_tokenizer.chat_template)
    # idempotent
    again = qwen_tokenizer.chat_template
    ensure_assistant_generation_template(qwen_tokenizer)
    assert qwen_tokenizer.chat_template == again


def test_rendered_text_matches_stock(qwen_tokenizer) -> None:
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello there"},
    ]
    stock_text = qwen_tokenizer.apply_chat_template(messages, tokenize=False)
    ensure_assistant_generation_template(qwen_tokenizer)
    train_text = render_chat(messages, qwen_tokenizer, check=True)
    assert train_text == stock_text
    assert_single_role_markers(messages, train_text)


def test_assistant_mask_nonempty_and_assistant_only(qwen_tokenizer) -> None:
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello there"},
    ]
    ensure_assistant_generation_template(qwen_tokenizer)
    out = qwen_tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        return_assistant_tokens_mask=True,
        return_dict=True,
    )
    mask = list(out["assistant_masks"])
    ids = list(out["input_ids"])
    assert sum(mask) > 0
    assert sum(mask) < len(mask)
    masked = qwen_tokenizer.decode([i for i, m in zip(ids, mask) if m])
    unmasked = qwen_tokenizer.decode([i for i, m in zip(ids, mask) if not m])
    assert "hello there" in masked
    assert "hi" not in masked
    assert "hi" in unmasked
    # default system turn is not assistant content
    assert "helpful assistant" in unmasked
    assert "helpful assistant" not in masked


def test_multi_turn_masks_each_assistant_span(qwen_tokenizer) -> None:
    messages = [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "2+2?"},
        {"role": "assistant", "content": "4"},
        {"role": "user", "content": "times 3?"},
        {"role": "assistant", "content": "12"},
    ]
    ensure_assistant_generation_template(qwen_tokenizer)
    out = qwen_tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        return_assistant_tokens_mask=True,
        return_dict=True,
    )
    mask = list(out["assistant_masks"])
    ids = list(out["input_ids"])
    masked = qwen_tokenizer.decode([i for i, m in zip(ids, mask) if m])
    assert "4" in masked
    assert "12" in masked
    assert "2+2" not in masked
