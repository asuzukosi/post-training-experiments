"""qwen chat template helpers and role-marker checks."""
from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

DEFAULT_TOKENIZER_ID = "Qwen/Qwen2.5-1.5B"
_TRAINING_TEMPLATE_PATH = Path(__file__).with_name("qwen_training_chat_template.jinja")

# qwen chatml markers used for double-render detection
ROLE_MARKERS = {
    "system": "<|im_start|>system",
    "user": "<|im_start|>user",
    "assistant": "<|im_start|>assistant",
}


def load_qwen_training_chat_template() -> str:
    """project-owned qwen chatml with {% generation %} around assistant content."""
    return _TRAINING_TEMPLATE_PATH.read_text(encoding="utf-8")


def has_generation_tags(chat_template: str | None) -> bool:
    text = chat_template or ""
    return "{% generation %}" in text and "{% endgeneration %}" in text


def ensure_pad_token(tokenizer: Any) -> Any:
    """set pad_token to eos_token when the tokenizer has no pad.

    decoder-only bases often ship without a pad id (pretraining is unpadded).
    trainers and batched encode still need one; reusing eos is the usual fix
    for causal lms. attention masks keep real eos from being treated as pad
    in the middle of a sequence.
    """
    if getattr(tokenizer, "pad_token", None) is None:
        eos = getattr(tokenizer, "eos_token", None)
        if eos is None:
            raise ValueError("tokenizer has neither pad_token nor eos_token")
        tokenizer.pad_token = eos
    return tokenizer


def ensure_assistant_generation_template(tokenizer: Any) -> Any:
    """set the training chat template when the stock one lacks generation tags.

    trl 0.19 assistant_only_loss needs {% generation %}…{% endgeneration %} so
    return_assistant_tokens_mask can build a non-empty assistant mask. stock
    qwen2.5 templates do not ship those tags.
    """
    if has_generation_tags(getattr(tokenizer, "chat_template", None)):
        return tokenizer
    tokenizer.chat_template = load_qwen_training_chat_template()
    if not has_generation_tags(tokenizer.chat_template):
        raise RuntimeError("failed to install qwen training chat template")
    print("installed qwen training chat template with generation tags")
    return tokenizer


def count_role_markers(rendered: str, role: str) -> int:
    """count `<|im_start|>{role}` markers in a rendered chat string."""
    marker = ROLE_MARKERS.get(role)
    if marker is None:
        raise ValueError(f"unknown role {role!r}; expected one of {sorted(ROLE_MARKERS)}")
    return rendered.count(marker)


def expected_role_counts(messages: Sequence[dict[str, Any]]) -> dict[str, int]:
    """expected marker counts from `messages`.

    qwen's template injects a default system turn when none is present, so system
    expected count is max(1, n_system_in_messages).
    """
    counts = Counter(m.get("role") for m in messages)
    return {
        "system": max(1, int(counts.get("system", 0))),
        "user": int(counts.get("user", 0)),
        "assistant": int(counts.get("assistant", 0)),
    }


def assert_single_role_markers(
    messages: Sequence[dict[str, Any]],
    rendered: str,
) -> None:
    """raise if rendered markers do not match messages (double-template / role bugs)."""
    expected = expected_role_counts(messages)
    actual = {role: count_role_markers(rendered, role) for role in ROLE_MARKERS}
    bad = {
        role: (expected[role], actual[role])
        for role in ROLE_MARKERS
        if expected[role] != actual[role]
    }
    if bad:
        detail = ", ".join(
            f"{role}: expected {exp} got {got}" for role, (exp, got) in sorted(bad.items())
        )
        raise ValueError(f"chat template role marker mismatch ({detail})")


def render_chat(
    messages: Sequence[dict[str, Any]],
    tokenizer: Any,
    *,
    add_generation_prompt: bool = False,
    check: bool = True,
) -> str:
    """render `messages` with `tokenizer.apply_chat_template`; optionally check markers.

    when `add_generation_prompt=True`, an extra trailing assistant marker is expected
    beyond the assistant turns already in `messages`.
    """
    if not messages:
        raise ValueError("messages must be non-empty")
    rendered = tokenizer.apply_chat_template(
        list(messages),
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )
    if not isinstance(rendered, str):
        raise TypeError(
            f"apply_chat_template returned {type(rendered).__name__}, expected str"
        )

    if check:
        if add_generation_prompt:
            # one extra <|im_start|>assistant for the open generation turn
            patched = [dict(m) for m in messages]
            patched.append({"role": "assistant", "content": ""})
            assert_single_role_markers(patched, rendered)
        else:
            assert_single_role_markers(messages, rendered)
    return rendered


def preview_chat_renders(
    rows: Sequence[dict[str, Any]],
    tokenizer: Any,
    *,
    messages_key: str = "messages",
    n: int = 5,
    add_generation_prompt: bool = False,
) -> list[str]:
    """render up to `n` rows and print marker counts (eyeball helper from the spec)."""
    out: list[str] = []
    for i, row in enumerate(rows[:n]):
        messages = row[messages_key]
        rendered = render_chat(
            messages,
            tokenizer,
            add_generation_prompt=add_generation_prompt,
            check=True,
        )
        print("=" * 80)
        print(f"sample {i + 1}/{min(n, len(rows))} id={row.get('id')}")
        for m in messages:
            content = m.get("content") or ""
            preview = content if len(content) <= 200 else content[:200] + "..."
            print(f"[{m.get('role')}] {preview}")
        print("--- apply_chat_template ---")
        print(rendered if len(rendered) <= 800 else rendered[:800] + "...")
        for role in ("user", "assistant", "system"):
            print(f"marker_count {ROLE_MARKERS[role]}={count_role_markers(rendered, role)}")
        out.append(rendered)
    return out


def rendered_token_length(
    messages: Sequence[dict[str, Any]],
    tokenizer: Any,
) -> int:
    """token count of chat-templated messages (no double-role check)."""
    rendered = render_chat(messages, tokenizer, check=False)
    return len(tokenizer(rendered, add_special_tokens=False)["input_ids"])
