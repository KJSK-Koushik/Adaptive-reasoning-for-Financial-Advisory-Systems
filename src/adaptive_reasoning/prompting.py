"""Prompt construction and answer extraction.

Shared by Phase 2 (difficulty sampling) and Phase 3 (trace generation) so that a
question is presented to the model *identically* in both. If the two diverged, the
difficulty labels would describe a prompt the stopping policy never actually sees.

The contract with the model is:

1. Reason step by step in prose.
2. End with ``Final answer: <value>`` on its own line.

Phase 3 additionally interrupts mid-reasoning and appends :data:`PROBE_PROMPT` to ask
what the answer would be if it stopped there - which is why the answer marker has to be
a short, fixed string that is cheap to generate and unambiguous to parse.
"""

from __future__ import annotations

import re

from .schema import AnswerType, QARecord

ANSWER_MARKER = "Final answer:"

#: Appended to a partial reasoning trace to force an early answer. Kept in sync with
#: ``traces.probe_prompt`` in the config.
PROBE_PROMPT = f"\n\n{ANSWER_MARKER}"

SYSTEM_PROMPT = (
    "You are a careful financial analyst. Work through the problem step by step, "
    "then state your conclusion on a new line in exactly this form:\n"
    f"{ANSWER_MARKER} <answer>\n"
    "Give only the value after the marker - a single number, or a single word for a "
    "classification. Do not add units, currency symbols, or explanation after it."
)

_NUMERIC_HINT = "Answer with a single number."
_CATEGORICAL_HINT = "Answer with exactly one of: {options}."


def build_user_prompt(record: QARecord) -> str:
    """Render the question exactly as the model will see it."""
    parts: list[str] = []
    if record.context:
        parts.append(record.context.strip())
    parts.append(record.question.strip())

    if record.answer_type == AnswerType.CATEGORICAL and record.answer_options:
        parts.append(_CATEGORICAL_HINT.format(options=", ".join(record.answer_options)))
    elif record.answer_type == AnswerType.NUMERIC:
        parts.append(_NUMERIC_HINT)

    return "\n\n".join(parts)


def build_messages(record: QARecord) -> list[dict[str, str]]:
    """Chat-format messages for an instruction-tuned model."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(record)},
    ]


# Reasoning models wrap their chain of thought in <think> tags; anything inside is
# working, not the answer.
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_OPEN_THINK = re.compile(r"<think>.*", re.DOTALL | re.IGNORECASE)
_MARKER = re.compile(rf"{re.escape(ANSWER_MARKER)}\s*(.+?)(?:\n|$)", re.IGNORECASE)


def strip_reasoning(text: str) -> str:
    """Remove ``<think>`` blocks, including one left unclosed by truncation."""
    text = _THINK_BLOCK.sub(" ", text)
    return _OPEN_THINK.sub(" ", text)


def extract_answer(text: str) -> str:
    """Pull the final answer out of a completion.

    Falls back progressively, because a truncated or non-compliant generation still
    has to yield *something* - returning nothing would be scored as wrong and would
    quietly bias the difficulty labels toward "hard".
    """
    if not text:
        return ""

    visible = strip_reasoning(text)

    # 1. The marker, last occurrence (the model sometimes restates it).
    matches = _MARKER.findall(visible)
    if matches:
        return _clean_answer(matches[-1])

    # 2. The marker inside a reasoning block, if the visible part had none.
    matches = _MARKER.findall(text)
    if matches:
        return _clean_answer(matches[-1])

    # 3. Last non-empty line of the visible text.
    lines = [ln.strip() for ln in visible.splitlines() if ln.strip()]
    if lines:
        return _clean_answer(lines[-1])

    return ""


def _clean_answer(text: str) -> str:
    """Trim decoration the model adds around the value.

    Tag removal matters: when the marker is found *inside* a reasoning block the raw
    match carries the closing tag with it, e.g. ``42</think>``.
    """
    text = re.sub(r"</?\w+>", "", text)
    text = text.strip().strip("*_`").strip()
    text = re.sub(r"^(is|the answer is|answer:)\s*", "", text, flags=re.IGNORECASE)
    return text.rstrip(".").strip()


def count_reasoning_tokens(text: str, tokenizer) -> int:
    """Number of tokens spent reasoning, i.e. before the answer marker."""
    idx = text.lower().rfind(ANSWER_MARKER.lower())
    reasoning = text if idx < 0 else text[:idx]
    return len(tokenizer.encode(reasoning, add_special_tokens=False))
