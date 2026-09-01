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

#: Separators used when composing prompts, named so the few-shot examples below
#: read cleanly instead of being peppered with escape sequences.
LINE = chr(10)
BREAK = chr(10) * 2

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

#: Two short worked examples, shown before the real question when
#: ``prompting.few_shot`` is on. They are invented rather than drawn from any split,
#: so no gold answer can leak into a question the model is later evaluated on.
#:
#: Their job is format compliance, not teaching finance. Measured on the 768-token
#: traces, 23% of wrong numeric answers were a sentence or an unfinished calculation
#: rather than a value - the model reasoning correctly and then failing to land on the
#: contract. One numeric and one categorical example cover both answer types.
FEW_SHOT: list[tuple[str, str]] = [
    (
        "Revenue was 240 in 2019 and 300 in 2020. What was the percentage increase?"
        + BREAK + "Answer with a single number.",
        "The increase is 300 - 240 = 60. As a percentage of 2019 revenue, "
        "60 / 240 = 0.25, which is 25 percent." + LINE + ANSWER_MARKER + " 25",
    ),
    (
        "The applicant has no savings account, six years of employment and no prior "
        "defaults. Classify the credit risk."
        + BREAK + "Answer with exactly one of: good, bad.",
        "Stable employment and a clean repayment history outweigh the absent "
        "savings account, so the profile is acceptable." + LINE + ANSWER_MARKER + " good",
    ),
]


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


def build_messages(record: QARecord, few_shot: bool = False) -> list[dict[str, str]]:
    """Chat-format messages for an instruction-tuned model.

    With ``few_shot``, two invented worked examples are prepended as prior turns. The
    system prompt already states the answer contract; the examples demonstrate it,
    which a zero-shot instruction alone was not reliably achieving.
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if few_shot:
        for question, answer in FEW_SHOT:
            messages.append({"role": "user", "content": question})
            messages.append({"role": "assistant", "content": answer})
    messages.append({"role": "user", "content": build_user_prompt(record)})
    return messages


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
