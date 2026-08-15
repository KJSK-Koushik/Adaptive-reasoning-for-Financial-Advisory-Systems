from __future__ import annotations

import pytest

from adaptive_reasoning.prompting import (
    ANSWER_MARKER,
    build_messages,
    build_user_prompt,
    extract_answer,
    strip_reasoning,
)
from adaptive_reasoning.schema import AnswerType, Domain, QARecord


def _numeric() -> QARecord:
    return QARecord(
        id="n1", source="synthetic", domain=Domain.INVESTMENT,
        question="What is 5% of 200?", gold_answer="10", answer_type=AnswerType.NUMERIC,
    )


def _categorical() -> QARecord:
    return QARecord(
        id="c1", source="paysim", domain=Domain.FRAUD,
        question="Is this fraudulent?", context="A transfer emptied the account.",
        gold_answer="yes", answer_type=AnswerType.CATEGORICAL,
        answer_options=["yes", "no"],
    )


def test_numeric_prompt_asks_for_a_single_number():
    assert "single number" in build_user_prompt(_numeric())


def test_categorical_prompt_lists_the_options():
    prompt = build_user_prompt(_categorical())
    assert "exactly one of: yes, no" in prompt
    assert "A transfer emptied the account." in prompt


def test_messages_have_system_and_user_roles():
    messages = build_messages(_numeric())
    assert [m["role"] for m in messages] == ["system", "user"]
    assert ANSWER_MARKER in messages[0]["content"]


# --------------------------------------------------------------------------- #
# answer extraction - this feeds grading, which feeds the RL reward
# --------------------------------------------------------------------------- #
def test_extracts_from_the_marker():
    assert extract_answer("Some reasoning.\nFinal answer: 42") == "42"


def test_uses_the_last_marker_when_restated():
    text = "Final answer: 10\nWait, let me redo that.\nFinal answer: 12"
    assert extract_answer(text) == "12"


def test_strips_think_blocks():
    text = "<think>the answer might be 7</think>\nFinal answer: 42"
    assert extract_answer(text) == "42"
    assert "7" not in strip_reasoning(text)


def test_handles_unclosed_think_block_from_truncation():
    """A generation cut off mid-reasoning leaves <think> open."""
    text = "<think>I should compute 3 times 4"
    assert "3 times 4" not in strip_reasoning(text)


def test_finds_marker_inside_reasoning_when_nowhere_else():
    text = "<think>working... Final answer: 42</think>"
    assert extract_answer(text) == "42"


def test_falls_back_to_last_line():
    assert extract_answer("Step one.\nStep two.\n3.14") == "3.14"


def test_strips_decoration():
    assert extract_answer("Final answer: **42**") == "42"
    assert extract_answer("Final answer: the answer is 42.") == "42"


def test_empty_input_returns_empty():
    assert extract_answer("") == ""
    assert extract_answer("   ") == ""


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Final answer: yes", "yes"),
        ("Final answer: 1,234.56", "1,234.56"),
        ("Final answer: -3.2%", "-3.2%"),
        ("FINAL ANSWER: bad", "bad"),
    ],
)
def test_answer_shapes(text, expected):
    assert extract_answer(text) == expected


def test_extraction_feeds_grading_correctly():
    """End to end: a realistic completion must grade against the gold answer."""
    from adaptive_reasoning.grading import is_correct

    completion = (
        "<think>The account held 72,665.95 and now holds 0. That is suspicious.</think>\n"
        "The transfer drained the entire balance.\n"
        "Final answer: yes"
    )
    assert is_correct(extract_answer(completion), "yes", "categorical")
