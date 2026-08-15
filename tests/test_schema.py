from __future__ import annotations

import pytest
from pydantic import ValidationError

from adaptive_reasoning.schema import AnswerType, Domain, QARecord, Trace, TraceStep


def _step(idx: int, correct: bool, tokens: int) -> TraceStep:
    return TraceStep(
        question_id="q1",
        step_index=idx,
        tokens_so_far=tokens,
        step_text="...",
        probe_answer="12.4" if correct else "99",
        probe_correct=correct,
        confidence=0.8,
        min_token_confidence=0.6,
        entropy=0.3,
        answer_changed=False,
    )


def test_qarecord_prompt_includes_context_and_options():
    rec = QARecord(
        id="p1",
        source="paysim",
        domain=Domain.FRAUD,
        question="Is this transaction fraudulent?",
        context="A TRANSFER of 181.00 emptied the origin account.",
        gold_answer="yes",
        answer_type=AnswerType.CATEGORICAL,
        answer_options=["yes", "no"],
    )
    prompt = rec.prompt()
    assert "TRANSFER" in prompt
    assert "Is this transaction fraudulent?" in prompt
    assert "Options: yes, no" in prompt


def test_qarecord_prompt_without_context():
    rec = QARecord(
        id="s1",
        source="synthetic",
        domain=Domain.INVESTMENT,
        question="What is 5% of 200?",
        gold_answer="10",
        answer_type=AnswerType.NUMERIC,
    )
    assert rec.prompt() == "What is 5% of 200?"


def test_difficulty_is_optional_until_phase_2():
    rec = QARecord(
        id="f1",
        source="finqa",
        domain=Domain.REPORT_QA,
        question="What was the change in revenue?",
        gold_answer="12.4",
        answer_type=AnswerType.NUMERIC,
    )
    assert rec.difficulty is None
    assert rec.difficulty_score is None


def test_unknown_field_rejected():
    with pytest.raises(ValidationError):
        QARecord(
            id="x",
            source="finqa",
            domain=Domain.REPORT_QA,
            question="q",
            gold_answer="1",
            answer_type=AnswerType.NUMERIC,
            typo_field=True,
        )


def test_earliest_correct_step_drives_the_oracle_baseline():
    trace = Trace(
        question_id="q1",
        total_tokens=300,
        final_answer="12.4",
        final_correct=True,
        steps=[_step(0, False, 40), _step(1, True, 95), _step(2, True, 160)],
    )
    assert trace.earliest_correct_step == 1


def test_earliest_correct_step_is_none_when_never_correct():
    trace = Trace(
        question_id="q2",
        total_tokens=300,
        final_answer="99",
        final_correct=False,
        steps=[_step(0, False, 40), _step(1, False, 95)],
    )
    assert trace.earliest_correct_step is None
