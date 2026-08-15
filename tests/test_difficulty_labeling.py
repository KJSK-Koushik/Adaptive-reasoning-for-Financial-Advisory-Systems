from __future__ import annotations

import pytest

from adaptive_reasoning.config import load_config
from adaptive_reasoning.difficulty.labeling import (
    SampleOutcome,
    distribution,
    label_all,
    label_question,
)
from adaptive_reasoning.schema import Difficulty


def outcomes(correct: list[bool], tokens: int = 50, answers: list[str] | None = None):
    answers = answers or ["42"] * len(correct)
    return [
        SampleOutcome(question_id="q1", answer=a, correct=c, reasoning_tokens=tokens)
        for c, a in zip(correct, answers, strict=True)
    ]


@pytest.fixture
def cfg():
    return load_config()


def test_all_correct_and_short_is_easy(cfg):
    v = label_question(outcomes([True] * 5, tokens=40), cfg)
    assert v.difficulty == Difficulty.EASY
    assert v.pass_rate == 1.0


def test_never_correct_is_hard(cfg):
    v = label_question(outcomes([False] * 5), cfg)
    assert v.difficulty == Difficulty.HARD
    assert v.pass_rate == 0.0


def test_mixed_is_medium(cfg):
    v = label_question(outcomes([True, True, True, False, False], tokens=40), cfg)
    assert v.difficulty == Difficulty.MEDIUM


def test_solved_but_only_after_long_reasoning_is_bumped(cfg):
    """5/5 correct but 600 tokens of deliberation is not an easy question."""
    short = label_question(outcomes([True] * 5, tokens=40), cfg)
    long = label_question(outcomes([True] * 5, tokens=600), cfg)
    assert short.difficulty == Difficulty.EASY
    assert long.difficulty == Difficulty.MEDIUM
    assert "bumped" in long.reason


def test_bump_saturates_at_hard(cfg):
    v = label_question(outcomes([False] * 5, tokens=900), cfg)
    assert v.difficulty == Difficulty.HARD


def test_fully_unstable_answers_demote_to_hard(cfg):
    """Every sample disagreeing means guessing, not reasoning."""
    v = label_question(
        outcomes([True, True, False, False, False], tokens=40,
                 answers=["1", "2", "3", "4", "5"]),
        cfg,
    )
    assert v.difficulty == Difficulty.HARD
    assert "unstable" in v.reason


def test_stable_answers_stay_medium(cfg):
    v = label_question(
        outcomes([True, True, False, False, False], tokens=40,
                 answers=["42", "42", "42", "7", "7"]),
        cfg,
    )
    assert v.difficulty == Difficulty.MEDIUM


def test_reasoning_length_measured_on_correct_attempts_only(cfg):
    """A wrong answer that rambled says nothing about how long the question needs."""
    mixed = [
        SampleOutcome("q1", "42", True, 30),
        SampleOutcome("q1", "42", True, 30),
        SampleOutcome("q1", "42", True, 30),
        SampleOutcome("q1", "42", True, 30),
        SampleOutcome("q1", "99", False, 5000),
    ]
    v = label_question(mixed, cfg)
    assert v.median_reasoning_tokens == 30
    assert v.difficulty == Difficulty.EASY


def test_falls_back_to_all_attempts_when_none_correct(cfg):
    v = label_question(outcomes([False] * 3, tokens=77), cfg)
    assert v.median_reasoning_tokens == 77


def test_score_is_the_pass_rate(cfg):
    v = label_question(outcomes([True, True, False, False]), cfg)
    assert v.score == 0.5


def test_empty_outcomes_raise(cfg):
    with pytest.raises(ValueError, match="no sample outcomes"):
        label_question([], cfg)


def test_label_all_groups_by_question(cfg):
    mixed = [
        SampleOutcome("a", "1", True, 10),
        SampleOutcome("b", "2", False, 10),
        SampleOutcome("a", "1", True, 10),
    ]
    verdicts = label_all(mixed, cfg)
    assert {v.question_id for v in verdicts} == {"a", "b"}
    assert {v.question_id: v.n_samples for v in verdicts} == {"a": 2, "b": 1}


def test_distribution_counts_tiers(cfg):
    verdicts = label_all(
        [SampleOutcome("a", "1", True, 10), SampleOutcome("b", "2", False, 10)], cfg
    )
    assert distribution(verdicts) == {"easy": 1, "hard": 1}


def test_thresholds_are_config_driven():
    """Tightening the easy threshold must move labels, or the config is decorative."""
    strict = load_config(overrides={"difficulty": {"easy_min_pass_rate": 1.0}})
    lenient = load_config(overrides={"difficulty": {"easy_min_pass_rate": 0.5}})
    data = outcomes([True, True, True, False], tokens=20)
    assert label_question(data, strict).difficulty == Difficulty.MEDIUM
    assert label_question(data, lenient).difficulty == Difficulty.EASY
