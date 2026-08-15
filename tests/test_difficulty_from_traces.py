"""Tests for deriving difficulty from traces.

This replaced the k-sampling route because that job measured at ~71% of the GPU budget
on a Kaggle T4. The rules must still separate the tiers sensibly, so the cases below
are written as recognisable trace *shapes* rather than abstract numbers.
"""

from __future__ import annotations

import pandas as pd
import pytest

from adaptive_reasoning.config import load_config
from adaptive_reasoning.difficulty.from_traces import (
    TraceEvidence,
    evidence_from_frames,
    label_all_from_traces,
    label_from_trace,
    observed_budget,
)
from adaptive_reasoning.schema import Difficulty


@pytest.fixture
def cfg():
    return load_config()


def _evidence(**overrides) -> TraceEvidence:
    base = {
        "question_id": "q1", "n_steps": 16, "n_correct_steps": 8, "total_tokens": 300,
        "final_correct": True, "earliest_correct_step": 8, "n_answer_changes": 2,
    }
    base.update(overrides)
    return TraceEvidence(**base)


# --------------------------------------------------------------------------- #
# the three signals
# --------------------------------------------------------------------------- #
def test_correct_early_and_held_is_easy(cfg):
    """Right from step 2 of 16 onwards - the policy should stop almost immediately."""
    verdict = label_from_trace(
        _evidence(n_correct_steps=14, earliest_correct_step=2, total_tokens=200), cfg
    )
    assert verdict.difficulty == Difficulty.EASY


def test_correct_only_at_the_very_end_is_hard(cfg):
    """One correct probe out of 16 - stopping early would have been wrong."""
    verdict = label_from_trace(
        _evidence(n_correct_steps=1, earliest_correct_step=15, total_tokens=200), cfg
    )
    assert verdict.difficulty == Difficulty.HARD


def test_never_correct_is_hard(cfg):
    verdict = label_from_trace(
        _evidence(n_correct_steps=0, earliest_correct_step=None, final_correct=False), cfg
    )
    assert verdict.difficulty == Difficulty.HARD
    assert "never correct" in verdict.reason


def test_middling_is_medium(cfg):
    """5 of 16 probes correct - between the easy and hard thresholds."""
    verdict = label_from_trace(
        _evidence(n_correct_steps=5, earliest_correct_step=10, total_tokens=200), cfg
    )
    assert verdict.difficulty == Difficulty.MEDIUM


def test_long_reasoning_bumps_a_tier(cfg):
    """Solved, but only after burning nearly the whole budget - not genuinely easy.

    The threshold sits at the top of the observed distribution (0.90 of budget), so
    this rule is an exception, not the norm. At its original 0.60 it fired on 36% of
    real questions and helped collapse the medium tier.
    """
    short = label_from_trace(
        _evidence(n_correct_steps=14, earliest_correct_step=2, total_tokens=200), cfg
    )
    long = label_from_trace(
        _evidence(n_correct_steps=14, earliest_correct_step=2, total_tokens=1000), cfg
    )
    assert short.difficulty == Difficulty.EASY
    assert long.difficulty == Difficulty.MEDIUM
    assert "token budget" in long.reason


def test_unstable_answers_demote_medium_to_hard(cfg):
    """Only *extreme* instability counts.

    Measured over 4,000 traces, change_ratio has a median of 0.67 - the tentative
    answer moves at two-thirds of steps as a matter of course. The threshold is at
    0.90 so this catches genuine outliers rather than normal reasoning.
    """
    stable = label_from_trace(
        _evidence(n_correct_steps=5, total_tokens=200, n_answer_changes=1), cfg
    )
    typical = label_from_trace(
        _evidence(n_correct_steps=5, total_tokens=200, n_answer_changes=11), cfg
    )
    unstable = label_from_trace(
        _evidence(n_correct_steps=5, total_tokens=200, n_answer_changes=15), cfg
    )
    assert stable.difficulty == Difficulty.MEDIUM
    assert typical.difficulty == Difficulty.MEDIUM, "0.69 change ratio is normal, not unstable"
    assert unstable.difficulty == Difficulty.HARD
    assert "instability" in unstable.reason


def test_budget_comes_from_the_data_not_the_config(cfg):
    """Traces are often generated with an overridden budget on the remote machine.

    The first real run used 768 tokens while the local config said 1024; scaling the
    length rule by the config value silently mis-fires.
    """
    evidence = _evidence(n_correct_steps=14, earliest_correct_step=2, total_tokens=700)
    assert label_from_trace(evidence, cfg, budget=1024).difficulty == Difficulty.EASY
    assert label_from_trace(evidence, cfg, budget=768).difficulty == Difficulty.MEDIUM


def test_bump_saturates_at_hard(cfg):
    verdict = label_from_trace(
        _evidence(n_correct_steps=0, earliest_correct_step=None, total_tokens=760), cfg
    )
    assert verdict.difficulty == Difficulty.HARD


# --------------------------------------------------------------------------- #
# evidence arithmetic
# --------------------------------------------------------------------------- #
def test_ratios():
    evidence = _evidence(n_steps=10, n_correct_steps=3, n_answer_changes=5)
    assert evidence.correct_ratio == 0.3
    assert evidence.change_ratio == 0.5


def test_ratios_handle_zero_steps():
    evidence = _evidence(n_steps=0, n_correct_steps=0, n_answer_changes=0)
    assert evidence.correct_ratio == 0.0
    assert evidence.change_ratio == 0.0


def test_verdict_carries_the_evidence(cfg):
    verdict = label_from_trace(_evidence(n_steps=16, n_correct_steps=8), cfg)
    assert verdict.pass_rate == 0.5
    assert verdict.n_samples == 16
    assert verdict.median_reasoning_tokens == 300.0


def test_thresholds_are_config_driven():
    strict = load_config(overrides={"difficulty": {"from_traces": {"easy_min_correct_ratio": 0.95}}})
    lenient = load_config(overrides={"difficulty": {"from_traces": {"easy_min_correct_ratio": 0.5}}})
    evidence = _evidence(n_correct_steps=12, earliest_correct_step=4, total_tokens=200)
    assert label_from_trace(evidence, strict).difficulty == Difficulty.MEDIUM
    assert label_from_trace(evidence, lenient).difficulty == Difficulty.EASY


# --------------------------------------------------------------------------- #
# frame aggregation
# --------------------------------------------------------------------------- #
def _frames():
    steps = pd.DataFrame([
        {"question_id": "a", "step_index": 0, "probe_correct": False, "answer_changed": False},
        {"question_id": "a", "step_index": 1, "probe_correct": True, "answer_changed": True},
        {"question_id": "a", "step_index": 2, "probe_correct": True, "answer_changed": False},
        {"question_id": "b", "step_index": 0, "probe_correct": False, "answer_changed": False},
        {"question_id": "b", "step_index": 1, "probe_correct": False, "answer_changed": True},
    ])
    summary = pd.DataFrame([
        {"question_id": "a", "total_tokens": 120, "final_correct": True,
         "earliest_correct_step": 1.0},
        {"question_id": "b", "total_tokens": 300, "final_correct": False,
         "earliest_correct_step": None},
    ])
    return steps, summary


def test_evidence_from_frames_aggregates_per_question():
    evidence = {e.question_id: e for e in evidence_from_frames(*_frames())}
    assert evidence["a"].n_steps == 3
    assert evidence["a"].n_correct_steps == 2
    assert evidence["a"].n_answer_changes == 1
    assert evidence["a"].earliest_correct_step == 1


def test_evidence_from_frames_handles_never_correct():
    evidence = {e.question_id: e for e in evidence_from_frames(*_frames())}
    assert evidence["b"].earliest_correct_step is None
    assert evidence["b"].n_correct_steps == 0


def test_evidence_skips_questions_without_a_summary_row():
    steps, summary = _frames()
    evidence = evidence_from_frames(steps, summary[summary.question_id == "a"])
    assert {e.question_id for e in evidence} == {"a"}


def test_observed_budget_reads_the_longest_trace(cfg):
    _, summary = _frames()
    assert observed_budget(summary, cfg) == 300


def test_observed_budget_falls_back_when_the_column_is_missing(cfg):
    assert observed_budget(pd.DataFrame({"other": [1]}), cfg) == cfg.llm.max_new_tokens


def test_observed_budget_falls_back_on_nonsense(cfg):
    frame = pd.DataFrame({"total_tokens": [0, 0]})
    assert observed_budget(frame, cfg) == cfg.llm.max_new_tokens


def test_label_all_from_traces_end_to_end(cfg):
    verdicts = label_all_from_traces(*_frames(), cfg)
    assert len(verdicts) == 2
    by_id = {v.question_id: v for v in verdicts}
    assert by_id["b"].difficulty == Difficulty.HARD      # never correct
    assert by_id["a"].difficulty in set(Difficulty)


def test_labels_are_not_degenerate_on_a_realistic_mix(cfg):
    """A spread of trace shapes must produce all three tiers, or the thresholds are wrong."""
    shapes = [
        dict(n_correct_steps=15, earliest_correct_step=1, total_tokens=150),   # easy
        dict(n_correct_steps=14, earliest_correct_step=2, total_tokens=180),   # easy
        dict(n_correct_steps=5, earliest_correct_step=10, total_tokens=250),   # medium
        dict(n_correct_steps=6, earliest_correct_step=10, total_tokens=300),   # medium
        dict(n_correct_steps=1, earliest_correct_step=15, total_tokens=400),   # hard
        dict(n_correct_steps=0, earliest_correct_step=None, total_tokens=760), # hard
    ]
    tiers = {
        label_from_trace(_evidence(question_id=str(i), **s), cfg).difficulty
        for i, s in enumerate(shapes)
    }
    assert tiers == {Difficulty.EASY, Difficulty.MEDIUM, Difficulty.HARD}
