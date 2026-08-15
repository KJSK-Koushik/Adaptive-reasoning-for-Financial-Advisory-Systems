"""Rollout tests.

Every reported number in Phases 5 and 6 goes through this module, so a bug here
silently corrupts the entire comparison between policies.
"""

from __future__ import annotations

import numpy as np
import pytest

from adaptive_reasoning.rl.rollout import (
    Trace,
    always_continue,
    evaluate,
    oracle,
    rollout,
)


def _trace(correct, tokens=None, difficulty="medium", qid="q1") -> Trace:
    n = len(correct)
    tokens = tokens or [(i + 1) * 50 for i in range(n)]
    return Trace(
        question_id=qid,
        states=np.zeros((n, 14), dtype=np.float32),
        tokens=np.array(tokens),
        correct=np.array(correct, dtype=bool),
        difficulty=difficulty,
    )


# --------------------------------------------------------------------------- #
# trace properties
# --------------------------------------------------------------------------- #
def test_full_cost_is_the_last_step():
    trace = _trace([False, True, True])
    assert trace.full_tokens == 150
    assert trace.full_correct


def test_earliest_correct():
    assert _trace([False, False, True]).earliest_correct == 2
    assert _trace([True, True]).earliest_correct == 0
    assert _trace([False, False]).earliest_correct is None


# --------------------------------------------------------------------------- #
# rollout mechanics
# --------------------------------------------------------------------------- #
def test_stops_at_the_first_true_decision():
    result = rollout(_trace([False, True, True]), lambda s, i: i == 1)
    assert result.stop_step == 1
    assert result.tokens == 100
    assert result.correct


def test_never_stopping_falls_through_to_the_end():
    result = rollout(_trace([False, False, True]), always_continue())
    assert result.stop_step == 2
    assert result.correct


def test_min_steps_blocks_an_immediate_stop():
    """The live system must never answer with no reasoning at all."""
    always_stop = lambda s, i: True   # noqa: E731
    assert rollout(_trace([True, True, True]), always_stop).stop_step == 0
    assert rollout(_trace([True, True, True]), always_stop, min_steps=2).stop_step == 2


def test_single_step_trace():
    result = rollout(_trace([True]), always_continue())
    assert result.stop_step == 0
    assert result.tokens == 50


# --------------------------------------------------------------------------- #
# reference policies
# --------------------------------------------------------------------------- #
def test_oracle_stops_at_the_earliest_correct_step():
    trace = _trace([False, False, True, True])
    result = rollout(trace, oracle(trace))
    assert result.stop_step == 2
    assert result.correct


def test_oracle_stops_immediately_when_never_correct():
    """Nothing can be salvaged, so save every token."""
    trace = _trace([False, False, False])
    result = rollout(trace, oracle(trace))
    assert result.stop_step == 0
    assert not result.correct


def test_oracle_is_never_worse_than_full_reasoning():
    traces = [_trace([False, True, False], qid="a"), _trace([True, False, False], qid="b")]
    full = evaluate(traces, always_continue())
    best = evaluate(traces, oracle, per_trace=True)
    assert best["accuracy"] >= full["accuracy"]
    assert best["mean_tokens"] <= full["mean_tokens"]


# --------------------------------------------------------------------------- #
# aggregation
# --------------------------------------------------------------------------- #
def test_full_reasoning_has_zero_token_reduction():
    traces = [_trace([False, True], qid=str(i)) for i in range(5)]
    assert evaluate(traces, always_continue())["token_reduction_pct"] == 0.0


def test_token_reduction_is_paired_against_the_same_traces():
    traces = [_trace([True, True, True, True], qid=str(i)) for i in range(4)]
    metrics = evaluate(traces, lambda s, i: i == 0)
    # Stopping at step 0 (50 tokens) instead of step 3 (200) is a 75% saving.
    assert metrics["token_reduction_pct"] == pytest.approx(75.0)


def test_accuracy_delta_is_relative_to_full_reasoning():
    traces = [_trace([False, False, True], qid=str(i)) for i in range(4)]
    metrics = evaluate(traces, lambda s, i: i == 0)
    assert metrics["accuracy"] == 0.0
    assert metrics["accuracy_delta_vs_full"] == pytest.approx(-1.0)


def test_per_tier_metrics_are_reported():
    traces = [
        _trace([True, True], difficulty="easy", qid="a"),
        _trace([False, True], difficulty="hard", qid="b"),
    ]
    metrics = evaluate(traces, always_continue())
    assert metrics["easy_n"] == 1
    assert metrics["hard_n"] == 1
    assert "easy_mean_tokens" in metrics


def test_stopped_early_percentage():
    traces = [_trace([True, True, True], qid=str(i)) for i in range(4)]
    assert evaluate(traces, lambda s, i: i == 0)["stopped_early_pct"] == 100.0
    assert evaluate(traces, always_continue())["stopped_early_pct"] == 0.0


def test_empty_input():
    assert evaluate([], always_continue()) == {}


def test_per_trace_factory_mode():
    """Used by the tree-based policies, which score a whole trace at once."""
    traces = [_trace([False, True, True], qid="a"), _trace([True, False, False], qid="b")]
    stops = {"a": 1, "b": 0}

    def factory(trace):
        target = stops[trace.question_id]
        return lambda s, i: i >= target

    metrics = evaluate(traces, factory, per_trace=True)
    assert metrics["accuracy"] == 1.0
