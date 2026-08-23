"""Phase 6 baselines: each policy must stop where it claims to."""

from __future__ import annotations

import numpy as np
import pytest

from adaptive_reasoning.eval import baselines as bl
from adaptive_reasoning.rl.rollout import Trace, evaluate, rollout


def _trace(n=6, question_id="q0", confidence=None, entropy=None, stability=None,
           correct=None, state_dim=14):
    """A trace whose state columns match the default rl.state_features order."""
    states = np.zeros((n, state_dim), dtype=np.float32)
    if confidence is not None:
        states[:, 3] = confidence
    if entropy is not None:
        states[:, 5] = entropy
    if stability is not None:
        states[:, 13] = stability
    return Trace(
        question_id=question_id,
        states=states,
        tokens=np.arange(1, n + 1) * 48,
        correct=np.array(correct if correct is not None else [False] * n, dtype=bool),
        difficulty="easy",
    )


def test_full_reasoning_never_stops_early():
    t = _trace(n=5)
    assert rollout(t, bl.full_reasoning()).stop_step == 4


@pytest.mark.parametrize("k", [0, 1, 3])
def test_fixed_step_stops_exactly_at_k(k):
    t = _trace(n=6)
    assert rollout(t, bl.fixed_step(k)).stop_step == k


def test_fixed_budget_stops_at_the_first_step_over_the_budget():
    t = _trace(n=6)                       # tokens are 48, 96, 144, 192, 240, 288
    r = rollout(t, bl.fixed_budget(150)(t))
    assert r.stop_step == 3               # 192 is the first at or above 150
    assert r.tokens == 192


def test_fixed_budget_never_stops_when_the_budget_exceeds_the_trace():
    t = _trace(n=4)
    assert rollout(t, bl.fixed_budget(10_000)(t)).stop_step == 3


def test_confidence_threshold_waits_for_the_threshold():
    t = _trace(n=5, confidence=[0.1, 0.2, 0.9, 0.95, 0.99])
    idx = 3
    assert rollout(t, bl.confidence_threshold(0.9, idx)).stop_step == 2
    assert rollout(t, bl.confidence_threshold(0.99, idx)).stop_step == 4


def test_entropy_threshold_stops_when_entropy_falls():
    t = _trace(n=5, entropy=[2.0, 1.5, 0.4, 0.3, 0.2])
    assert rollout(t, bl.entropy_threshold(0.5, 5)).stop_step == 2


def test_answer_stability_uses_normalised_units():
    # steps_since_answer_change is stored as steps / max_steps.
    max_steps = 32
    t = _trace(n=5, stability=[0.0, 1 / 32, 2 / 32, 3 / 32, 4 / 32])
    assert rollout(t, bl.answer_stability(3, 13, max_steps)).stop_step == 3
    assert rollout(t, bl.answer_stability(1, 13, max_steps)).stop_step == 1


def test_random_stop_is_reproducible_per_question():
    t = _trace(n=8, question_id="q-abc")
    first = rollout(t, bl.random_stop(0.3, seed=7)(t)).stop_step
    second = rollout(t, bl.random_stop(0.3, seed=7)(t)).stop_step
    assert first == second


def test_random_stop_differs_across_questions():
    a = _trace(n=12, question_id="alpha")
    b = _trace(n=12, question_id="beta")
    steps = {rollout(x, bl.random_stop(0.2, seed=1)(x)).stop_step for x in (a, b)}
    assert len(steps) > 1, "same draws for different questions - rng is not keyed on id"


def test_probability_one_stops_immediately_and_zero_never_stops():
    t = _trace(n=6)
    assert rollout(t, bl.random_stop(1.0, seed=3)(t)).stop_step == 0
    assert rollout(t, bl.random_stop(0.0, seed=3)(t)).stop_step == 5


def test_feature_index_reads_the_configured_order(cfg):
    assert bl.feature_index(cfg, "confidence") == 3
    assert bl.feature_index(cfg, "entropy") == 5
    with pytest.raises(ValueError, match="does-not-exist"):
        bl.feature_index(cfg, "does-not-exist")


def test_tune_threshold_picks_the_best_scoring_candidate():
    # Correct only at step 4; a threshold that stops there should win.
    traces = [_trace(n=6, confidence=[0.1, 0.2, 0.3, 0.4, 0.95, 0.99],
                     correct=[False] * 4 + [True, True], question_id=f"q{i}")
              for i in range(5)]
    value, metrics = bl.tune_threshold(
        traces, lambda t: bl.confidence_threshold(t, 3), [0.2, 0.95],
        score=lambda m: m["accuracy"])
    assert value == 0.95
    assert metrics["accuracy"] == 1.0


def test_tune_threshold_handles_per_trace_factories():
    traces = [_trace(n=6, correct=[False] * 3 + [True] * 3, question_id=f"q{i}")
              for i in range(4)]
    value, metrics = bl.tune_threshold(
        traces, lambda b: bl.fixed_budget(int(b)), [48, 192],
        score=lambda m: m["accuracy"], per_trace=True)
    assert value == 192
    assert metrics["accuracy"] == 1.0


def test_tune_threshold_rejects_an_empty_grid():
    with pytest.raises(ValueError, match="no candidate"):
        bl.tune_threshold([_trace()], bl.fixed_step, [], score=lambda m: 0.0)


def test_every_baseline_shares_the_rollout_interface():
    """The comparison is only fair because all policies are measured identically."""
    traces = [_trace(n=6, confidence=0.5, entropy=0.5, stability=0.5,
                     correct=[False, False, True, True, False, False],
                     question_id=f"q{i}") for i in range(3)]
    plain = [bl.full_reasoning(), bl.fixed_step(2),
             bl.confidence_threshold(0.4, 3), bl.entropy_threshold(0.6, 5),
             bl.answer_stability(1, 13, 32)]
    for policy in plain:
        assert set(evaluate(traces, policy)) >= {"accuracy", "mean_tokens",
                                                 "token_reduction_pct"}
    for factory in (bl.fixed_budget(100), bl.random_stop(0.5, 1)):
        assert evaluate(traces, factory, per_trace=True)["n"] == 3
