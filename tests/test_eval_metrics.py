"""Phase 6 statistics.

Every headline claim in the report rests on these, so they are tested against cases
with known answers rather than only for self-consistency.
"""

from __future__ import annotations

import numpy as np
import pytest

from adaptive_reasoning.eval.metrics import (
    mcnemar,
    paired_bootstrap,
    token_cost_model,
    wilson_interval,
)


# --------------------------------------------------------------------------- #
# Wilson interval
# --------------------------------------------------------------------------- #
def test_wilson_brackets_the_point_estimate():
    low, high = wilson_interval(194, 599)
    assert low < 194 / 599 < high


def test_wilson_stays_inside_zero_and_one_at_the_extremes():
    for successes, n in ((0, 30), (30, 30), (1, 1000)):
        low, high = wilson_interval(successes, n)
        assert 0.0 <= low <= high <= 1.0


def test_wilson_narrows_as_the_sample_grows():
    small = wilson_interval(50, 100)
    large = wilson_interval(500, 1000)
    assert (large[1] - large[0]) < (small[1] - small[0])


def test_wilson_handles_an_empty_sample():
    assert wilson_interval(0, 0) == (0.0, 0.0)


# --------------------------------------------------------------------------- #
# paired bootstrap
# --------------------------------------------------------------------------- #
def test_bootstrap_reports_the_observed_difference():
    a = np.array([1, 1, 1, 0, 0], dtype=bool)
    b = np.array([1, 0, 0, 0, 0], dtype=bool)
    out = paired_bootstrap(a, b, n_samples=200, seed=1)
    assert out["difference"] == pytest.approx(0.4)


def test_bootstrap_interval_contains_the_observed_difference():
    rng = np.random.default_rng(0)
    a = rng.random(400) < 0.35
    b = rng.random(400) < 0.28
    out = paired_bootstrap(a, b, n_samples=500, seed=2)
    assert out["ci_low"] <= out["difference"] <= out["ci_high"]


def test_identical_policies_have_zero_difference_and_a_degenerate_interval():
    a = np.array([1, 0, 1, 1, 0], dtype=bool)
    out = paired_bootstrap(a, a, n_samples=200, seed=3)
    assert out["difference"] == 0.0
    assert out["ci_low"] == out["ci_high"] == 0.0


def test_bootstrap_is_reproducible_for_a_fixed_seed():
    rng = np.random.default_rng(5)
    a, b = rng.random(200) < 0.4, rng.random(200) < 0.3
    first = paired_bootstrap(a, b, n_samples=300, seed=11)
    second = paired_bootstrap(a, b, n_samples=300, seed=11)
    assert first == second


def test_bootstrap_flags_an_unreliable_difference():
    """A tiny, noisy edge should show a high chance of the sign flipping."""
    rng = np.random.default_rng(7)
    a = rng.random(120) < 0.31
    b = rng.random(120) < 0.30
    out = paired_bootstrap(a, b, n_samples=800, seed=9)
    assert out["p_sign_flip"] > 0.1


def test_bootstrap_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="equal lengths"):
        paired_bootstrap(np.ones(5, dtype=bool), np.ones(4, dtype=bool))


def test_bootstrap_handles_an_empty_sample():
    out = paired_bootstrap(np.array([], dtype=bool), np.array([], dtype=bool))
    assert out["difference"] == 0.0


# --------------------------------------------------------------------------- #
# McNemar
# --------------------------------------------------------------------------- #
def test_mcnemar_counts_only_the_disagreements():
    a = np.array([1, 1, 1, 0, 1, 0], dtype=bool)
    b = np.array([1, 0, 0, 0, 1, 1], dtype=bool)
    out = mcnemar(a, b)
    assert out["only_a"] == 2      # positions 1 and 2
    assert out["only_b"] == 1      # position 5


def test_mcnemar_is_not_significant_when_disagreements_are_balanced():
    a = np.array([1, 0, 1, 0], dtype=bool)
    b = np.array([0, 1, 0, 1], dtype=bool)
    assert mcnemar(a, b)["p_value"] == pytest.approx(1.0)


def test_mcnemar_is_significant_for_a_one_sided_sweep():
    a = np.ones(12, dtype=bool)
    b = np.zeros(12, dtype=bool)
    out = mcnemar(a, b)
    assert out["only_b"] == 0
    assert out["p_value"] < 0.001


def test_mcnemar_matches_the_hand_computed_exact_value():
    # 5 disagreements, all one way: two-sided p = 2 * (1/2)^5 = 0.0625
    a = np.array([1, 1, 1, 1, 1, 1], dtype=bool)
    b = np.array([0, 0, 0, 0, 0, 1], dtype=bool)
    assert mcnemar(a, b)["p_value"] == pytest.approx(0.0625)


def test_mcnemar_is_symmetric_in_its_arguments():
    rng = np.random.default_rng(4)
    a, b = rng.random(80) < 0.4, rng.random(80) < 0.3
    assert mcnemar(a, b)["p_value"] == mcnemar(b, a)["p_value"]


def test_mcnemar_returns_one_when_the_policies_never_disagree():
    a = np.array([1, 0, 1], dtype=bool)
    assert mcnemar(a, a)["p_value"] == 1.0


# --------------------------------------------------------------------------- #
# cost model
# --------------------------------------------------------------------------- #
def test_cost_scales_linearly_with_tokens():
    one = token_cost_model(100.0)
    two = token_cost_model(200.0)
    assert two["latency_seconds"] == pytest.approx(2 * one["latency_seconds"], rel=1e-6)
    assert two["energy_joules"] == pytest.approx(2 * one["energy_joules"], rel=1e-3)


def test_cost_uses_the_measured_throughput():
    # 91.5 tokens/second was measured by the Phase 3 pilot on a Kaggle T4.
    assert token_cost_model(915.0)["latency_seconds"] == pytest.approx(10.0, abs=0.01)


def test_cost_of_zero_tokens_is_zero():
    out = token_cost_model(0.0)
    assert out["latency_seconds"] == 0.0
    assert out["energy_joules"] == 0.0
