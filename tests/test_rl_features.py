"""State-feature tests.

The critical property is **no future leakage**: every feature must be computable from
what a live reasoning stream exposes at that moment. A feature that peeks ahead, or at
the gold answer, would train a policy on information it will not have at inference
time and make every offline number fiction.
"""

from __future__ import annotations

import numpy as np
import pytest

from adaptive_reasoning.config import load_config
from adaptive_reasoning.rl.features import build_states, difficulty_one_hot

BUDGET = 768


@pytest.fixture
def cfg():
    return load_config()


def _steps(n=4, **overrides):
    steps = []
    for i in range(n):
        step = {
            "step_index": i,
            "tokens_so_far": (i + 1) * 48,
            "step_text": "some reasoning here",
            "confidence": 0.5 + 0.1 * i,
            "min_token_confidence": 0.4 + 0.1 * i,
            "entropy": 1.0 - 0.1 * i,
            "answer_changed": False,
            "probe_correct": i >= 2,
        }
        step.update(overrides)
        steps.append(step)
    return steps


def _named(cfg, rows, index):
    return dict(zip(cfg.rl.state_features, rows[index], strict=True))


# --------------------------------------------------------------------------- #
# shape and ordering
# --------------------------------------------------------------------------- #
def test_shape_matches_the_configured_state_dim(cfg):
    rows = build_states(_steps(5), difficulty_one_hot("easy"), cfg, BUDGET)
    assert rows.shape == (5, cfg.rl.state_dim)


def test_missing_feature_implementation_is_caught(cfg):
    """A name in the config with no implementation must fail loudly, not silently zero."""
    broken = load_config(overrides={"rl": {"state_features": ["confidence", "invented"]}})
    with pytest.raises(ValueError, match="invented"):
        build_states(_steps(), difficulty_one_hot("easy"), broken, BUDGET)


def test_difficulty_features_come_from_the_supplied_vector(cfg):
    rows = build_states(_steps(), np.array([0.2, 0.3, 0.5], np.float32), cfg, BUDGET)
    values = _named(cfg, rows, 0)
    assert (values["difficulty_easy"], values["difficulty_medium"],
            values["difficulty_hard"]) == pytest.approx((0.2, 0.3, 0.5))


def test_one_hot_encoding():
    assert difficulty_one_hot("easy").tolist() == [1, 0, 0]
    assert difficulty_one_hot("hard").tolist() == [0, 0, 1]
    assert difficulty_one_hot(None).tolist() == pytest.approx([1 / 3, 1 / 3, 1 / 3])


# --------------------------------------------------------------------------- #
# no leakage
# --------------------------------------------------------------------------- #
def test_state_does_not_depend_on_future_steps(cfg):
    """Truncating the trace must not change the states that came before."""
    full = build_states(_steps(6), difficulty_one_hot("medium"), cfg, BUDGET)
    short = build_states(_steps(6)[:3], difficulty_one_hot("medium"), cfg, BUDGET)
    assert np.allclose(full[:3], short)


def test_state_ignores_probe_correctness(cfg):
    """probe_correct is the gold-answer comparison and must never reach the state."""
    right = build_states(_steps(4, probe_correct=True), difficulty_one_hot("easy"),
                         cfg, BUDGET)
    wrong = build_states(_steps(4, probe_correct=False), difficulty_one_hot("easy"),
                         cfg, BUDGET)
    assert np.allclose(right, wrong)


# --------------------------------------------------------------------------- #
# individual features
# --------------------------------------------------------------------------- #
def test_first_step_has_zero_slopes(cfg):
    rows = build_states(_steps(), difficulty_one_hot("easy"), cfg, BUDGET)
    values = _named(cfg, rows, 0)
    assert values["delta_confidence"] == 0.0
    assert values["entropy_slope"] == 0.0


def test_delta_confidence_tracks_the_change(cfg):
    rows = build_states(_steps(3), difficulty_one_hot("easy"), cfg, BUDGET)
    assert _named(cfg, rows, 1)["delta_confidence"] == pytest.approx(0.1, abs=1e-5)


def test_entropy_slope_is_negative_when_settling(cfg):
    rows = build_states(_steps(3), difficulty_one_hot("easy"), cfg, BUDGET)
    assert _named(cfg, rows, 1)["entropy_slope"] < 0


def test_token_ratio_grows_and_is_normalised(cfg):
    rows = build_states(_steps(4), difficulty_one_hot("easy"), cfg, BUDGET)
    ratios = [_named(cfg, rows, i)["token_ratio"] for i in range(4)]
    assert ratios == sorted(ratios)
    assert all(0 <= r <= 1 for r in ratios)


def test_answer_stability_resets_when_the_answer_moves(cfg):
    steps = _steps(4)
    steps[2]["answer_changed"] = True
    rows = build_states(steps, difficulty_one_hot("easy"), cfg, BUDGET)
    assert _named(cfg, rows, 2)["answer_stability"] == 0.0
    assert _named(cfg, rows, 3)["answer_stability"] > 0.0


def test_progress_cue_detects_wrap_up_language(cfg):
    plain = _steps(1, step_text="computing the ratio of the two values")
    wrap = _steps(1, step_text="therefore the answer is clear")
    a = _named(cfg, build_states(plain, difficulty_one_hot("easy"), cfg, BUDGET), 0)
    b = _named(cfg, build_states(wrap, difficulty_one_hot("easy"), cfg, BUDGET), 0)
    assert b["progress_cue"] > a["progress_cue"] == 0.0


def test_doubt_cue_detects_backtracking(cfg):
    plain = _steps(1, step_text="computing the ratio of the two values")
    doubt = _steps(1, step_text="wait, actually let me reconsider that")
    a = _named(cfg, build_states(plain, difficulty_one_hot("easy"), cfg, BUDGET), 0)
    b = _named(cfg, build_states(doubt, difficulty_one_hot("easy"), cfg, BUDGET), 0)
    assert b["doubt_cue"] > a["doubt_cue"] == 0.0


def test_cue_density_is_bounded(cfg):
    spam = _steps(1, step_text="wait wait wait wait wait wait wait wait")
    values = _named(cfg, build_states(spam, difficulty_one_hot("easy"), cfg, BUDGET), 0)
    assert 0.0 <= values["doubt_cue"] <= 1.0


def test_all_features_are_finite(cfg):
    rows = build_states(_steps(8), difficulty_one_hot("hard"), cfg, BUDGET)
    assert np.isfinite(rows).all()


def test_empty_trace_returns_empty_matrix(cfg):
    rows = build_states([], difficulty_one_hot("easy"), cfg, BUDGET)
    assert rows.shape == (0, cfg.rl.state_dim)
