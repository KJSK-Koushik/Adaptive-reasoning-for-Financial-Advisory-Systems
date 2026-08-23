"""Phase 7 controller.

The safety rules and the stop bookkeeping are what a demonstration rests on, so they
are tested directly rather than only through the end-to-end script.
"""

from __future__ import annotations

import numpy as np
import pytest

from adaptive_reasoning.serve.controller import (
    BY_POLICY,
    BY_STREAM_END,
    BY_TOKEN_CAP,
    CONTINUE,
    STOP,
    AdaptiveController,
    ReplaySource,
    always_continue_policy,
    compare,
    load_policy,
)


def _steps(n=6, tokens=48, answer="42"):
    """Step dicts with every key rl.features reads."""
    return [
        {
            "step_index": i,
            "tokens_so_far": (i + 1) * tokens,
            "step_text": f"step {i} reasoning text",
            "probe_answer": f"{answer}-{i}",
            "probe_correct": i >= 3,
            "confidence": 0.5 + 0.05 * i,
            "min_token_confidence": 0.3,
            "entropy": 1.0 - 0.1 * i,
            "answer_changed": i == 0,
            "is_terminal": i == n - 1,
        }
        for i in range(n)
    ]


def _source(n=6, tokens=48, question_id="q1"):
    return ReplaySource(question_id, _steps(n, tokens))


def stop_at(k):
    return lambda state, index: index >= k


# --------------------------------------------------------------------------- #
# ReplaySource
# --------------------------------------------------------------------------- #
def test_replay_source_sorts_steps_by_index():
    scrambled = list(reversed(_steps(5)))
    source = ReplaySource("q", scrambled)
    assert [s["step_index"] for s in source.steps()] == [0, 1, 2, 3, 4]


def test_replay_source_reports_full_reasoning_cost():
    assert _source(n=5, tokens=48).total_tokens == 240


def test_from_frame_raises_for_an_unknown_question(cfg):
    import pandas as pd

    frame = pd.DataFrame(_steps(3)).assign(question_id="known")
    with pytest.raises(KeyError, match="nope"):
        ReplaySource.from_frame(frame, "nope")


# --------------------------------------------------------------------------- #
# stopping behaviour
# --------------------------------------------------------------------------- #
def test_controller_stops_where_the_policy_says(cfg):
    out = AdaptiveController(cfg, stop_at(2), min_steps=0).run(_source())
    assert out.stop_step == 2
    assert out.stop_reason == BY_POLICY
    assert out.decisions[-1].action == STOP
    assert all(d.action == CONTINUE for d in out.decisions[:-1])


def test_controller_runs_to_the_end_when_the_policy_never_stops(cfg):
    out = AdaptiveController(cfg, always_continue_policy(), min_steps=0).run(_source(n=6))
    assert out.stop_step == 5
    assert out.stop_reason == BY_STREAM_END


def test_min_steps_floor_blocks_an_early_stop(cfg):
    """A policy that wants to stop immediately must still wait for the floor."""
    out = AdaptiveController(cfg, stop_at(0), min_steps=3).run(_source())
    assert out.stop_step == 3


def test_hard_token_cap_overrides_the_policy(cfg):
    # 6 steps of 400 tokens; the cap is well below the last of them.
    out = AdaptiveController(cfg, always_continue_policy(), min_steps=0).run(
        _source(n=6, tokens=400))
    assert out.stop_reason == BY_TOKEN_CAP
    assert out.tokens_used >= cfg.serve.hard_token_cap


def test_the_cap_takes_precedence_over_a_continue_decision(cfg):
    out = AdaptiveController(cfg, always_continue_policy(), min_steps=0).run(
        _source(n=8, tokens=300))
    stops = [d for d in out.decisions if d.stopped]
    assert len(stops) == 1
    assert stops[0].reason == BY_TOKEN_CAP


def test_empty_source_is_rejected(cfg):
    with pytest.raises(ValueError, match="no steps"):
        AdaptiveController(cfg, stop_at(0)).run(ReplaySource("q", []))


# --------------------------------------------------------------------------- #
# bookkeeping
# --------------------------------------------------------------------------- #
def test_outcome_token_arithmetic(cfg):
    out = AdaptiveController(cfg, stop_at(1), min_steps=0).run(_source(n=6, tokens=48))
    assert out.tokens_used == 96          # stopped at step 1
    assert out.tokens_available == 288    # full reasoning would have been 6 * 48
    assert out.tokens_saved == 192
    assert out.token_reduction_pct == pytest.approx(66.7, abs=0.1)


def test_reduction_is_zero_when_nothing_was_saved(cfg):
    out = AdaptiveController(cfg, always_continue_policy(), min_steps=0).run(_source(n=4))
    assert out.tokens_saved == 0
    assert out.token_reduction_pct == 0.0


def test_answer_is_the_one_held_at_the_stopping_step(cfg):
    out = AdaptiveController(cfg, stop_at(2), min_steps=0).run(_source())
    assert out.answer == "42-2"


def test_decisions_record_the_observed_signals(cfg):
    out = AdaptiveController(cfg, stop_at(3), min_steps=0).run(_source())
    first = out.decisions[0]
    assert first.confidence == pytest.approx(0.5)
    assert first.entropy == pytest.approx(1.0)
    assert first.tokens_so_far == 48
    assert "step 0" in first.step_text


def test_on_decision_callback_sees_every_step(cfg):
    seen = []
    AdaptiveController(cfg, stop_at(3), min_steps=0).run(
        _source(), on_decision=seen.append)
    assert [d.step_index for d in seen] == [0, 1, 2, 3]
    assert seen[-1].stopped


def test_disclaimer_is_attached(cfg):
    out = AdaptiveController(cfg, stop_at(1), min_steps=0).run(_source())
    assert "not personalised investment advice" in out.disclaimer


def test_summary_is_json_friendly(cfg):
    import json

    out = AdaptiveController(cfg, stop_at(1), min_steps=0).run(_source())
    assert json.loads(json.dumps(out.summary()))["stop_step"] == 1


# --------------------------------------------------------------------------- #
# the budget matters - this was a real bug
# --------------------------------------------------------------------------- #
def test_budget_changes_the_token_ratio_feature(cfg):
    """token_ratio is normalised by the training budget, not the current config.

    Feeding the wrong budget shifts this feature by up to 0.25, which is enough to
    change decisions - the Phase 7 consistency check exists because of exactly this.
    """
    captured = {}

    def spy(state, index):
        captured.setdefault(index, state.copy())
        return False

    AdaptiveController(cfg, spy, min_steps=0, budget=768).run(_source(n=3, tokens=192))
    at_768 = captured[2][7]
    captured.clear()
    AdaptiveController(cfg, spy, min_steps=0, budget=1024).run(_source(n=3, tokens=192))
    at_1024 = captured[2][7]

    assert at_768 == pytest.approx(576 / 768)
    assert at_1024 == pytest.approx(576 / 1024)
    assert at_768 != at_1024


# --------------------------------------------------------------------------- #
# comparison and loading
# --------------------------------------------------------------------------- #
def test_compare_reports_both_runs_and_the_saving(cfg):
    result = compare(cfg, _source(n=6, tokens=48), stop_at(1), min_steps=0)
    assert result["adaptive"].stop_step == 1
    assert result["full"].stop_step == 5
    assert result["tokens_saved"] == 192
    assert result["answer_changed"] is True


def test_compare_detects_an_unchanged_answer(cfg):
    steps = _steps(4)
    for s in steps:
        s["probe_answer"] = "same"
    result = compare(cfg, ReplaySource("q", steps), stop_at(1), min_steps=0)
    assert result["answer_changed"] is False


def test_load_policy_rejects_an_unknown_kind(cfg):
    with pytest.raises(ValueError, match="unknown policy kind"):
        load_policy(cfg, kind="magic")


def test_controller_accepts_an_explicit_difficulty_vector(cfg):
    captured = {}

    def spy(state, index):
        captured[index] = state.copy()
        return index >= 1

    vector = np.array([0.7, 0.2, 0.1], dtype=np.float32)
    AdaptiveController(cfg, spy, difficulty_vector=vector, min_steps=0).run(_source())
    assert captured[0][:3] == pytest.approx(vector)


def test_difficulty_string_becomes_a_one_hot(cfg):
    captured = {}

    def spy(state, index):
        captured[index] = state.copy()
        return True

    AdaptiveController(cfg, spy, difficulty="hard", min_steps=0).run(_source())
    assert captured[0][:3] == pytest.approx([0.0, 0.0, 1.0])
