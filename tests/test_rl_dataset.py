from __future__ import annotations

import numpy as np
import pytest

from adaptive_reasoning.config import load_config
from adaptive_reasoning.rl.dataset import (
    ACTION_CONTINUE,
    ACTION_STOP,
    _outcomes,
    transitions_for_trace,
)
from adaptive_reasoning.rl.features import difficulty_one_hot

BUDGET = 768


@pytest.fixture
def cfg():
    return load_config()


def _steps(n=4, correct_from=2):
    return [
        {
            "step_index": i,
            "tokens_so_far": (i + 1) * 48,
            "step_text": "reasoning",
            "confidence": 0.6,
            "min_token_confidence": 0.5,
            "entropy": 0.8,
            "answer_changed": False,
            "probe_correct": i >= correct_from,
        }
        for i in range(n)
    ]


def _build(cfg, steps, difficulty="medium"):
    return transitions_for_trace(
        "q1", steps, difficulty, difficulty_one_hot(difficulty), cfg, BUDGET
    )


# --------------------------------------------------------------------------- #
# structure
# --------------------------------------------------------------------------- #
def test_both_actions_are_emitted_for_every_step(cfg):
    rows = _build(cfg, _steps(4))
    assert len(rows) == 8
    for index in range(4):
        actions = {r["action"] for r in rows if r["step_index"] == index}
        assert actions == {ACTION_STOP, ACTION_CONTINUE}


def test_stop_transitions_are_always_terminal(cfg):
    rows = _build(cfg, _steps(4))
    assert all(r["done"] for r in rows if r["action"] == ACTION_STOP)


def test_continue_is_non_terminal_except_at_the_end(cfg):
    rows = _build(cfg, _steps(4))
    conts = sorted(
        (r for r in rows if r["action"] == ACTION_CONTINUE),
        key=lambda r: r["step_index"],
    )
    assert [r["done"] for r in conts] == [False, False, False, True]


def test_continue_at_the_last_step_scores_the_same_as_stopping(cfg):
    """There is nothing left to continue into, so the outcome is the forced stop.

    Dropping the transition instead would leave the network with no gradient for
    CONTINUE in exactly the states where it most needs to have learned to stop.
    """
    rows = _build(cfg, _steps(3))
    last = max(r["step_index"] for r in rows)
    stop = next(r for r in rows if r["step_index"] == last and r["action"] == ACTION_STOP)
    cont = next(r for r in rows if r["step_index"] == last and r["action"] == ACTION_CONTINUE)
    assert cont["reward"] == stop["reward"]
    assert cont["done"]


def test_next_state_chains_to_the_following_step(cfg):
    rows = _build(cfg, _steps(4))
    by_step = {}
    for r in rows:
        by_step.setdefault(r["step_index"], {})[r["action"]] = r
    for index in range(3):
        assert by_step[index][ACTION_CONTINUE]["next_state"] == by_step[index + 1][ACTION_STOP]["state"]


def test_terminal_transitions_have_a_zero_next_state(cfg):
    rows = _build(cfg, _steps(3))
    for r in rows:
        if r["done"]:
            assert not np.any(r["next_state"])


def test_empty_trace_yields_nothing(cfg):
    assert _build(cfg, []) == []


def test_single_step_trace(cfg):
    rows = _build(cfg, _steps(1))
    assert len(rows) == 2
    assert all(r["done"] for r in rows)


# --------------------------------------------------------------------------- #
# rewards inside the transitions
# --------------------------------------------------------------------------- #
def test_correct_steps_earn_more_than_incorrect(cfg):
    rows = _build(cfg, _steps(4, correct_from=2))
    stops = {r["step_index"]: r["reward"] for r in rows if r["action"] == ACTION_STOP}
    assert stops[2] > stops[0]


def test_difficulty_changes_the_rewards(cfg):
    steps = _steps(4)
    easy = _build(cfg, steps, "easy")
    hard = _build(cfg, steps, "hard")
    easy_stop = [r["reward"] for r in easy if r["action"] == ACTION_STOP]
    hard_stop = [r["reward"] for r in hard if r["action"] == ACTION_STOP]
    assert all(e < h for e, h in zip(easy_stop, hard_stop, strict=True))


def test_outcomes_measure_the_gap_to_the_next_step(cfg):
    outcomes = _outcomes(_steps(3))
    assert [o.tokens_in_next_step for o in outcomes] == [48, 48, 0]


def test_outcomes_never_report_negative_token_gaps(cfg):
    """Guards against a malformed trace where tokens_so_far goes backwards."""
    steps = _steps(3)
    steps[1]["tokens_so_far"] = 10
    assert all(o.tokens_in_next_step >= 0 for o in _outcomes(steps))


def test_state_dimension_is_consistent(cfg):
    rows = _build(cfg, _steps(5))
    assert all(len(r["state"]) == cfg.rl.state_dim for r in rows)
    assert all(len(r["next_state"]) == cfg.rl.state_dim for r in rows)


def test_every_row_has_the_same_columns(cfg):
    """Regression: the STOP and CONTINUE branches drifted apart.

    Two of the three row literals lost ``tokens_in_next_step`` and ``answer_changed``,
    which parquet then filled with NaN. ``recompute_rewards`` produced NaN rewards, and
    the DQN trained on them from the first step with no error - the only symptom was a
    policy that never learned.
    """
    rows = _build(cfg, _steps(4))
    keys = {frozenset(r) for r in rows}
    assert len(keys) == 1, f"row schemas differ: {keys}"


def test_reward_recompute_columns_are_present_on_every_row(cfg):
    required = {"tokens_in_next_step", "answer_changed", "probe_correct",
                "tokens_so_far", "difficulty", "action", "done"}
    for row in _build(cfg, _steps(4)):
        assert required <= set(row)
        assert row["tokens_in_next_step"] is not None
        assert row["answer_changed"] is not None
