"""Reward function tests.

The difficulty-aware ordering is the project's central claim, so it is asserted
directly rather than left implicit in the config.
"""

from __future__ import annotations

import pytest

from adaptive_reasoning.config import load_config
from adaptive_reasoning.rl.reward import (
    StepOutcome,
    continue_reward,
    oracle_return,
    stop_reward,
    token_cost,
)

BUDGET = 768


@pytest.fixture
def cfg():
    return load_config()


def _outcome(**overrides) -> StepOutcome:
    base = {"tokens_so_far": 200, "probe_correct": True, "answer_changed": False,
            "tokens_in_next_step": 48}
    base.update(overrides)
    return StepOutcome(**base)


# --------------------------------------------------------------------------- #
# the difficulty-aware mechanism
# --------------------------------------------------------------------------- #
def test_token_cost_is_ordered_by_difficulty(cfg):
    """Wasting tokens on an easy question must cost more than on a hard one."""
    assert (token_cost(cfg, "easy") > token_cost(cfg, "medium")
            > token_cost(cfg, "hard") > 0)


def test_unlabelled_difficulty_gets_the_middle_rate(cfg):
    assert token_cost(cfg, None) == token_cost(cfg, "medium")
    assert token_cost(cfg, "nonsense") == token_cost(cfg, "medium")


def test_same_trace_is_penalised_more_when_easy(cfg):
    """The identical trace should score worse if the question was easy."""
    outcome = _outcome(tokens_so_far=600)
    easy = stop_reward(outcome, cfg, "easy", BUDGET)
    hard = stop_reward(outcome, cfg, "hard", BUDGET)
    assert easy < hard


def test_continue_is_cheaper_on_hard_questions(cfg):
    outcome = _outcome()
    assert continue_reward(outcome, cfg, "hard", BUDGET) > continue_reward(
        outcome, cfg, "easy", BUDGET
    )


# --------------------------------------------------------------------------- #
# stop reward
# --------------------------------------------------------------------------- #
def test_correct_beats_incorrect(cfg):
    right = stop_reward(_outcome(probe_correct=True), cfg, "medium", BUDGET)
    wrong = stop_reward(_outcome(probe_correct=False), cfg, "medium", BUDGET)
    assert right > wrong


def test_stopping_earlier_scores_better_when_both_are_correct(cfg):
    early = stop_reward(_outcome(tokens_so_far=100), cfg, "medium", BUDGET)
    late = stop_reward(_outcome(tokens_so_far=700), cfg, "medium", BUDGET)
    assert early > late


def test_stability_bonus_applies_only_when_the_answer_settled(cfg):
    settled = stop_reward(_outcome(answer_changed=False), cfg, "medium", BUDGET)
    moving = stop_reward(_outcome(answer_changed=True), cfg, "medium", BUDGET)
    assert settled - moving == pytest.approx(cfg.rl.reward.stability_bonus)


def test_accuracy_dominates_token_cost(cfg):
    """A correct late answer must still beat a wrong early one, or the policy will
    learn to answer instantly and always be wrong."""
    correct_late = stop_reward(
        _outcome(probe_correct=True, tokens_so_far=BUDGET), cfg, "easy", BUDGET
    )
    wrong_early = stop_reward(
        _outcome(probe_correct=False, tokens_so_far=0), cfg, "easy", BUDGET
    )
    assert correct_late > wrong_early


# --------------------------------------------------------------------------- #
# continue reward
# --------------------------------------------------------------------------- #
def test_continue_is_never_positive(cfg):
    assert continue_reward(_outcome(), cfg, "hard", BUDGET) <= 0


def test_continue_is_free_when_no_tokens_follow(cfg):
    assert continue_reward(_outcome(tokens_in_next_step=0), cfg, "easy", BUDGET) == 0.0


def test_longer_next_step_costs_more(cfg):
    short = continue_reward(_outcome(tokens_in_next_step=10), cfg, "medium", BUDGET)
    long = continue_reward(_outcome(tokens_in_next_step=200), cfg, "medium", BUDGET)
    assert long < short


# --------------------------------------------------------------------------- #
# oracle
# --------------------------------------------------------------------------- #
def test_oracle_picks_the_first_correct_step(cfg):
    steps = [
        StepOutcome(48, False, True, 48),
        StepOutcome(96, True, False, 48),
        StepOutcome(144, True, False, 0),
    ]
    value, index = oracle_return(steps, cfg, "medium", BUDGET)
    assert index == 1
    assert value > 0


def test_oracle_on_a_never_correct_trace_still_returns_a_step(cfg):
    steps = [StepOutcome(48, False, False, 48), StepOutcome(96, False, False, 0)]
    value, index = oracle_return(steps, cfg, "medium", BUDGET)
    assert index == 0, "if nothing is ever right, stop immediately and save the tokens"
    assert value < 0


def test_oracle_beats_stopping_at_the_end(cfg):
    steps = [
        StepOutcome(48, True, False, 48),
        StepOutcome(400, True, False, 48),
        StepOutcome(700, True, False, 0),
    ]
    value, index = oracle_return(steps, cfg, "easy", BUDGET)
    assert index == 0
    assert value > stop_reward(steps[-1], cfg, "easy", BUDGET)


def test_oracle_handles_a_single_step(cfg):
    value, index = oracle_return([StepOutcome(40, True, False, 0)], cfg, "easy", BUDGET)
    assert index == 0
    assert value == pytest.approx(stop_reward(StepOutcome(40, True, False, 0), cfg,
                                              "easy", BUDGET))
