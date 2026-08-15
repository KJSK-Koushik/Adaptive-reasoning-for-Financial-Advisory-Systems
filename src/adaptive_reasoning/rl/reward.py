"""The reward function.

This is where "difficulty-aware" stops being a description and becomes a mechanism.

    STOP      R = +correct_bonus if the answer is right else incorrect_penalty
                  - beta(difficulty) * tokens_used / budget
                  + stability_bonus if the answer had settled

    CONTINUE  R = - beta(difficulty) * tokens_in_the_next_step / budget

``beta`` is the per-token price of thinking, and it is *higher for easy questions*:
``easy 0.60 > medium 0.30 > hard 0.12``. Wasting tokens on something the model already
knows is punished hard; on a genuinely hard question the agent is given room. A single
fixed beta - which is what every fixed-threshold baseline effectively assumes - cannot
express that.

**Difficulty enters the reward as the *measured* label, not the classifier's
prediction.** The reward is the environment, computed offline where ground truth is
available; the classifier's noisier estimate belongs in the *state*, because that is
what the agent will actually observe at inference time. Conflating the two would
either leak ground truth into the policy's input or add noise to the training signal
for no reason.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..config import Config


@dataclass
class StepOutcome:
    """What is known about one probe point, for reward purposes."""

    tokens_so_far: int
    probe_correct: bool
    answer_changed: bool
    tokens_in_next_step: int = 0


def token_cost(cfg: Config, difficulty: str | None) -> float:
    """The per-token price of thinking for this difficulty tier.

    ``cost_multiplier`` scales all three tiers together, leaving the difficulty-aware
    *ratios* intact. Sweeping it walks the accuracy-versus-cost frontier.
    """
    costs = cfg.rl.reward.token_cost
    if difficulty in {"easy", "medium", "hard"}:
        base = costs.for_difficulty(difficulty)
    else:
        # Unlabelled questions get the middle rate rather than a free pass.
        base = costs.medium
    return base * cfg.rl.reward.cost_multiplier


def stop_reward(outcome: StepOutcome, cfg: Config, difficulty: str | None,
                budget: int) -> float:
    """Reward for answering now."""
    r = cfg.rl.reward
    value = r.correct_bonus if outcome.probe_correct else r.incorrect_penalty
    value -= token_cost(cfg, difficulty) * outcome.tokens_so_far / max(budget, 1)
    if not outcome.answer_changed:
        # A small nudge toward stopping once the answer has settled, rather than
        # holding on for one more step "just in case".
        value += r.stability_bonus
    return float(value)


def continue_reward(outcome: StepOutcome, cfg: Config, difficulty: str | None,
                    budget: int) -> float:
    """Reward for thinking on: the price of the tokens the next step will burn."""
    cost = token_cost(cfg, difficulty) * outcome.tokens_in_next_step / max(budget, 1)
    return float(-cost)


def oracle_return(steps: list[StepOutcome], cfg: Config, difficulty: str | None,
                  budget: int) -> tuple[float, int]:
    """Best achievable discounted return over a trace, and the step that achieves it.

    Used as the ceiling in Phase 6: it answers "what would a policy with perfect
    foresight have scored on this question?", which is what makes a DQN result
    interpretable rather than just a number.
    """
    gamma = cfg.rl.dqn.gamma
    best_value, best_step = float("-inf"), 0

    for index, step in enumerate(steps):
        # Cost of continuing through every earlier step, then stopping here.
        value = 0.0
        for earlier in range(index):
            value += (gamma ** earlier) * continue_reward(
                steps[earlier], cfg, difficulty, budget
            )
        value += (gamma ** index) * stop_reward(step, cfg, difficulty, budget)
        if value > best_value:
            best_value, best_step = value, index

    return best_value, best_step
