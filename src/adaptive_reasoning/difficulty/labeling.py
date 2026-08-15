"""Deriving difficulty labels from repeated sampling.

No dataset ships with difficulty labels, and hand-labelling 30,000 questions is not
possible. So we *measure* it: run the model ``k`` times per question and observe how
it copes.

This gives **model-perceived difficulty**, which is the right notion for a stopping
policy. A question a human finds hard but the model answers instantly should be
labelled easy, because the policy's job is to predict how long *the model* needs - not
how hard the question is in the abstract.

Three signals go into the label:

* **pass rate** - how many of the ``k`` samples were correct. The primary signal.
* **reasoning length** - a question answered correctly every time but only after 600
  tokens of deliberation is not easy. Such questions are bumped one tier.
* **answer instability** - when the model gives a different answer nearly every time,
  it is guessing. A question can be demoted to hard on this basis even if it happened
  to score a lucky pass.

This module is deliberately free of any LLM dependency: it consumes sample *results*,
so the rules can be tuned and tested in milliseconds without touching a GPU.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from ..config import Config
from ..schema import Difficulty


@dataclass
class SampleOutcome:
    """One model attempt at one question."""

    question_id: str
    answer: str
    correct: bool
    reasoning_tokens: int


@dataclass
class DifficultyVerdict:
    """The label for one question, plus the evidence behind it."""

    question_id: str
    difficulty: Difficulty
    pass_rate: float
    n_samples: int
    median_reasoning_tokens: float
    answer_diversity: float          # distinct answers / samples, 0..1
    reason: str                      # which rule fired, for auditing

    @property
    def score(self) -> float:
        """Stored as ``difficulty_score`` in the unified dataset."""
        return self.pass_rate


def _median(values: list[int]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[mid])
    return (ordered[mid - 1] + ordered[mid]) / 2


_ORDER = [Difficulty.EASY, Difficulty.MEDIUM, Difficulty.HARD]


def _bump(level: Difficulty) -> Difficulty:
    """Move one tier harder, saturating at hard."""
    return _ORDER[min(_ORDER.index(level) + 1, len(_ORDER) - 1)]


def label_question(outcomes: list[SampleOutcome], cfg: Config) -> DifficultyVerdict:
    """Assign a difficulty tier to one question from its sample outcomes."""
    if not outcomes:
        raise ValueError("cannot label a question with no sample outcomes")

    n = len(outcomes)
    n_correct = sum(1 for o in outcomes if o.correct)
    pass_rate = n_correct / n

    # Reasoning length is measured on the *correct* attempts: a wrong answer that
    # rambled for 800 tokens says nothing about how long the question really needs.
    correct_tokens = [o.reasoning_tokens for o in outcomes if o.correct]
    median_tokens = _median(correct_tokens or [o.reasoning_tokens for o in outcomes])

    distinct = len({o.answer.strip().lower() for o in outcomes if o.answer.strip()})
    diversity = distinct / n if n else 0.0

    d = cfg.difficulty

    if pass_rate >= d.easy_min_pass_rate:
        level, reason = Difficulty.EASY, f"pass_rate {pass_rate:.2f} >= {d.easy_min_pass_rate}"
    elif pass_rate <= d.hard_max_pass_rate:
        level, reason = Difficulty.HARD, f"pass_rate {pass_rate:.2f} <= {d.hard_max_pass_rate}"
    else:
        level, reason = Difficulty.MEDIUM, f"pass_rate {pass_rate:.2f} between thresholds"

    # Solved, but only after a long think - not genuinely easy.
    if level is not Difficulty.HARD and median_tokens > d.long_reasoning_token_threshold:
        level = _bump(level)
        reason += f"; bumped for {median_tokens:.0f} reasoning tokens"

    # Every sample disagreed with every other: the model is guessing, not reasoning.
    if n >= 3 and diversity == 1.0 and level is Difficulty.MEDIUM:
        level = Difficulty.HARD
        reason += "; bumped for fully unstable answers"

    return DifficultyVerdict(
        question_id=outcomes[0].question_id,
        difficulty=level,
        pass_rate=pass_rate,
        n_samples=n,
        median_reasoning_tokens=median_tokens,
        answer_diversity=diversity,
        reason=reason,
    )


def label_all(outcomes: list[SampleOutcome], cfg: Config) -> list[DifficultyVerdict]:
    """Group outcomes by question and label each one."""
    grouped: dict[str, list[SampleOutcome]] = {}
    for outcome in outcomes:
        grouped.setdefault(outcome.question_id, []).append(outcome)
    return [label_question(items, cfg) for items in grouped.values()]


def distribution(verdicts: list[DifficultyVerdict]) -> dict[str, int]:
    return dict(Counter(str(v.difficulty) for v in verdicts))
