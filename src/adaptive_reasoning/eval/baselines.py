"""Stopping baselines, all sharing one interface so the comparison is like for like.

Every policy here is a ``Decide`` - given a state vector and a step index, should we
stop? - or a factory returning one for a specific trace. That is the same interface the
DQN and behaviour cloning expose, so ``rl.rollout.evaluate`` measures all of them
through identical code and no policy gets an accidental advantage.

The threshold policies (confidence, entropy, stability) read a named feature out of the
state vector rather than a fixed column, so they keep working if ``rl.state_features``
is reordered.
"""

from __future__ import annotations

import random
from collections.abc import Callable

import numpy as np

from ..config import Config
from ..rl.rollout import Decide, Trace


def feature_index(cfg: Config, name: str) -> int:
    """Column of ``name`` in the state vector, by configured feature order."""
    try:
        return cfg.rl.state_features.index(name)
    except ValueError as exc:
        raise ValueError(
            f"baseline needs the '{name}' feature, but rl.state_features is "
            f"{cfg.rl.state_features}"
        ) from exc


# --------------------------------------------------------------------------- #
# policies that ignore the state entirely
# --------------------------------------------------------------------------- #
def full_reasoning() -> Decide:
    """Never stop early. The cost ceiling and the accuracy the model ships with."""
    return lambda state, index: False


def fixed_step(k: int) -> Decide:
    """Stop at step ``k`` regardless of the question. The simplest possible rule."""
    return lambda state, index: index >= k


def fixed_budget(max_tokens: int) -> Callable[[Trace], Decide]:
    """Stop once ``max_tokens`` have been generated.

    A token budget rather than a step count, which is the fairer fixed baseline: it
    spends the same compute on every question instead of the same number of steps,
    and steps vary in length.
    """

    def factory(trace: Trace) -> Decide:
        tokens = trace.tokens

        def decide(state: np.ndarray, index: int) -> bool:
            return bool(tokens[index] >= max_tokens)

        return decide

    return factory


def random_stop(probability: float, seed: int) -> Callable[[Trace], Decide]:
    """Stop with fixed probability at each step. The sanity floor.

    Any learned policy must beat this, or its apparent skill is just the cost of
    stopping early. Seeded per question so a rerun reproduces exactly.
    """

    def factory(trace: Trace) -> Decide:
        rng = random.Random(f"{seed}:{trace.question_id}")
        draws = [rng.random() for _ in range(trace.n_steps)]

        def decide(state: np.ndarray, index: int) -> bool:
            return draws[index] < probability

        return decide

    return factory


# --------------------------------------------------------------------------- #
# threshold policies - what prior early-exit work actually does
# --------------------------------------------------------------------------- #
def confidence_threshold(tau: float, index_of: int) -> Decide:
    """Stop once the model is confident enough in its current answer."""
    return lambda state, index: bool(state[index_of] >= tau)


def entropy_threshold(tau: float, index_of: int) -> Decide:
    """Stop once the next-token distribution has settled (low entropy)."""
    return lambda state, index: bool(state[index_of] <= tau)


def answer_stability(n_steps: int, index_of: int, max_steps: int) -> Decide:
    """Stop once the answer has held still for ``n_steps`` probes.

    ``steps_since_answer_change`` is stored normalised by ``traces.max_steps``, so the
    comparison is done in the same units rather than by un-normalising the feature.
    """
    threshold = n_steps / max(max_steps, 1)
    return lambda state, index: bool(state[index_of] >= threshold)


# --------------------------------------------------------------------------- #
# tuning
# --------------------------------------------------------------------------- #
def tune_threshold(
    traces: list[Trace],
    build: Callable[[float], Decide],
    candidates,
    score: Callable[[dict], float],
    min_steps: int = 0,
    per_trace: bool = False,
) -> tuple[float, dict]:
    """Pick the threshold that scores best on ``traces`` (the validation split).

    Baselines get the same tuning budget the DQN gets, otherwise beating them proves
    nothing. Returns the chosen value and its validation metrics.

    Set ``per_trace`` when ``build`` returns a factory taking a trace rather than a
    plain decision function - ``fixed_budget`` and ``random_stop`` both do.
    """
    from ..rl.rollout import evaluate

    best: tuple[float, float, dict] | None = None
    for value in candidates:
        metrics = evaluate(traces, build(value), min_steps, per_trace=per_trace)
        s = score(metrics)
        if best is None or s > best[0]:
            best = (s, value, metrics)
    if best is None:
        raise ValueError("no candidate thresholds supplied")
    return best[1], best[2]
