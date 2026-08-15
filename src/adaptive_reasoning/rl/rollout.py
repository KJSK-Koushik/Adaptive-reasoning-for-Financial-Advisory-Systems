"""Replaying a stopping policy over recorded traces.

Any policy - DQN, behaviour cloning, a fixed threshold, the oracle - is evaluated the
same way: walk the trace step by step, ask the policy whether to stop, and read off
what actually happened at the step where it did. Because the trace recorded the probe
answer at every boundary, this is an exact simulation, not an approximation.

This module is the shared measuring instrument for Phases 5 and 6. Every reported
number goes through it, so the comparison between policies is genuinely like for like.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

#: A policy: given the state vector and the step index, should we stop here?
Decide = Callable[[np.ndarray, int], bool]


@dataclass
class Trace:
    """One question's states and outcomes, ready to replay."""

    question_id: str
    states: np.ndarray          # [n_steps, state_dim]
    tokens: np.ndarray          # tokens_so_far at each step
    correct: np.ndarray         # would stopping here have been right?
    difficulty: str | None

    @property
    def n_steps(self) -> int:
        return len(self.tokens)

    @property
    def full_tokens(self) -> int:
        """Cost of never stopping early - the full-reasoning baseline."""
        return int(self.tokens[-1])

    @property
    def full_correct(self) -> bool:
        return bool(self.correct[-1])

    @property
    def earliest_correct(self) -> int | None:
        hits = np.flatnonzero(self.correct)
        return int(hits[0]) if hits.size else None


@dataclass
class Rollout:
    question_id: str
    stop_step: int
    tokens: int
    correct: bool
    difficulty: str | None


def rollout(trace: Trace, decide: Decide, min_steps: int = 0) -> Rollout:
    """Replay one trace under ``decide``.

    ``min_steps`` forces the policy to look at least that far before it may stop; the
    live system uses the same guard so it can never answer with no reasoning at all.
    """
    stop_at = trace.n_steps - 1          # forced stop when the model finishes
    for index in range(trace.n_steps):
        if index < min_steps:
            continue
        if decide(trace.states[index], index):
            stop_at = index
            break

    return Rollout(
        question_id=trace.question_id,
        stop_step=stop_at,
        tokens=int(trace.tokens[stop_at]),
        correct=bool(trace.correct[stop_at]),
        difficulty=trace.difficulty,
    )


def evaluate(
    traces: list[Trace],
    decide: Decide | Callable[[Trace], Decide],
    min_steps: int = 0,
    per_trace: bool = False,
) -> dict:
    """Roll a policy over every trace and summarise.

    ``token_reduction_pct`` is measured against full reasoning on the *same* traces,
    so it is a paired comparison rather than two independent averages.

    With ``per_trace=True``, ``decide`` is a *factory* taking a trace and returning its
    decision function. That lets a policy score a whole trace in one vectorised call -
    for the tree-based models the per-state Python call overhead dominates everything
    else.
    """
    if not traces:
        return {}

    rollouts = [
        rollout(t, decide(t) if per_trace else decide, min_steps) for t in traces
    ]

    tokens = np.array([r.tokens for r in rollouts], dtype=float)
    correct = np.array([r.correct for r in rollouts], dtype=float)
    steps = np.array([r.stop_step for r in rollouts], dtype=float)

    full_tokens = np.array([t.full_tokens for t in traces], dtype=float)
    full_correct = np.array([t.full_correct for t in traces], dtype=float)

    result = {
        "n": len(traces),
        "accuracy": round(float(correct.mean()), 4),
        "mean_tokens": round(float(tokens.mean()), 1),
        "mean_stop_step": round(float(steps.mean()), 2),
        "token_reduction_pct": round(
            float(100 * (1 - tokens.sum() / max(full_tokens.sum(), 1))), 2
        ),
        "accuracy_delta_vs_full": round(float(correct.mean() - full_correct.mean()), 4),
        "stopped_early_pct": round(
            float(100 * np.mean(steps < np.array([t.n_steps - 1 for t in traces]))), 1
        ),
    }

    # Per-tier behaviour is the whole point of a difficulty-aware policy: it should
    # stop sooner on easy questions than on hard ones. Reported so that claim is
    # checkable rather than asserted.
    for tier in ("easy", "medium", "hard"):
        mask = np.array([r.difficulty == tier for r in rollouts])
        if mask.any():
            result[f"{tier}_mean_tokens"] = round(float(tokens[mask].mean()), 1)
            result[f"{tier}_accuracy"] = round(float(correct[mask].mean()), 4)
            result[f"{tier}_n"] = int(mask.sum())

    return result


# --------------------------------------------------------------------------- #
# reference policies
# --------------------------------------------------------------------------- #
def always_continue() -> Decide:
    """Full reasoning: never stop early. The cost and accuracy ceiling."""
    return lambda state, index: False


def oracle(trace: Trace) -> Decide:
    """Stop at the earliest correct step, or immediately if never correct.

    Cheating by construction - it reads the gold answer - and that is the point: it is
    the best any stopping policy could possibly do on this trace, which turns "we saved
    45%" into "we captured 70% of what was available".
    """
    target = trace.earliest_correct
    stop_at = 0 if target is None else target
    return lambda state, index: index >= stop_at


def load_traces(frame, state_dim: int, split: str | None = None) -> list[Trace]:
    """Rebuild replayable traces from the Phase 4 transition table."""
    from .dataset import ACTION_STOP

    rows = frame[frame.action == ACTION_STOP]
    if split:
        rows = rows[rows.split == split]

    traces: list[Trace] = []
    for question_id, group in rows.groupby("question_id", sort=False):
        group = group.sort_values("step_index")
        states = np.vstack([np.asarray(s, dtype=np.float32) for s in group.state])
        if states.shape[1] != state_dim:
            raise ValueError(
                f"{question_id}: state dimension {states.shape[1]} does not match the "
                f"configured {state_dim}; rebuild with scripts/run_phase4.py"
            )
        difficulty = group.difficulty.iloc[0]
        traces.append(
            Trace(
                question_id=str(question_id),
                states=states,
                tokens=group.tokens_so_far.to_numpy(),
                correct=group.probe_correct.to_numpy().astype(bool),
                difficulty=None if difficulty is None or difficulty != difficulty
                else str(difficulty),
            )
        )
    return traces
