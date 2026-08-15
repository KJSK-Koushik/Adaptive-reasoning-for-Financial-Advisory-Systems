"""Turning a reasoning trace into DQN state vectors.

One state vector per probe point. The feature list lives in ``rl.state_features`` and
the order here must match it exactly - :func:`build_states` asserts that, because a
silent reordering would train the network on scrambled inputs and be nearly impossible
to diagnose afterwards.

Every feature is computable **online**, from what a live reasoning stream exposes at
that moment. Nothing here peeks at the future or at the gold answer: if it did, the
policy would learn from information it will not have at inference time and the
evaluation numbers would be fiction.
"""

from __future__ import annotations

import re

import numpy as np

from ..config import Config
from ..logging_utils import get_logger

log = get_logger("rl.features")

#: Wrap-up language - the model signalling it is converging.
PROGRESS_CUES = re.compile(
    r"\b(therefore|thus|so the answer|in conclusion|hence|overall|final answer)\b",
    re.IGNORECASE,
)

#: Backtracking language - the model second-guessing itself. This is the textual
#: signature of the overthinking the policy is meant to cut short.
DOUBT_CUES = re.compile(
    r"\b(wait|hmm|actually|but|however|let me reconsider|on second thought|"
    r"alternatively|hold on)\b",
    re.IGNORECASE,
)


def _cue_density(text: str, pattern: re.Pattern) -> float:
    """Matches per 100 characters, clipped to [0, 1].

    A density rather than a raw count, so a long step is not automatically scored as
    more certain (or more doubtful) than a short one.
    """
    if not text:
        return 0.0
    return float(min(1.0, len(pattern.findall(text)) / max(len(text) / 100.0, 1.0)))


def build_states(
    steps: list[dict],
    difficulty_vector: np.ndarray,
    cfg: Config,
    budget: int,
) -> np.ndarray:
    """Build the ``[n_steps, state_dim]`` matrix for one trace.

    Args:
        steps: probe points for a single question, ordered by ``step_index``.
        difficulty_vector: length-3 vector over (easy, medium, hard). One-hot when
            using measured labels, a probability distribution when using the
            classifier - see ``rl.difficulty_source``.
        budget: the token cap the trace was generated under.
    """
    expected = cfg.rl.state_features
    n = len(steps)
    rows = np.zeros((n, len(expected)), dtype=np.float32)

    max_steps = max(cfg.traces.max_steps, 1)
    steps_since_change = 0
    previous_confidence = None
    previous_entropy = None

    for i, step in enumerate(steps):
        confidence = float(step["confidence"])
        entropy = float(step["entropy"])
        changed = bool(step["answer_changed"])

        if changed:
            steps_since_change = 0
        elif i > 0:
            steps_since_change += 1

        values = {
            "difficulty_easy": float(difficulty_vector[0]),
            "difficulty_medium": float(difficulty_vector[1]),
            "difficulty_hard": float(difficulty_vector[2]),
            "confidence": confidence,
            "min_token_confidence": float(step["min_token_confidence"]),
            "entropy": entropy,
            # Falling entropy means the model is settling; rising means it has opened a
            # new line of thought. Zero on the first step, where there is no history.
            "entropy_slope": 0.0 if previous_entropy is None else entropy - previous_entropy,
            "token_ratio": float(step["tokens_so_far"]) / max(budget, 1),
            "step_index_norm": min(1.0, i / max_steps),
            "delta_confidence": 0.0 if previous_confidence is None
            else confidence - previous_confidence,
            # 1.0 when the answer has just moved, falling toward 0 as it holds steady.
            "answer_stability": 0.0 if changed else min(1.0, steps_since_change / max_steps),
            "progress_cue": _cue_density(step.get("step_text", ""), PROGRESS_CUES),
            "doubt_cue": _cue_density(step.get("step_text", ""), DOUBT_CUES),
            "steps_since_answer_change": min(1.0, steps_since_change / max_steps),
        }

        missing = set(expected) - set(values)
        if missing:
            raise ValueError(
                f"rl.state_features names features this module does not build: "
                f"{sorted(missing)}"
            )

        rows[i] = [values[name] for name in expected]
        previous_confidence = confidence
        previous_entropy = entropy

    return rows


def difficulty_one_hot(difficulty: str | None) -> np.ndarray:
    """Measured label as a one-hot vector. Unknown difficulty spreads evenly."""
    order = ["easy", "medium", "hard"]
    vector = np.zeros(3, dtype=np.float32)
    if difficulty in order:
        vector[order.index(difficulty)] = 1.0
    else:
        vector[:] = 1.0 / 3.0
    return vector
