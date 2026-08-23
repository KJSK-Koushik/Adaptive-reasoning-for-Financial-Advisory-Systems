"""The transition builder and the difficulty switch.

These were the least-covered lines in the project and they carry a lot: the builder
produced NaN rewards once and trained a DQN on them without error, and the difficulty
switch is what the Phase 9 ablations turn.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from adaptive_reasoning.config import load_config
from adaptive_reasoning.rl.dataset import _difficulty_vectors


@pytest.fixture
def meta():
    """The columns build() reads out of unified.parquet."""
    return pd.DataFrame(
        {
            "difficulty": ["easy", "medium", "hard", None],
            "split": ["train", "train", "test", "val"],
        },
        index=pd.Index(["q0", "q1", "q2", "q3"], name="id"),
    )


@pytest.fixture
def steps():
    return pd.DataFrame({
        "question_id": ["q0", "q0", "q1", "q2", "q3"],
        "step_index": [0, 1, 0, 0, 0],
    })


def _cfg(source):
    return load_config(overrides={"rl": {"difficulty_source": source}})


# --------------------------------------------------------------------------- #
# the difficulty switch - one vector per question, always a distribution
# --------------------------------------------------------------------------- #
def test_none_gives_every_question_the_same_flat_vector(meta, steps):
    vectors = _difficulty_vectors(_cfg("none"), meta, steps)
    assert set(vectors) == {"q0", "q1", "q2", "q3"}
    for vector in vectors.values():
        assert vector == pytest.approx([1 / 3, 1 / 3, 1 / 3])


def test_true_gives_one_hot_measured_labels(meta, steps):
    vectors = _difficulty_vectors(_cfg("true"), meta, steps)
    assert vectors["q0"] == pytest.approx([1.0, 0.0, 0.0])
    assert vectors["q1"] == pytest.approx([0.0, 1.0, 0.0])
    assert vectors["q2"] == pytest.approx([0.0, 0.0, 1.0])


def test_an_unlabelled_question_spreads_evenly_rather_than_guessing(meta, steps):
    vectors = _difficulty_vectors(_cfg("true"), meta, steps)
    assert vectors["q3"] == pytest.approx([1 / 3, 1 / 3, 1 / 3])


@pytest.mark.parametrize("source", ["none", "true"])
def test_vectors_are_distributions(meta, steps, source):
    for vector in _difficulty_vectors(_cfg(source), meta, steps).values():
        assert np.isclose(np.sum(vector), 1.0)
        assert np.all(np.asarray(vector) >= 0)


def test_true_and_none_are_not_the_same_thing(meta, steps):
    """A regression guard: an ablation that silently equals its control proves nothing."""
    true = _difficulty_vectors(_cfg("true"), meta, steps)
    none = _difficulty_vectors(_cfg("none"), meta, steps)
    assert not all(np.allclose(true[q], none[q]) for q in ("q0", "q1", "q2"))


def test_every_question_in_the_steps_frame_gets_a_vector(meta, steps):
    """A missing vector would fall back to a default and quietly change the state."""
    for source in ("none", "true"):
        vectors = _difficulty_vectors(_cfg(source), meta, steps)
        assert set(vectors) == set(steps.question_id.astype(str))


def test_an_unknown_source_is_rejected(meta, steps):
    """The config rejects it before any vector is built, which is the right place."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        _cfg("astrology")
