"""Difficulty-classifier tests.

The sentence encoder is stubbed with a deterministic hash-based embedding so these run
offline in milliseconds. What is under test is the feature assembly, the training
contract and the persistence round trip - not MiniLM's quality.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from adaptive_reasoning.config import load_config
from adaptive_reasoning.difficulty.classifier import (
    CLASSES,
    SURFACE_FEATURES,
    DifficultyClassifier,
    surface_features,
)
from adaptive_reasoning.schema import Difficulty

_DIM = 32


def _fake_embed(questions, batch_size=64):
    """Deterministic pseudo-embedding: same text always maps to the same vector."""
    out = np.zeros((len(questions), _DIM), dtype=np.float32)
    for i, text in enumerate(questions):
        digest = hashlib.sha256(text.encode()).digest()
        out[i] = np.frombuffer(digest[:_DIM], dtype=np.uint8).astype(np.float32) / 255.0
    return out


@pytest.fixture
def model(monkeypatch):
    cfg = load_config()
    clf = DifficultyClassifier(cfg)
    monkeypatch.setattr(clf, "embed", _fake_embed)
    return clf


def _dataset(n=180):
    """Separable toy data: label is recoverable from the text, so learning must work."""
    questions, contexts, labels = [], [], []
    for i in range(n):
        tier = CLASSES[i % 3]
        if tier == Difficulty.EASY:
            questions.append(f"What is the sentiment of statement {i}?")
        elif tier == Difficulty.MEDIUM:
            questions.append(f"What was the percentage change between 2018 and 2019 in row {i}?")
        else:
            questions.append(
                f"What is the net present value over 5 years, and then how much does "
                f"that change represent relative to the 2015 base in case {i}?"
            )
        contexts.append("| year | value |\n" * (CLASSES.index(tier) + 1))
        labels.append(tier)
    return questions, contexts, labels


# --------------------------------------------------------------------------- #
# surface features
# --------------------------------------------------------------------------- #
def test_surface_features_length_matches_the_declared_names():
    assert len(surface_features("q?", "ctx")) == len(SURFACE_FEATURES)


def test_surface_features_count_numbers():
    values = dict(zip(SURFACE_FEATURES, surface_features("What is 5% of 1,200?", "3 and 4"),
                      strict=True))
    assert values["n_numbers_question"] == 2
    assert values["n_numbers_context"] == 2
    assert values["has_percent"] == 1.0


def test_surface_features_detect_multi_step_cues():
    plain = dict(zip(SURFACE_FEATURES, surface_features("What was revenue?", ""), strict=True))
    multi = dict(zip(SURFACE_FEATURES,
                     surface_features("What was revenue, and how much does that change represent?", ""),
                     strict=True))
    assert plain["has_multi_step_cue"] == 0.0
    assert multi["has_multi_step_cue"] == 1.0


def test_surface_features_detect_comparison():
    values = dict(zip(SURFACE_FEATURES,
                      surface_features("Which year was higher than 2018?", ""), strict=True))
    assert values["has_comparison"] == 1.0


def test_surface_features_count_table_pipes():
    values = dict(zip(SURFACE_FEATURES, surface_features("q", "a | b | c"), strict=True))
    assert values["n_table_pipes"] == 2


# --------------------------------------------------------------------------- #
# training
# --------------------------------------------------------------------------- #
def test_featurise_shape(model):
    x = model.featurise(["a", "b"], ["", ""])
    assert x.shape == (2, _DIM + len(SURFACE_FEATURES))


def test_fit_learns_separable_data(model):
    questions, contexts, labels = _dataset()
    report = model.fit(questions, contexts, labels)
    assert report.accuracy > 0.9
    assert report.n_train == len(labels)


def test_report_includes_the_majority_baseline(model):
    """A classifier that cannot beat 'always predict the commonest tier' is useless."""
    questions, contexts, labels = _dataset()
    report = model.fit(questions, contexts, labels)
    assert 0.3 <= report.baseline_majority <= 0.4     # three balanced classes
    assert report.accuracy > report.baseline_majority


def test_fit_rejects_single_class_data(model):
    with pytest.raises(ValueError, match="at least two difficulty classes"):
        model.fit(["a", "b", "c"], ["", "", ""], [Difficulty.EASY] * 3)


def test_confusion_matrix_is_three_by_three(model):
    questions, contexts, labels = _dataset()
    report = model.fit(questions, contexts, labels)
    assert len(report.confusion) == 3
    assert all(len(row) == 3 for row in report.confusion)


def test_per_class_metrics_cover_every_tier(model):
    questions, contexts, labels = _dataset()
    report = model.fit(questions, contexts, labels)
    assert set(report.per_class) == {"easy", "medium", "hard"}


def test_eval_split_is_used_when_given(model):
    questions, contexts, labels = _dataset(150)
    report = model.fit(
        questions[:100], contexts[:100], labels[:100],
        eval_split=(questions[100:], contexts[100:], labels[100:]),
    )
    assert report.n_train == 100
    assert report.n_eval == 50


# --------------------------------------------------------------------------- #
# inference
# --------------------------------------------------------------------------- #
def test_predict_returns_valid_tiers(model):
    questions, contexts, labels = _dataset()
    model.fit(questions, contexts, labels)
    predictions = model.predict(questions[:10], contexts[:10])
    assert len(predictions) == 10
    assert all(p in CLASSES for p in predictions)


def test_predict_works_without_context(model):
    """A live user query arrives with no context attached."""
    questions, contexts, labels = _dataset()
    model.fit(questions, contexts, labels)
    assert len(model.predict(["What is 5% of 200?"])) == 1


def test_predict_proba_rows_sum_to_one(model):
    questions, contexts, labels = _dataset()
    model.fit(questions, contexts, labels)
    probabilities = model.predict_proba(questions[:5], contexts[:5])
    assert probabilities.shape == (5, 3)
    assert np.allclose(probabilities.sum(axis=1), 1.0)


def test_predict_before_training_raises(model):
    with pytest.raises(RuntimeError, match="not trained"):
        model.predict(["anything"])


# --------------------------------------------------------------------------- #
# persistence
# --------------------------------------------------------------------------- #
def test_save_and_load_round_trip(model, monkeypatch, tmp_path):
    questions, contexts, labels = _dataset()
    model.fit(questions, contexts, labels)
    before = model.predict(questions[:20], contexts[:20])

    path = tmp_path / "clf.joblib"
    model.save(path)

    restored = DifficultyClassifier.load(load_config(), path)
    monkeypatch.setattr(restored, "embed", _fake_embed)
    assert restored.predict(questions[:20], contexts[:20]) == before
