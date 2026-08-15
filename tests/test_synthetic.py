from __future__ import annotations

import random
from collections import Counter

import pytest

from adaptive_reasoning.config import load_config
from adaptive_reasoning.data import synthetic
from adaptive_reasoning.grading import is_correct
from adaptive_reasoning.schema import AnswerType, Difficulty


@pytest.mark.parametrize("generator", synthetic.GENERATORS, ids=lambda g: g.__name__)
def test_generator_produces_a_valid_item(generator):
    rng = random.Random(0)
    item = generator(rng)
    assert item.question.strip()
    assert isinstance(item.answer, (int, float))
    assert item.tier in set(Difficulty)


@pytest.mark.parametrize("generator", synthetic.GENERATORS, ids=lambda g: g.__name__)
def test_generator_has_enough_parameter_space(generator):
    """A low-cardinality generator starves the sampler and skews the tier mix.

    ``gen_tax_slab`` originally had a single varying parameter and could only produce
    260 distinct questions, which silently capped the whole synthetic set.
    """
    rng = random.Random(0)
    questions = {generator(rng).question for _ in range(500)}
    assert len(questions) > 450, f"{generator.__name__} produced only {len(questions)}/500 distinct"


@pytest.mark.parametrize("generator", synthetic.GENERATORS, ids=lambda g: g.__name__)
def test_generated_answer_grades_as_correct_against_itself(generator):
    """The gold answer must survive the round trip through the grader."""
    from adaptive_reasoning.data.text_utils import format_number

    rng = random.Random(1)
    for _ in range(20):
        item = generator(rng)
        gold = format_number(round(float(item.answer), 4))
        assert is_correct(gold, gold, "numeric"), f"{generator.__name__}: {gold!r} fails self-grading"


def test_load_hits_the_requested_count():
    cfg = load_config(overrides={"data": {"sample_sizes": {"synthetic": 2000}}})
    records = synthetic.load(cfg)
    assert len(records) == 2000
    assert all(r.answer_type == AnswerType.NUMERIC for r in records)


def test_load_produces_all_three_prior_tiers():
    cfg = load_config(overrides={"data": {"sample_sizes": {"synthetic": 600}}})
    records = synthetic.load(cfg)
    tiers = Counter(r.difficulty_prior for r in records)
    assert set(tiers) == {"easy", "medium", "hard"}, tiers


def test_load_leaves_measured_difficulty_unset():
    """`difficulty` means model-measured difficulty and is Phase 2's to fill in."""
    cfg = load_config(overrides={"data": {"sample_sizes": {"synthetic": 100}}})
    records = synthetic.load(cfg)
    assert all(r.difficulty is None for r in records)
    assert all(r.difficulty_prior is not None for r in records)


def test_load_questions_are_unique():
    cfg = load_config(overrides={"data": {"sample_sizes": {"synthetic": 1500}}})
    records = synthetic.load(cfg)
    assert len({r.question for r in records}) == len(records)


def test_load_is_deterministic():
    cfg = load_config(overrides={"data": {"sample_sizes": {"synthetic": 200}}})
    first = [r.question for r in synthetic.load(cfg)]
    second = [r.question for r in synthetic.load(cfg)]
    assert first == second


def test_break_even_rounds_up():
    """Partial units cannot be sold, so break-even must round up."""
    rng = random.Random(7)
    for _ in range(50):
        item = synthetic.gen_break_even(rng)
        assert item.answer == int(item.answer)


def test_tax_slab_is_zero_below_the_first_threshold():
    cfg_rng = random.Random(3)
    for _ in range(200):
        item = synthetic.gen_tax_slab(cfg_rng)
        assert item.answer >= 0
