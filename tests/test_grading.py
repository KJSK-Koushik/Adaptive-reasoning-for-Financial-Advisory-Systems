from __future__ import annotations

import pytest

from adaptive_reasoning.grading import (
    categorical_match,
    extract_number,
    extract_numbers,
    is_correct,
    normalise_categorical,
    numeric_match,
)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("1234", 1234.0),
        ("1,234.56", 1234.56),
        ("$1,234.56", 1234.56),
        ("₹45,000", 45000.0),
        ("12.4%", 12.4),
        ("-5.2", -5.2),
        ("(1,234)", -1234.0),          # accounting negative
        ("2.5 million", 2_500_000.0),
        ("3 crore", 30_000_000.0),
        ("1.5 lakh", 150_000.0),
        (".75", 0.75),
        ("no digits here", None),
        ("", None),
    ],
)
def test_extract_number(text, expected):
    assert extract_number(text) == expected


def test_extract_number_prefers_last_by_default():
    text = "Revenue rose from 100 to 250, so the answer is 150"
    assert extract_number(text) == 150.0
    assert extract_number(text, prefer="first") == 100.0


def test_numeric_match_within_tolerance():
    assert numeric_match("100.5", "100", tolerance=0.01)
    assert not numeric_match("102", "100", tolerance=0.01)


def test_numeric_match_handles_percent_fraction_confusion():
    assert numeric_match("0.124", "12.4")
    assert numeric_match("12.4", "0.124")


def test_numeric_match_zero_gold():
    assert numeric_match("0.001", "0", tolerance=0.01)
    assert not numeric_match("5", "0", tolerance=0.01)


def test_numeric_match_requires_a_number_on_both_sides():
    assert not numeric_match("I am not sure", "42")
    assert not numeric_match("42", "unknown")


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Yes", "yes"),
        ("FRAUDULENT", "yes"),
        ("legitimate", "no"),
        ("high risk", "bad"),
        ("Approve", "good"),
        ("Neutral.", "neutral"),
    ],
)
def test_normalise_categorical(text, expected):
    assert normalise_categorical(text) == expected


def test_categorical_match_inside_a_sentence():
    assert categorical_match("Based on the pattern, this is fraudulent.", "yes")
    assert categorical_match("The applicant is a good risk", "good")


def test_negation_is_not_mistaken_for_the_positive_label():
    """'not fraud' must beat the shorter 'fraud' substring."""
    assert normalise_categorical("not fraud") == "no"
    assert categorical_match("not fraud", "no")
    assert not categorical_match("not fraud", "yes")


def test_is_correct_dispatch():
    assert is_correct("12.4%", "12.4", "numeric")
    assert is_correct("yes", "yes", "categorical")
    with pytest.raises(ValueError, match="unknown answer_type"):
        is_correct("a", "b", "freeform")


def test_is_correct_handles_none():
    assert not is_correct(None, "12", "numeric")
    assert not is_correct("12", None, "numeric")


# --------------------------------------------------------------------------- #
# scale words - the unit is often stated in the question, not the answer
# --------------------------------------------------------------------------- #
class TestScaleWords:
    """FinQA asks "...in millions?" and stores ``407``. A model answering
    "407 million dollars" is right, and scoring it wrong cost 3.3% of numeric probes.
    """

    @pytest.mark.parametrize("predicted", [
        "407 million dollars",
        "$407 million",
        "407 million",
        "the answer is 407 million",
        "407.0 million",
    ])
    def test_explicit_unit_matches_a_bare_gold_value(self, predicted):
        assert numeric_match(predicted, "407")

    def test_explicit_unit_still_matches_a_fully_written_gold_value(self):
        assert numeric_match("407 million", "407000000")

    def test_both_readings_are_offered(self):
        assert extract_numbers("407 million") == [407_000_000.0, 407.0]

    def test_a_bare_number_has_a_single_reading(self):
        assert extract_numbers("407") == [407.0]

    def test_the_sign_survives_both_readings(self):
        assert extract_numbers("-2.5 billion") == [-2.5e9, -2.5]
        assert numeric_match("(407) million", "-407")

    def test_leniency_does_not_extend_to_a_different_value(self):
        """The allowance is about units, not about being roughly in the area."""
        assert not numeric_match("500 million", "407")
        assert not numeric_match("407 million", "450000000")

    def test_gold_may_carry_the_scale_word_instead(self):
        assert numeric_match("1234", "1,234 thousand")

    def test_empty_input_has_no_readings(self):
        assert extract_numbers("") == []
        assert extract_numbers("no digits here") == []

    def test_extract_number_still_returns_the_scaled_reading(self):
        """The single-value helper is unchanged - adapters rely on it."""
        assert extract_number("407 million") == 407_000_000.0
