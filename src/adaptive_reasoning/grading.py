"""Answer grading.

The RL reward is built directly on top of ``is_correct``. If grading is sloppy,
the DQN learns from noise, so this module is deliberately strict and well tested.

Two modes:

* **numeric** - pull a number out of free text and compare within a relative
  tolerance. Handles currency symbols, thousands separators, percentages,
  accounting negatives ``(1,234)``, scale words, and trailing prose.
* **categorical** - normalise and match against the gold label or a synonym.
"""

from __future__ import annotations

import re

# Trailing scale words, applied as multipliers.
_SCALES: dict[str, float] = {
    "thousand": 1e3,
    "k": 1e3,
    "lakh": 1e5,
    "lakhs": 1e5,
    "million": 1e6,
    "m": 1e6,
    "mn": 1e6,
    "crore": 1e7,
    "crores": 1e7,
    "billion": 1e9,
    "bn": 1e9,
    "trillion": 1e12,
}

# Common categorical answers and the variants a model might produce.
_SYNONYMS: dict[str, str] = {
    "yes": "yes", "true": "yes", "fraudulent": "yes", "fraud": "yes", "positive": "positive",
    "no": "no", "false": "no", "legitimate": "no", "not fraud": "no", "genuine": "no",
    "negative": "negative", "neutral": "neutral",
    "good": "good", "good risk": "good", "low risk": "good", "approve": "good",
    "bad": "bad", "bad risk": "bad", "high risk": "bad", "reject": "bad", "deny": "bad",
}

_NUMBER_RE = re.compile(
    r"""
    (?P<paren>\()?                 # accounting negative
    \s*
    (?P<sign>[-+])?
    \s*
    [$£€₹]?\s*                     # currency
    (?P<num>\d{1,3}(?:,\d{2,3})+(?:\.\d+)? | \d*\.\d+ | \d+)
    \s*
    (?P<pct>%)?
    \s*
    (?P<scale>thousand|lakhs?|crores?|million|billion|trillion|bn|mn|[km])?\b
    (?(paren)\s*\))
    """,
    re.IGNORECASE | re.VERBOSE,
)


def extract_number(text: str, prefer: str = "last") -> float | None:
    """Extract a single number from free text.

    Args:
        text: the string to search.
        prefer: ``"last"`` takes the final number, which is usually the answer in a
            sentence like "...so the growth is 12.4%". ``"first"`` takes the earliest.

    Returns:
        The parsed value, or ``None`` if no number is present. Percentages are
        returned as written (``12.4%`` -> ``12.4``), matching FinQA convention.
    """
    if not text:
        return None

    matches = list(_NUMBER_RE.finditer(text))
    if not matches:
        return None

    match = matches[-1] if prefer == "last" else matches[0]

    raw = match.group("num").replace(",", "")
    try:
        value = float(raw)
    except ValueError:
        return None

    if match.group("scale"):
        value *= _SCALES[match.group("scale").lower()]

    if match.group("sign") == "-" or match.group("paren"):
        value = -value

    return value


def normalise_categorical(text: str) -> str:
    """Lowercase, strip punctuation and map through the synonym table."""
    if not text:
        return ""
    cleaned = re.sub(r"[^\w\s]", " ", text.lower())
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    if cleaned in _SYNONYMS:
        return _SYNONYMS[cleaned]

    # The model often answers in a sentence. Look for a known label inside it,
    # preferring the longest match so "not fraud" beats "fraud".
    for phrase in sorted(_SYNONYMS, key=len, reverse=True):
        if re.search(rf"\b{re.escape(phrase)}\b", cleaned):
            return _SYNONYMS[phrase]

    return cleaned


def numeric_match(predicted: str, gold: str, tolerance: float = 0.01) -> bool:
    """Compare two numeric answers within a *relative* tolerance.

    Falls back to absolute tolerance when the gold value is zero. Also accepts a
    prediction that is off by exactly 100x, which is the classic
    percent-vs-fraction mismatch (``0.124`` vs ``12.4``).
    """
    p = extract_number(predicted)
    g = extract_number(gold)
    if p is None or g is None:
        return False

    if g == 0:
        return abs(p) <= tolerance

    if abs(p - g) / abs(g) <= tolerance:
        return True

    # Percent / fraction confusion.
    return any(abs(p * factor - g) / abs(g) <= tolerance for factor in (100.0, 0.01))


def categorical_match(predicted: str, gold: str) -> bool:
    p = normalise_categorical(predicted)
    g = normalise_categorical(gold)
    if not p or not g:
        return False
    return p == g


def is_correct(predicted: str, gold: str, answer_type: str, tolerance: float = 0.01) -> bool:
    """Grade one answer. ``answer_type`` is ``"numeric"`` or ``"categorical"``."""
    if predicted is None or gold is None:
        return False
    if answer_type == "numeric":
        return numeric_match(predicted, gold, tolerance)
    if answer_type == "categorical":
        return categorical_match(predicted, gold)
    raise ValueError(f"unknown answer_type {answer_type!r}")
