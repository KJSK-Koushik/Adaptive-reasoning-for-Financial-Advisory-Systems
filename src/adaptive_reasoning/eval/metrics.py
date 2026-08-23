"""Uncertainty and significance for the Phase 6 comparison.

A difference of a few accuracy points on 599 questions can easily be noise, so every
headline comparison is reported with a confidence interval and a significance test
rather than as a bare difference.

All tests here are *paired*: each policy is measured on the same questions, replayed
over the same traces, so the pairing is exact and a paired test is both valid and much
more sensitive than treating the two as independent samples.
"""

from __future__ import annotations

import math

import numpy as np


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a single accuracy.

    Preferred over the normal approximation because accuracy here sits near 0.3 with a
    few hundred samples, where the normal interval misbehaves and can leave [0, 1].
    """
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def paired_bootstrap(
    correct_a: np.ndarray,
    correct_b: np.ndarray,
    n_samples: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> dict:
    """Bootstrap the accuracy difference (a - b) by resampling questions.

    Questions are resampled together for both policies, which preserves the pairing:
    a question that is hard for one policy stays hard for the other in every resample.
    """
    a = np.asarray(correct_a, dtype=float)
    b = np.asarray(correct_b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"paired test needs equal lengths, got {a.shape} and {b.shape}")

    n = len(a)
    if n == 0:
        return {"difference": 0.0, "ci_low": 0.0, "ci_high": 0.0,
                "n_samples": 0, "p_sign_flip": 1.0}

    observed = float(a.mean() - b.mean())

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_samples, n))
    diffs = a[idx].mean(axis=1) - b[idx].mean(axis=1)

    return {
        "difference": round(observed, 4),
        "ci_low": round(float(np.quantile(diffs, alpha / 2)), 4),
        "ci_high": round(float(np.quantile(diffs, 1 - alpha / 2)), 4),
        "n_samples": n_samples,
        # The fraction of resamples where the sign flips - a direct read on how safe
        # the claim "a beats b" is.
        "p_sign_flip": round(float((diffs <= 0).mean() if observed > 0
                                   else (diffs >= 0).mean()), 4),
    }


def mcnemar(correct_a: np.ndarray, correct_b: np.ndarray) -> dict:
    """Exact McNemar test on paired binary outcomes.

    Only the questions where the two policies disagree carry information; questions
    both get right, or both get wrong, tell us nothing about which is better.
    """
    a = np.asarray(correct_a, dtype=bool)
    b = np.asarray(correct_b, dtype=bool)
    only_a = int(np.sum(a & ~b))
    only_b = int(np.sum(~a & b))
    n = only_a + only_b

    if n == 0:
        return {"only_a": 0, "only_b": 0, "p_value": 1.0}

    # Two-sided exact binomial test against p = 0.5.
    k = min(only_a, only_b)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return {
        "only_a": only_a,
        "only_b": only_b,
        "p_value": round(min(1.0, 2 * tail), 5),
    }


def token_cost_model(mean_tokens: float, tokens_per_second: float = 91.5) -> dict:
    """Translate a token count into the numbers a deployment actually cares about.

    ``tokens_per_second`` defaults to the throughput measured by the Phase 3 pilot on a
    Kaggle T4, so latency here is grounded in a real measurement rather than a
    datasheet figure. Energy uses the T4's 70 W board power.
    """
    seconds = mean_tokens / max(tokens_per_second, 1e-9)
    return {
        "mean_tokens": round(mean_tokens, 1),
        "latency_seconds": round(seconds, 3),
        "energy_joules": round(seconds * 70.0, 1),
    }
