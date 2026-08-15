from __future__ import annotations

import pytest

from adaptive_reasoning.config import load_config, parse_overrides


def test_parses_a_scalar():
    assert parse_overrides(["llm.batch_size=32"]) == {"llm": {"batch_size": 32}}


def test_infers_types_via_yaml():
    parsed = parse_overrides([
        "a.i=32", "a.f=0.5", "a.b=true", "a.n=null", "a.s=hello",
    ])["a"]
    assert parsed == {"i": 32, "f": 0.5, "b": True, "n": None, "s": "hello"}


def test_merges_multiple_keys_in_one_section():
    parsed = parse_overrides(["llm.batch_size=32", "llm.max_new_tokens=768"])
    assert parsed == {"llm": {"batch_size": 32, "max_new_tokens": 768}}


def test_handles_deep_paths():
    parsed = parse_overrides(["rl.reward.token_cost.easy=0.9"])
    assert parsed == {"rl": {"reward": {"token_cost": {"easy": 0.9}}}}


def test_empty_input():
    assert parse_overrides(None) == {}
    assert parse_overrides([]) == {}


def test_rejects_malformed_assignment():
    with pytest.raises(ValueError, match="key.path=value"):
        parse_overrides(["llm.batch_size"])


def test_overrides_reach_the_loaded_config():
    cfg = load_config(overrides=parse_overrides([
        "llm.batch_size=48", "traces.n_questions=4000", "difficulty.k_samples=3",
    ]))
    assert cfg.llm.batch_size == 48
    assert cfg.traces.n_questions == 4000
    assert cfg.difficulty.k_samples == 3


def test_overrides_are_still_validated():
    """A bad value must fail validation, not slip through as a string."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        load_config(overrides=parse_overrides(["difficulty.k_samples=0"]))
