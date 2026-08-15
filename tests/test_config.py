from __future__ import annotations

import pytest
from pydantic import ValidationError

from adaptive_reasoning.config import Config, _deep_merge, load_config


def test_default_config_loads(cfg: Config):
    assert cfg.project.name
    assert cfg.project.seed == 42


def test_state_dim_matches_feature_list(cfg: Config):
    assert cfg.rl.state_dim == len(cfg.rl.state_features)
    assert cfg.rl.state_dim > 0
    # Difficulty must be part of the state - it is the core of the contribution.
    assert {"difficulty_easy", "difficulty_medium", "difficulty_hard"} <= set(
        cfg.rl.state_features
    )


def test_token_cost_is_difficulty_aware(cfg: Config):
    """Easy questions must be penalised most heavily for spending tokens."""
    tc = cfg.rl.reward.token_cost
    assert tc.easy > tc.medium > tc.hard
    assert tc.for_difficulty("easy") == tc.easy


def test_splits_must_sum_to_one():
    with pytest.raises(ValidationError):
        load_config(overrides={"data": {"splits": {"train": 0.9, "val": 0.2, "test": 0.1}}})


def test_difficulty_thresholds_must_be_ordered():
    with pytest.raises(ValidationError):
        load_config(overrides={"difficulty": {"easy_min_pass_rate": 0.2,
                                              "hard_max_pass_rate": 0.8}})


def test_unknown_key_is_rejected():
    """A YAML typo should fail loudly, not silently do nothing."""
    with pytest.raises(ValidationError):
        load_config(overrides={"project": {"speed": 42}})


def test_overrides_apply(cfg: Config):
    modified = load_config(overrides={"rl": {"dqn": {"gamma": 0.5}}})
    assert modified.rl.dqn.gamma == 0.5
    assert cfg.rl.dqn.gamma != 0.5, "override leaked into the base config"


def test_deep_merge_does_not_mutate_inputs():
    base = {"a": {"b": 1, "c": 2}}
    override = {"a": {"b": 9}}
    merged = _deep_merge(base, override)
    assert merged == {"a": {"b": 9, "c": 2}}
    assert base == {"a": {"b": 1, "c": 2}}
