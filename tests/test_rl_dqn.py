from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from adaptive_reasoning.config import load_config
from adaptive_reasoning.rl.bc import BehaviourCloning, evaluate_bc, oracle_labels
from adaptive_reasoning.rl.dataset import ACTION_CONTINUE, ACTION_STOP, recompute_rewards
from adaptive_reasoning.rl.dqn import (
    OfflineBuffer,
    QNetwork,
    action_balance,
    greedy_policy,
    selection_score,
    train,
)
from adaptive_reasoning.rl.rollout import Trace

STATE_DIM = 14


@pytest.fixture
def cfg():
    return load_config()


def _trace(correct, difficulty="medium", qid="q1", seed=0) -> Trace:
    n = len(correct)
    rng = np.random.default_rng(seed)
    return Trace(
        question_id=qid,
        states=rng.random((n, STATE_DIM)).astype(np.float32),
        tokens=np.array([(i + 1) * 50 for i in range(n)]),
        correct=np.array(correct, dtype=bool),
        difficulty=difficulty,
    )


def _state(step: int, n_steps: int, rng) -> np.ndarray:
    """Random features, except dimension 0 which encodes progress through the trace.

    Without a feature that correlates with the reward there is nothing to learn, and a
    "did training work?" test would only ever be measuring noise.
    """
    state = rng.random(STATE_DIM).astype(np.float32)
    state[0] = step / max(n_steps - 1, 1)
    return state


def _frame(n_questions=40) -> pd.DataFrame:
    """A small synthetic transition table with a learnable structure."""
    rng = np.random.default_rng(0)
    rows = []
    for q in range(n_questions):
        n_steps = 4
        for step in range(n_steps):
            state = _state(step, n_steps, rng)
            nxt = _state(min(step + 1, n_steps - 1), n_steps, rng)
            rows.append({
                "question_id": f"q{q}", "step_index": step, "difficulty": "medium",
                "action": ACTION_STOP, "reward": 1.0 if step >= 2 else -1.0,
                "done": True, "state": state.tolist(),
                "next_state": np.zeros(STATE_DIM).tolist(),
                "probe_correct": step >= 2, "tokens_so_far": (step + 1) * 50,
                "tokens_in_next_step": 50 if step < n_steps - 1 else 0,
                "answer_changed": False, "split": "train",
            })
            rows.append({
                "question_id": f"q{q}", "step_index": step, "difficulty": "medium",
                "action": ACTION_CONTINUE, "reward": -0.05,
                "done": step == n_steps - 1, "state": state.tolist(),
                "next_state": nxt.tolist(),
                "probe_correct": step >= 2, "tokens_so_far": (step + 1) * 50,
                "tokens_in_next_step": 50 if step < n_steps - 1 else 0,
                "answer_changed": False, "split": "train",
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# network
# --------------------------------------------------------------------------- #
def test_network_shapes():
    net = QNetwork(STATE_DIM, [32, 32])
    out = net(torch.zeros(7, STATE_DIM))
    assert out.shape == (7, 2)


def test_network_parameter_count_is_small(cfg):
    """The controller must stay negligible beside the model it gates."""
    net = QNetwork(cfg.rl.state_dim, cfg.rl.dqn.hidden_sizes)
    total = sum(p.numel() for p in net.parameters())
    assert total < 50_000, f"{total} parameters is larger than intended"


def test_greedy_policy_reads_the_stop_action():
    net = QNetwork(STATE_DIM, [8])
    with torch.no_grad():
        net.net[-1].bias[:] = torch.tensor([-10.0, 10.0])   # STOP dominant
        net.net[-1].weight[:] = 0.0
    assert greedy_policy(net)(np.zeros(STATE_DIM, np.float32), 0) is True

    with torch.no_grad():
        net.net[-1].bias[:] = torch.tensor([10.0, -10.0])   # CONTINUE dominant
    assert greedy_policy(net)(np.zeros(STATE_DIM, np.float32), 0) is False


# --------------------------------------------------------------------------- #
# buffer
# --------------------------------------------------------------------------- #
def test_buffer_loads_the_frame():
    buffer = OfflineBuffer(_frame(5), STATE_DIM)
    assert buffer.n == 40
    assert buffer.states.shape == (40, STATE_DIM)


def test_buffer_rejects_a_dimension_mismatch():
    with pytest.raises(ValueError, match="state dimension"):
        OfflineBuffer(_frame(3), STATE_DIM + 1)


def test_buffer_sampling_is_reproducible():
    buffer = OfflineBuffer(_frame(10), STATE_DIM)
    a = buffer.sample(8, torch.Generator().manual_seed(1))
    b = buffer.sample(8, torch.Generator().manual_seed(1))
    assert torch.equal(a.states, b.states)


# --------------------------------------------------------------------------- #
# training
# --------------------------------------------------------------------------- #
def test_training_learns_the_structure(cfg):
    """Stopping is rewarded from step 2 on; the network should pick that up."""
    tuned = load_config(overrides={"rl": {"dqn": {"train_steps": 600, "eval_every": 10_000}}})
    net, _, _ = train(_frame(60), tuned)
    balance = action_balance(net, np.vstack([np.asarray(s, np.float32)
                                             for s in _frame(20).state]))
    assert 0 < balance["stop_pct"] < 100, "policy collapsed to a single action"


def test_training_is_deterministic(cfg):
    tuned = load_config(overrides={"rl": {"dqn": {"train_steps": 200, "eval_every": 10_000}}})
    a, _, _ = train(_frame(20), tuned)
    b, _, _ = train(_frame(20), tuned)
    for pa, pb in zip(a.parameters(), b.parameters(), strict=True):
        assert torch.allclose(pa, pb)


def test_weighted_selection_balances_both_objectives():
    """Accuracy alone picks full reasoning; savings alone picks instant wrong answers."""
    weighted = load_config(overrides={"rl": {"selection": {"objective": "weighted_score"}}})
    accurate_slow = {"accuracy": 0.42, "token_reduction_pct": 0.0}
    fast_wrong = {"accuracy": 0.10, "token_reduction_pct": 90.0}
    balanced = {"accuracy": 0.40, "token_reduction_pct": 50.0}
    scores = [selection_score(m, weighted) for m in (accurate_slow, fast_wrong, balanced)]
    assert scores[2] == max(scores)


def test_budget_selection_maximises_accuracy_above_the_budget():
    """With an explicit cost budget, the most accurate qualifying policy wins."""
    cfg = load_config(overrides={"rl": {"selection": {
        "objective": "accuracy_at_budget", "min_token_reduction": 45.0}}})
    below = {"accuracy": 0.99, "token_reduction_pct": 40.0}
    qualifying = {"accuracy": 0.37, "token_reduction_pct": 50.0}
    wasteful = {"accuracy": 0.30, "token_reduction_pct": 80.0}
    scores = [selection_score(m, cfg) for m in (below, qualifying, wasteful)]
    assert scores[1] == max(scores), "should pick the accurate one that meets the budget"
    assert scores[0] < 0, "a policy under budget must never outrank one that meets it"


def test_budget_selection_ranks_near_misses_higher():
    """If nothing meets the budget, prefer the closest - not an arbitrary one."""
    cfg = load_config(overrides={"rl": {"selection": {
        "objective": "accuracy_at_budget", "min_token_reduction": 45.0}}})
    close = selection_score({"accuracy": 0.2, "token_reduction_pct": 44.0}, cfg)
    far = selection_score({"accuracy": 0.9, "token_reduction_pct": 5.0}, cfg)
    assert close > far


def test_action_balance_detects_collapse():
    net = QNetwork(STATE_DIM, [8])
    with torch.no_grad():
        net.net[-1].weight[:] = 0.0
        net.net[-1].bias[:] = torch.tensor([-5.0, 5.0])
    balance = action_balance(net, np.random.default_rng(0).random((50, STATE_DIM)))
    assert balance["stop_pct"] == 100.0


# --------------------------------------------------------------------------- #
# reward recomputation (the sweep)
# --------------------------------------------------------------------------- #
def test_recompute_matches_a_multiplier_of_one(cfg):
    frame = _frame(10)
    once = recompute_rewards(frame, cfg, budget=768)
    twice = recompute_rewards(once, cfg, budget=768)
    assert np.allclose(once.reward, twice.reward)


def test_higher_multiplier_lowers_stop_rewards(cfg):
    frame = _frame(10)
    cheap = load_config(overrides={"rl": {"reward": {"cost_multiplier": 0.1}}})
    dear = load_config(overrides={"rl": {"reward": {"cost_multiplier": 4.0}}})
    a = recompute_rewards(frame, cheap, 768)
    b = recompute_rewards(frame, dear, 768)
    stop = a.action == ACTION_STOP
    assert (b.reward[stop] < a.reward[stop]).all()


def test_zero_multiplier_makes_thinking_free(cfg):
    free = load_config(overrides={"rl": {"reward": {"cost_multiplier": 0.0}}})
    frame = recompute_rewards(_frame(5), free, 768)
    cont = (frame.action == ACTION_CONTINUE) & (~frame.done)
    assert np.allclose(frame.reward[cont], 0.0)


# --------------------------------------------------------------------------- #
# behaviour cloning
# --------------------------------------------------------------------------- #
def test_oracle_labels_continue_until_the_first_correct_step():
    labels = oracle_labels(_trace([False, False, True, True]))
    assert labels.tolist() == [0, 0, 1, 1]


def test_oracle_labels_stop_everywhere_when_never_correct():
    assert oracle_labels(_trace([False, False])).tolist() == [1, 1]


def test_oracle_labels_stop_immediately_when_correct_from_the_start():
    assert oracle_labels(_trace([True, True, True])).tolist() == [1, 1, 1]


def test_behaviour_cloning_fits_and_evaluates(cfg):
    traces = [
        _trace([False, False, True, True], qid=f"a{i}", seed=i) for i in range(30)
    ] + [
        _trace([True, True, True, True], qid=f"b{i}", seed=100 + i) for i in range(30)
    ]
    model = BehaviourCloning(cfg)
    stats = model.fit(traces)
    assert stats["n_states"] == 240
    metrics = evaluate_bc(model, traces)
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert metrics["n"] == 60


def test_behaviour_cloning_requires_training(cfg):
    with pytest.raises(RuntimeError, match="not trained"):
        BehaviourCloning(cfg).decide()
