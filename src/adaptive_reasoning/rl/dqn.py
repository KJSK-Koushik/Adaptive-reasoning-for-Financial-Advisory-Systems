"""The difficulty-aware stopping policy: a Double DQN trained offline.

The network is deliberately tiny - 14 inputs, two hidden layers, two outputs, about
19,000 parameters - roughly one hundred-thousandth the size of the 1.5B model it
controls. All the expensive work happened in Phase 3; this trains in minutes on a CPU
and can be re-run dozens of times while tuning the reward.

**Why offline TD learning is well-behaved here.** Offline RL usually suffers from
overestimation on out-of-distribution actions: the network extrapolates a high value
for something the dataset never tried, and nothing corrects it. That failure mode does
not arise in this problem. There are exactly two actions, and Phase 3 recorded the
outcome of *both* at *every* state, so the dataset covers the action space completely.
No conservative penalty (CQL, BCQ) is needed, and adding one would only bias the
result.

Epsilon-greedy exploration is likewise irrelevant - there is no environment to explore,
only a fixed table of transitions to fit.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

from ..config import Config
from ..logging_utils import get_logger
from .dataset import ACTION_CONTINUE, ACTION_STOP

log = get_logger("rl.dqn")


class QNetwork(nn.Module):
    """Maps a reasoning state to Q-values for (CONTINUE, STOP)."""

    def __init__(self, state_dim: int, hidden_sizes: list[int], n_actions: int = 2):
        super().__init__()
        layers: list[nn.Module] = []
        size = state_dim
        for width in hidden_sizes:
            layers += [nn.Linear(size, width), nn.ReLU()]
            size = width
        layers.append(nn.Linear(size, n_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class Batch:
    states: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_states: torch.Tensor
    dones: torch.Tensor


@dataclass
class TrainHistory:
    steps: list[int] = field(default_factory=list)
    loss: list[float] = field(default_factory=list)
    val_accuracy: list[float] = field(default_factory=list)
    val_tokens: list[float] = field(default_factory=list)
    val_score: list[float] = field(default_factory=list)


class OfflineBuffer:
    """The Phase 4 transition table, as tensors."""

    def __init__(self, frame, state_dim: int, device: str = "cpu"):
        self.states = torch.tensor(
            np.vstack([np.asarray(s, dtype=np.float32) for s in frame.state]),
            device=device,
        )
        self.next_states = torch.tensor(
            np.vstack([np.asarray(s, dtype=np.float32) for s in frame.next_state]),
            device=device,
        )
        self.actions = torch.tensor(frame.action.to_numpy(), dtype=torch.long, device=device)
        self.rewards = torch.tensor(
            frame.reward.to_numpy(), dtype=torch.float32, device=device
        )
        self.dones = torch.tensor(
            frame.done.to_numpy().astype(np.float32), dtype=torch.float32, device=device
        )
        if self.states.shape[1] != state_dim:
            raise ValueError(
                f"transitions have state dimension {self.states.shape[1]}, config says "
                f"{state_dim}; rebuild with scripts/run_phase4.py"
            )

        # A NaN anywhere here poisons every gradient from the first step onward, and
        # the only symptom is a policy that never learns - no exception, no warning.
        for name, tensor in (("states", self.states), ("next_states", self.next_states),
                             ("rewards", self.rewards)):
            if not torch.isfinite(tensor).all():
                n_bad = int((~torch.isfinite(tensor)).sum())
                raise ValueError(
                    f"{n_bad} non-finite values in {name}. Rebuild the transitions "
                    f"with scripts/run_phase4.py."
                )

        self.n = len(self.actions)

    def sample(self, batch_size: int, generator: torch.Generator) -> Batch:
        idx = torch.randint(0, self.n, (batch_size,), generator=generator)
        return Batch(
            states=self.states[idx],
            actions=self.actions[idx],
            rewards=self.rewards[idx],
            next_states=self.next_states[idx],
            dones=self.dones[idx],
        )


def greedy_policy(network: QNetwork):
    """Wrap a trained network as a decision function for :mod:`.rollout`."""
    def decide(state: np.ndarray, index: int) -> bool:
        with torch.no_grad():
            q = network(torch.tensor(state, dtype=torch.float32).unsqueeze(0))
        return bool(q.argmax(dim=1).item() == ACTION_STOP)
    return decide


def selection_score(metrics: dict, cfg: Config) -> float:
    """Single number for choosing the best checkpoint.

    Accuracy alone would pick the do-nothing policy that always reasons to the end;
    token reduction alone would pick the one that answers instantly and is always
    wrong.

    ``accuracy_at_budget`` states the operating point explicitly - maximise accuracy
    subject to a minimum saving - instead of letting an arbitrary trade-off weight
    decide it silently. That matters here because the comparison against behaviour
    cloning is only meaningful at matched cost.
    """
    accuracy = metrics.get("accuracy", 0.0)
    reduction = metrics.get("token_reduction_pct", 0.0)

    if cfg.rl.selection.objective == "accuracy_at_budget":
        budget = cfg.rl.selection.min_token_reduction
        if reduction < budget:
            # Below the budget: rank by how close it gets, always beneath any
            # candidate that meets it.
            return -1.0 + reduction / max(budget, 1e-9)
        return float(accuracy)

    return float(accuracy + cfg.rl.selection.weight * reduction / 100.0)


def train(
    frame,
    cfg: Config,
    val_traces=None,
    device: str = "cpu",
) -> tuple[QNetwork, TrainHistory, dict]:
    """Train a Double DQN on the offline transitions.

    Returns the best network by validation score, its history, and the best metrics.
    """
    from .rollout import evaluate

    dqn_cfg = cfg.rl.dqn
    torch.manual_seed(cfg.project.seed)
    generator = torch.Generator().manual_seed(cfg.project.seed)

    buffer = OfflineBuffer(frame, cfg.rl.state_dim, device)
    log.info("training on %d transitions", buffer.n)

    online = QNetwork(cfg.rl.state_dim, dqn_cfg.hidden_sizes).to(device)
    target = QNetwork(cfg.rl.state_dim, dqn_cfg.hidden_sizes).to(device)
    target.load_state_dict(online.state_dict())
    target.eval()

    optimiser = torch.optim.Adam(online.parameters(), lr=dqn_cfg.learning_rate)
    history = TrainHistory()

    best_score = float("-inf")
    best_state = {k: v.clone() for k, v in online.state_dict().items()}
    best_metrics: dict = {}

    for step in range(1, dqn_cfg.train_steps + 1):
        batch = buffer.sample(dqn_cfg.batch_size, generator)

        q_values = online(batch.states).gather(1, batch.actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            if dqn_cfg.double_dqn:
                # Choose with the online network, value with the target network. This
                # is what stops the max operator from compounding its own optimism.
                best_next = online(batch.next_states).argmax(dim=1, keepdim=True)
                next_q = target(batch.next_states).gather(1, best_next).squeeze(1)
            else:
                next_q = target(batch.next_states).max(dim=1).values
            targets = batch.rewards + dqn_cfg.gamma * (1 - batch.dones) * next_q

        loss = nn.functional.smooth_l1_loss(q_values, targets)

        optimiser.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(online.parameters(), dqn_cfg.grad_clip)
        optimiser.step()

        if step % dqn_cfg.target_update_interval == 0:
            target.load_state_dict(online.state_dict())

        if val_traces and step % dqn_cfg.eval_every == 0:
            online.eval()
            metrics = evaluate(val_traces, greedy_policy(online),
                               min_steps=cfg.serve.min_steps_before_stop)
            online.train()

            score = selection_score(metrics, cfg)
            history.steps.append(step)
            history.loss.append(float(loss.item()))
            history.val_accuracy.append(metrics["accuracy"])
            history.val_tokens.append(metrics["mean_tokens"])
            history.val_score.append(score)

            if score > best_score:
                best_score, best_metrics = score, metrics
                best_state = {k: v.clone() for k, v in online.state_dict().items()}

            log.info(
                "step %6d  loss %.4f  val acc %.3f  tokens %.0f  saved %.1f%%  score %.4f",
                step, loss.item(), metrics["accuracy"], metrics["mean_tokens"],
                metrics["token_reduction_pct"], score,
            )

    online.load_state_dict(best_state)
    online.eval()
    return online, history, best_metrics


def action_balance(network: QNetwork, states: np.ndarray) -> dict:
    """How often the policy chooses each action - catches collapse to one action."""
    with torch.no_grad():
        q = network(torch.tensor(states, dtype=torch.float32))
        actions = q.argmax(dim=1).numpy()
    return {
        "stop_pct": round(float(100 * (actions == ACTION_STOP).mean()), 1),
        "continue_pct": round(float(100 * (actions == ACTION_CONTINUE).mean()), 1),
        "mean_q_stop": round(float(q[:, ACTION_STOP].mean()), 4),
        "mean_q_continue": round(float(q[:, ACTION_CONTINUE].mean()), 4),
    }
