"""Behaviour cloning: supervised imitation of the oracle.

This is the control experiment for the whole project. The obvious objection to a DQN
result is *"you got a gain because you trained something and the baselines didn't - the
reinforcement learning is decorative."* Fixed thresholds cannot answer that, because
they do not learn at all.

Behaviour cloning can. It sees the same features, the same traces, and the same oracle
information, and it is trained with plain supervised learning. The only thing it lacks
is any notion of a sequential cost-accuracy trade-off: it is told *what the oracle did*
and imitates it, rather than optimising a return.

So if the DQN beats it, the gain is attributable to sequential decision-making under
the difficulty-aware reward - not merely to "we fitted a model". And if it does not,
that is a real finding worth reporting honestly, and there is still a working learned
policy to fall back on.

Cost: about twenty seconds of CPU on top of data Phase 4 already produced.
"""

from __future__ import annotations

import numpy as np

from ..config import Config
from ..logging_utils import get_logger
from .rollout import Trace

log = get_logger("rl.bc")

STOP, CONTINUE = 1, 0


def oracle_labels(trace: Trace) -> np.ndarray:
    """Per-step target actions: CONTINUE until the oracle's stopping point, then STOP.

    A trace that is never correct is labelled STOP throughout - if no stopping point
    would have produced a right answer, the least-bad action is to stop immediately and
    save the tokens.
    """
    target = trace.earliest_correct
    labels = np.full(trace.n_steps, STOP, dtype=np.int64)
    if target is not None:
        labels[:target] = CONTINUE
    return labels


def build_dataset(traces: list[Trace]) -> tuple[np.ndarray, np.ndarray]:
    states = np.vstack([t.states for t in traces])
    labels = np.concatenate([oracle_labels(t) for t in traces])
    return states, labels


class BehaviourCloning:
    """Gradient-boosted classifier over the same state features the DQN sees."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.model = None

    def fit(self, traces: list[Trace]) -> dict:
        import lightgbm as lgb

        x, y = build_dataset(traces)
        self.model = lgb.LGBMClassifier(
            objective="binary",
            n_estimators=300,
            learning_rate=0.05,
            # The oracle stops early, so CONTINUE labels are the minority. Without
            # rebalancing the model learns to answer "stop" everywhere.
            class_weight="balanced",
            random_state=self.cfg.project.seed,
            verbose=-1,
        )
        self.model.fit(x, y)

        predictions = self.model.predict(x)
        return {
            "n_states": int(len(y)),
            "stop_label_pct": round(float(100 * (y == STOP).mean()), 1),
            "train_agreement": round(float((predictions == y).mean()), 4),
        }

    def decide(self):
        """Decision function for :mod:`.rollout`."""
        if self.model is None:
            raise RuntimeError("behaviour cloning model is not trained")

        def _decide(state: np.ndarray, index: int) -> bool:
            return bool(self.model.predict(state.reshape(1, -1))[0] == STOP)

        return _decide

    def decide_batch(self, trace: Trace):
        """Faster equivalent for offline replay: score the whole trace at once.

        Per-state ``predict`` calls dominate the runtime otherwise - the model is
        trivially fast but the Python call overhead is not.
        """
        predictions = self.model.predict(trace.states)

        def _decide(state: np.ndarray, index: int) -> bool:
            return bool(predictions[index] == STOP)

        return _decide

    def save(self, path) -> None:
        import joblib

        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)
        log.info("saved behaviour cloning model to %s", path)

    @classmethod
    def load(cls, cfg: Config, path):
        import joblib

        model = cls(cfg)
        model.model = joblib.load(path)
        return model


def evaluate_bc(model: BehaviourCloning, traces: list[Trace], min_steps: int = 0) -> dict:
    """Evaluate through the shared rollout path, scoring each trace in one call."""
    from .rollout import evaluate

    return evaluate(traces, model.decide_batch, min_steps, per_trace=True)
