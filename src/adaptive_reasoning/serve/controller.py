"""Phase 7 - the adaptive reasoning controller.

Phases 4-6 replayed policies over recorded traces. This runs one for real: it consumes
a reasoning stream step by step and decides, at each boundary, whether to let the model
keep thinking or to stop and answer now.

The decisive design choice is that state vectors are built by
:func:`adaptive_reasoning.rl.features.build_states` - the same function that built the
training data, called on the steps seen so far. A second, "online" implementation of
the same features is the classic way to get a train/serve mismatch: the policy would be
scored on inputs subtly unlike the ones it learned from, and the failure is silent.
Rebuilding the whole (at most 32-row) matrix each step costs microseconds and removes
that entire class of bug.

Two sources feed the controller:

  * :class:`ReplaySource` - steps from a recorded trace. No GPU, instant, and exact,
    because the recorded probe answer is what the model really would have said.
  * :class:`LiveSource` - steps from a model generating now (see ``serve.live``).

The controller cannot tell them apart, which is the point: what you demonstrate from
replay is what runs live.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Protocol

from ..config import Config
from ..logging_utils import get_logger
from ..rl.features import build_states, difficulty_one_hot

log = get_logger("serve.controller")

CONTINUE, STOP = "CONTINUE", "STOP"

#: Why the controller stopped. Recorded per run because "the policy chose to" and
#: "we hit the safety cap" are very different things when reading a demo.
BY_POLICY = "policy"
BY_TOKEN_CAP = "token_cap"
BY_STREAM_END = "stream_end"


class StepSource(Protocol):
    """Anything that can produce reasoning steps one at a time."""

    question_id: str

    def steps(self) -> Iterator[dict]:
        """Yield step dicts with the keys ``rl.features`` expects."""
        ...


@dataclass
class Decision:
    """What the controller saw and chose at one step boundary."""

    step_index: int
    action: str
    reason: str
    answer: str
    tokens_so_far: int
    confidence: float
    entropy: float
    step_text: str = ""

    @property
    def stopped(self) -> bool:
        return self.action == STOP


@dataclass
class Outcome:
    """The result of running one question through the controller."""

    question_id: str
    answer: str
    stop_step: int
    tokens_used: int
    tokens_available: int
    stop_reason: str
    decisions: list[Decision] = field(default_factory=list)
    disclaimer: str = ""

    @property
    def tokens_saved(self) -> int:
        return max(0, self.tokens_available - self.tokens_used)

    @property
    def token_reduction_pct(self) -> float:
        if self.tokens_available <= 0:
            return 0.0
        return round(100.0 * self.tokens_saved / self.tokens_available, 1)

    def summary(self) -> dict:
        return {
            "question_id": self.question_id,
            "answer": self.answer,
            "stop_step": self.stop_step,
            "tokens_used": self.tokens_used,
            "tokens_available": self.tokens_available,
            "token_reduction_pct": self.token_reduction_pct,
            "stop_reason": self.stop_reason,
            "n_decisions": len(self.decisions),
        }


class ReplaySource:
    """Steps from a recorded trace.

    The demo path. Every field the controller reads was measured during Phase 3, so a
    replay shows exactly what the live system would do - without a GPU in the room.
    """

    def __init__(self, question_id: str, steps: list[dict]):
        self.question_id = question_id
        self._steps = sorted(steps, key=lambda s: s["step_index"])

    @classmethod
    def from_frame(cls, frame, question_id: str) -> ReplaySource:
        """Build from the Phase 3 ``traces.parquet`` table."""
        rows = frame[frame.question_id == question_id]
        if rows.empty:
            raise KeyError(f"no recorded trace for question_id {question_id!r}")
        return cls(question_id, rows.to_dict("records"))

    @property
    def total_tokens(self) -> int:
        """What full reasoning would have cost - the baseline a saving is measured against."""
        return int(self._steps[-1]["tokens_so_far"]) if self._steps else 0

    def steps(self) -> Iterator[dict]:
        yield from self._steps


class AdaptiveController:
    """Runs a stopping policy over a live or replayed reasoning stream."""

    def __init__(self, cfg: Config, policy, difficulty: str | None = None,
                 difficulty_vector=None, budget: int | None = None,
                 min_steps: int | None = None):
        """
        Args:
            policy: a ``Decide`` - ``(state, step_index) -> bool``, True meaning stop.
            difficulty: predicted tier for this question, or None if unknown. It is a
                *prediction* at serve time, never a measured label, matching how the
                policy was trained.
            difficulty_vector: the raw length-3 distribution, when the classifier's
                probabilities are available rather than just its argmax. Overrides
                ``difficulty``. Phase 7's consistency check uses this to feed the
                controller precisely what training saw.
            budget: the token budget the *policy was trained under*, which normalises
                the ``token_ratio`` feature. This is a property of the checkpoint, not
                of the current config: the traces were generated at 768 tokens while
                ``llm.max_new_tokens`` defaults to 1024, and using the wrong one shifts
                that feature by up to 0.25 - enough to change the policy's decisions.
                Phase 4 records the correct value as ``token_budget``.
            min_steps: how many steps must pass before the policy may stop. Also a
                property of the checkpoint - Phase 5 tunes it on validation and every
                reported number uses the tuned value, so serving with a different one
                would silently produce different behaviour from the results table.
                ``serve.min_steps_before_stop`` is the fallback when none is recorded.
        """
        self.cfg = cfg
        self.policy = policy
        self.difficulty = difficulty
        self._difficulty_vector = difficulty_vector
        self.budget = budget if budget is not None else cfg.llm.max_new_tokens
        self.floor = (min_steps if min_steps is not None
                      else cfg.serve.min_steps_before_stop)
        self.cap = cfg.serve.hard_token_cap

    def run(self, source: StepSource, on_decision=None) -> Outcome:
        """Consume ``source`` until the policy stops, a cap trips, or steps run out.

        ``on_decision`` is called with each :class:`Decision` as it is made, which is
        what lets the dashboard stream the reasoning as it happens.
        """
        difficulty_vector = (
            self._difficulty_vector if self._difficulty_vector is not None
            else difficulty_one_hot(self.difficulty)
        )
        seen: list[dict] = []
        decisions: list[Decision] = []
        stop_reason = BY_STREAM_END
        last: dict | None = None

        for step in source.steps():
            seen.append(step)
            last = step
            index = len(seen) - 1

            # Rebuilt from scratch every step, using the training-time code path.
            states = build_states(seen, difficulty_vector, self.cfg, self.budget)
            state = states[-1]

            tokens = int(step["tokens_so_far"])
            over_cap = tokens >= self.cap
            allowed = index >= self.floor
            wants_stop = bool(self.policy(state, index)) if allowed else False

            if over_cap:
                action, reason = STOP, BY_TOKEN_CAP
            elif wants_stop:
                action, reason = STOP, BY_POLICY
            else:
                action, reason = CONTINUE, ""

            decision = Decision(
                step_index=index,
                action=action,
                reason=reason,
                answer=str(step.get("probe_answer", "")),
                tokens_so_far=tokens,
                confidence=float(step.get("confidence", 0.0)),
                entropy=float(step.get("entropy", 0.0)),
                step_text=str(step.get("step_text", "")),
            )
            decisions.append(decision)
            if on_decision is not None:
                on_decision(decision)

            if action == STOP:
                stop_reason = reason
                break

        if last is None:
            raise ValueError(f"{source.question_id}: the source produced no steps")

        available = getattr(source, "total_tokens", int(last["tokens_so_far"]))
        return Outcome(
            question_id=source.question_id,
            answer=str(last.get("probe_answer", "")),
            stop_step=len(decisions) - 1,
            tokens_used=int(last["tokens_so_far"]),
            tokens_available=int(available),
            stop_reason=stop_reason,
            decisions=decisions,
            disclaimer=self.cfg.serve.disclaimer.strip(),
        )


def load_policy(cfg: Config, path=None, kind: str = "dqn"):
    """Load a trained stopping policy as a ``Decide``.

    ``kind`` is ``dqn`` or ``bc``. Both were trained on the same state vectors, so the
    controller does not care which it is given - useful, since Phase 6 found behaviour
    cloning the stronger of the two.
    """
    from .. import paths

    if kind == "bc":
        from ..rl.bc import BehaviourCloning

        model = BehaviourCloning.load(cfg, path or paths.MODELS / "behaviour_cloning.joblib")
        return model.decide()

    if kind != "dqn":
        raise ValueError(f"unknown policy kind {kind!r}, expected 'dqn' or 'bc'")

    import torch

    from ..rl.dqn import QNetwork, greedy_policy

    blob = torch.load(path or paths.DQN_POLICY, map_location="cpu", weights_only=False)
    network = QNetwork(blob["state_dim"], blob["hidden_sizes"])
    network.load_state_dict(blob["state_dict"])
    network.eval()

    if blob["state_dim"] != cfg.rl.state_dim:
        raise ValueError(
            f"checkpoint expects state_dim {blob['state_dim']} but the config says "
            f"{cfg.rl.state_dim} - the policy would read scrambled features"
        )
    return greedy_policy(network)


def always_continue_policy():
    """The full-reasoning control, for side-by-side demonstration."""
    return lambda state, index: False


def compare(cfg: Config, source: ReplaySource, policy, difficulty=None,
            difficulty_vector=None, budget=None, min_steps=None) -> dict:
    """Run one question with and without the policy - the demo's core claim.

    Returns both outcomes plus what the early stop actually cost or bought, which is
    the number an audience wants: did stopping change the answer?
    """
    adaptive = AdaptiveController(cfg, policy, difficulty, difficulty_vector,
                                  budget, min_steps).run(source)
    full = AdaptiveController(cfg, always_continue_policy(), difficulty,
                              difficulty_vector, budget, min_steps).run(source)
    return {
        "adaptive": adaptive,
        "full": full,
        "tokens_saved": full.tokens_used - adaptive.tokens_used,
        "token_reduction_pct": round(
            100.0 * (full.tokens_used - adaptive.tokens_used) / max(full.tokens_used, 1), 1
        ),
        "answer_changed": adaptive.answer.strip() != full.answer.strip(),
    }


def training_budget(cfg: Config) -> int:
    """The token budget the traces were generated under.

    Read from Phase 4's summary rather than the config, because ``llm.max_new_tokens``
    can be changed for a later run while the trained policy still expects the old
    normalisation.
    """
    import json

    from .. import paths

    summary = paths.RESULTS / "phase4_summary.json"
    if summary.exists():
        recorded = json.loads(summary.read_text(encoding="utf-8")).get("token_budget")
        if recorded:
            return int(recorded)
    return int(cfg.llm.max_new_tokens)


def training_min_steps(cfg: Config, kind: str = "dqn") -> int:
    """The step floor the reported results were produced with.

    Phase 5 tunes this on validation and stores it beside the policy. Reading it back
    keeps serving identical to evaluation; falling back to the serve config would
    change the policy's behaviour without changing any number in the report.
    """
    import json

    from .. import paths

    summary = paths.RESULTS / "phase5_summary.json"
    if summary.exists():
        entry = json.loads(summary.read_text(encoding="utf-8"))["results"].get(kind, {})
        if "min_steps" in entry:
            return int(entry["min_steps"])
    return int(cfg.serve.min_steps_before_stop)
