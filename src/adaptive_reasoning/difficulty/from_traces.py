"""Deriving difficulty labels from reasoning traces, at zero GPU cost.

The original plan sampled the model ``k`` times per question purely to measure how hard
each one was. Measured on a Kaggle T4, that job was **71% of the total GPU budget** -
about 20 million of 29 million generated tokens - and would have pushed the run to
roughly 60 hours against a 30-hour weekly quota.

It is also redundant. The Phase 3 traces already probe the model at every step
boundary, recording whether it was right at that point and whether its answer moved.
That is the same evidence, already paid for.

The primary signal here is ``correct_ratio``: the fraction of probe points at which the
answer was already correct. It is arguably a *better* measure than a k-sample pass rate
for this particular purpose, because it answers the question the stopping policy
actually faces - **how early, and how consistently, does the model know this?**

* A question solved at step 2 of 16 and held scores ~0.88 → genuinely easy to stop on.
* A question only correct at the final step scores ~0.06 → the policy must be patient.
* A question never correct scores 0.0 → hard, and no stopping point would have helped.

The three signals mirror the k-sampling rules in :mod:`.labeling`, so the methodology
story stays consistent: a correctness rate, a length penalty, and an instability
demotion.

This module is pure CPU and has no LLM dependency, so the thresholds can be re-tuned
and the labels regenerated in seconds without touching a GPU.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..config import Config
from ..logging_utils import get_logger
from ..schema import Difficulty
from .labeling import DifficultyVerdict

log = get_logger("difficulty.from_traces")


@dataclass
class TraceEvidence:
    """Per-question aggregates taken from a trace."""

    question_id: str
    n_steps: int
    n_correct_steps: int
    total_tokens: int
    final_correct: bool
    earliest_correct_step: int | None
    n_answer_changes: int

    @property
    def correct_ratio(self) -> float:
        """Fraction of probe points at which the answer was already correct."""
        return self.n_correct_steps / self.n_steps if self.n_steps else 0.0

    @property
    def change_ratio(self) -> float:
        """Fraction of steps at which the tentative answer moved."""
        return self.n_answer_changes / self.n_steps if self.n_steps else 0.0


_ORDER = [Difficulty.EASY, Difficulty.MEDIUM, Difficulty.HARD]


def _bump(level: Difficulty) -> Difficulty:
    return _ORDER[min(_ORDER.index(level) + 1, len(_ORDER) - 1)]


def label_from_trace(
    evidence: TraceEvidence, cfg: Config, budget: int | None = None
) -> DifficultyVerdict:
    """Assign a difficulty tier to one question from its trace.

    ``budget`` is the token cap the traces were actually generated under. It must come
    from the data rather than ``cfg.llm.max_new_tokens``: traces are often produced on
    a remote machine with an overridden budget (the first real run used 768 while the
    local default said 1024), and using the config value silently mis-scales the
    length rule.
    """
    rules = cfg.difficulty.from_traces
    budget = budget or cfg.llm.max_new_tokens
    ratio = evidence.correct_ratio

    if evidence.earliest_correct_step is None:
        # The model was never right at any point. No stopping decision could have
        # rescued it, which is the definition of hard for this policy.
        level = Difficulty.HARD
        reason = "never correct at any probe point"
    elif ratio >= rules.easy_min_correct_ratio:
        level = Difficulty.EASY
        reason = f"correct at {ratio:.0%} of probes >= {rules.easy_min_correct_ratio:.0%}"
    elif ratio <= rules.hard_max_correct_ratio:
        level = Difficulty.HARD
        reason = f"correct at only {ratio:.0%} of probes <= {rules.hard_max_correct_ratio:.0%}"
    else:
        level = Difficulty.MEDIUM
        reason = f"correct at {ratio:.0%} of probes"

    # Solved, but only after burning most of the budget - not genuinely easy.
    length_ratio = evidence.total_tokens / budget if budget else 0.0
    if level is not Difficulty.HARD and length_ratio > rules.long_reasoning_ratio:
        level = _bump(level)
        reason += f"; bumped for using {length_ratio:.0%} of the token budget"

    # The answer kept moving: the model is casting about, not converging.
    if level is Difficulty.MEDIUM and evidence.change_ratio >= rules.unstable_answer_ratio:
        level = Difficulty.HARD
        reason += f"; bumped for {evidence.change_ratio:.0%} answer instability"

    return DifficultyVerdict(
        question_id=evidence.question_id,
        difficulty=level,
        pass_rate=ratio,
        n_samples=evidence.n_steps,
        median_reasoning_tokens=float(evidence.total_tokens),
        answer_diversity=evidence.change_ratio,
        reason=reason,
    )


def evidence_from_frames(steps, summary) -> list[TraceEvidence]:
    """Aggregate the Phase 3 parquet outputs into per-question evidence.

    ``steps`` is ``traces.parquet``; ``summary`` is ``trace_summary.parquet``.
    """
    grouped = steps.groupby("question_id").agg(
        n_steps=("step_index", "count"),
        n_correct_steps=("probe_correct", "sum"),
        n_answer_changes=("answer_changed", "sum"),
    )

    indexed = summary.set_index("question_id")
    evidence: list[TraceEvidence] = []

    for question_id, row in grouped.iterrows():
        if question_id not in indexed.index:
            log.warning("no summary row for %s, skipping", question_id)
            continue
        info = indexed.loc[question_id]
        earliest = info.get("earliest_correct_step")
        evidence.append(
            TraceEvidence(
                question_id=str(question_id),
                n_steps=int(row.n_steps),
                n_correct_steps=int(row.n_correct_steps),
                total_tokens=int(info.total_tokens),
                final_correct=bool(info.final_correct),
                earliest_correct_step=None if earliest is None or earliest != earliest
                else int(earliest),
                n_answer_changes=int(row.n_answer_changes),
            )
        )
    return evidence


def observed_budget(summary, cfg: Config) -> int:
    """The token cap the traces were really generated under.

    Taken as the maximum observed length: with a few thousand traces, some will always
    hit the cap. Falls back to the config value if the data looks implausible.
    """
    try:
        observed = int(summary["total_tokens"].max())
    except (KeyError, ValueError, TypeError):
        return cfg.llm.max_new_tokens
    if observed <= 0:
        return cfg.llm.max_new_tokens
    if observed != cfg.llm.max_new_tokens:
        log.info(
            "traces were generated with a %d-token budget (config says %d); using the "
            "observed value for the length rule", observed, cfg.llm.max_new_tokens,
        )
    return observed


def label_all_from_traces(steps, summary, cfg: Config) -> list[DifficultyVerdict]:
    """Label every question that has a trace."""
    evidence = evidence_from_frames(steps, summary)
    budget = observed_budget(summary, cfg)
    verdicts = [label_from_trace(e, cfg, budget) for e in evidence]
    log.info("derived %d difficulty labels from traces", len(verdicts))
    return verdicts
