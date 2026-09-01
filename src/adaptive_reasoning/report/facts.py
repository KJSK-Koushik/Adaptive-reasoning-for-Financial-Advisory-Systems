"""Every figure that appears in a deliverable, read from the results files.

This module exists because the same numbers were being copied by hand into a status
report, a slide deck and a findings document, and they drifted: at one point three
deliverables in the project root quoted three different accuracies, all of which had
been true at some moment. A reader comparing two of them would have caught it.

Nothing here computes a result. It reads what the phases wrote, validates that the
files agree with each other, and exposes them under names a document can use. If a
figure is missing or two phases disagree, loading fails rather than quietly producing a
document with a plausible wrong number in it.
"""

from __future__ import annotations

import contextlib
import json
from dataclasses import dataclass, field
from datetime import date

from .. import paths


class MissingResults(RuntimeError):
    """A phase has not been run, so a figure a document needs does not exist."""


def _load(name: str) -> dict:
    path = paths.RESULTS / name
    if not path.exists():
        raise MissingResults(
            f"{path} not found. Run the phase that writes it before building a "
            f"deliverable - a document with a guessed number in it is worse than no "
            f"document."
        )
    return json.loads(path.read_text(encoding="utf-8"))


@dataclass
class Policy:
    """One row of the results table."""

    key: str
    label: str
    accuracy: float
    mean_tokens: float
    token_reduction_pct: float

    @property
    def accuracy_pct(self) -> str:
        return f"{self.accuracy * 100:.1f}%"

    @property
    def tokens(self) -> str:
        return f"{self.mean_tokens:.0f}"

    @property
    def saved(self) -> str:
        return "—" if self.token_reduction_pct < 0.05 else f"{self.token_reduction_pct:.1f}%"


#: Display names. Keeping them here rather than in each renderer means the deck and the
#: report cannot describe the same policy differently.
LABELS = {
    "full_reasoning": "Full reasoning — no early stop",
    "fixed_step_matched": "Fixed step — the standard approach",
    "fixed_budget_matched": "Fixed token budget",
    "confidence_matched": "Confidence threshold — prior work",
    "entropy_matched": "Entropy threshold — prior work",
    "behaviour_cloning": "Behaviour cloning — supervised control",
    "dqn": "Our RL agent",
    "oracle": "Oracle — upper bound, uses hindsight",
}

#: The order a table presents them in: the baseline first, ours second to last, the
#: unreachable upper bound last.
TABLE_ORDER = [
    "full_reasoning", "fixed_step_matched", "confidence_matched",
    "entropy_matched", "behaviour_cloning", "dqn", "oracle",
]


@dataclass
class Facts:
    """Everything a deliverable is allowed to state."""

    built_on: str
    model_id: str
    budget_tokens: int

    n_questions: int
    n_sources: int
    n_traces: int
    n_transitions: int
    n_test: int
    state_dim: int

    final_accuracy: float
    ever_correct: float
    mean_steps: float

    policies: dict[str, Policy]
    comparisons: dict[str, dict]
    ablations: dict[str, dict] = field(default_factory=dict)

    n_tests_passing: int = 0

    # -- derived, so a document never does arithmetic of its own ------------- #
    @property
    def overthinking_gap(self) -> str:
        return f"{(self.ever_correct - self.final_accuracy) * 100:.1f}"

    @property
    def headline_margin(self) -> str:
        d = self.comparisons["dqn_vs_fixed_step_matched"]["difference"]
        return f"{d * 100:+.1f}"

    @property
    def headline_p(self) -> str:
        p = self.comparisons["dqn_vs_fixed_step_matched"]["p_value"]
        return "< 0.0001" if p < 0.0001 else f"= {p:.4f}"

    @property
    def tokens_saved_pct(self) -> str:
        return f"{self.policies['dqn'].token_reduction_pct:.0f}%"

    @property
    def accuracy_vs_full(self) -> str:
        """How our agent compares to reasoning all the way, in points."""
        d = self.policies["dqn"].accuracy - self.policies["full_reasoning"].accuracy
        return f"{d * 100:+.1f}"

    def margin(self, rival: str) -> str:
        key = f"dqn_vs_{rival}"
        if key not in self.comparisons:
            raise MissingResults(f"no recorded comparison against {rival}")
        return f"{self.comparisons[key]['difference'] * 100:+.1f}"

    def table_rows(self) -> list[Policy]:
        return [self.policies[k] for k in TABLE_ORDER if k in self.policies]


def load(n_tests_passing: int = 0) -> Facts:
    """Read the current results and check the phases agree with each other."""
    p1, p3 = _load("phase1_summary.json"), _load("phase3_summary.json")
    p4, p5 = _load("phase4_summary.json"), _load("phase5_summary.json")
    p6 = _load("phase6_summary.json")

    results = p6["results"]
    policies = {
        key: Policy(
            key=key, label=LABELS.get(key, key.replace("_", " ")),
            accuracy=row["accuracy"], mean_tokens=row["mean_tokens"],
            token_reduction_pct=row["token_reduction_pct"],
        )
        for key, row in results.items() if key in LABELS
    }

    # Phase 5 and Phase 6 measure the same policies on the same split. If they
    # disagree, one of them is stale and a document built now would be wrong.
    for key in ("dqn", "oracle", "full_reasoning"):
        a = p5["results"].get(key, {}).get("accuracy")
        b = results.get(key, {}).get("accuracy")
        if a is not None and b is not None and abs(a - b) > 1e-9:
            raise MissingResults(
                f"phase 5 and phase 6 disagree on {key}: {a} vs {b}. Rerun the later "
                f"phase - the deliverable would otherwise quote whichever ran last."
            )

    ablations = {}
    with contextlib.suppress(MissingResults):
        # A report may legitimately predate the ablations.
        ablations = _load("phase9_summary.json")["results"]

    return Facts(
        built_on=date.today().isoformat(),
        model_id="DeepSeek-R1-Distill-Qwen-1.5B",
        budget_tokens=int(p4["token_budget"]),
        n_questions=int(p1["total"]),
        n_sources=len(p1["by_source"]),
        n_traces=int(p3["n_traces"]),
        n_transitions=int(p4["n_transitions"]),
        n_test=int(results["dqn"]["n"]),
        state_dim=int(p4["state_dim"]),
        final_accuracy=float(p3["final_accuracy"]),
        ever_correct=float(p3["solvable_fraction"]),
        mean_steps=float(p3["mean_steps"]),
        policies=policies,
        comparisons=p6.get("comparisons", {}),
        ablations=ablations,
        n_tests_passing=n_tests_passing,
    )
