"""Run every question the dashboard can show, and look for defects.

    python scripts/audit_ui.py --experiment reported
    python scripts/audit_ui.py --experiment reported --min-year 2015

A demonstration is only as good as its worst question, and nobody clicks through 599
of them by hand. This drives each one through the same controller path the dashboard
uses and checks the things that would make a reviewer lose confidence: a crash, a blank
answer, an inconsistent token count, a correctness verdict the app cannot resolve.

It reports accuracy but does not try to improve it - accuracy is a property of the
system, measured in Phase 6. What this looks for is the app misrepresenting whatever
the system did.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from adaptive_reasoning.app.api import DemoStore  # noqa: E402
from adaptive_reasoning.config import load_config, parse_overrides  # noqa: E402
from adaptive_reasoning.serve.controller import (  # noqa: E402
    AdaptiveController,
    ReplaySource,
    always_continue_policy,
)

#: FinQA and ConvFinQA ids carry the filing year, e.g. finqa::C/2009/page_141.pdf-3.
#: The other five sources have no year and are never filtered out by one.
YEAR = re.compile(r"/((?:19|20)\d{2})/")


def question_year(question_id: str) -> int | None:
    match = YEAR.search(question_id)
    return int(match.group(1)) if match else None


def main() -> int:
    parser = argparse.ArgumentParser(description="audit every dashboard question")
    parser.add_argument("--experiment", default="reported")
    parser.add_argument("--min-year", type=int, default=None,
                        help="drop questions from filings older than this")
    parser.add_argument("--policy", default="dqn", choices=["dqn", "bc"])
    parser.add_argument("--set", dest="overrides", action="append", metavar="KEY=VALUE")
    args = parser.parse_args()

    cfg = load_config(args.experiment, parse_overrides(args.overrides))
    store = DemoStore(cfg)
    decide, floor = store.policies[args.policy]

    ids = [q for q in store.questions.index if str(q) in store.test_ids]
    if args.min_year is not None:
        ids = [q for q in ids
               if (question_year(str(q)) or args.min_year) >= args.min_year]
    print(f"auditing {len(ids)} questions with policy={args.policy}"
          + (f", min year {args.min_year}" if args.min_year else ""))

    problems: list[str] = []
    correct = 0
    by_source: Counter = Counter()
    right_by_source: Counter = Counter()

    for question_id in ids:
        qid = str(question_id)
        row = store.questions.loc[question_id]
        source = str(row.source)
        by_source[source] += 1

        try:
            source_trace = ReplaySource.from_frame(store.traces, qid)
            adaptive = AdaptiveController(
                cfg, decide, difficulty_vector=store.difficulty_vectors.get(qid),
                budget=store.budget, min_steps=floor).run(source_trace)
            full = AdaptiveController(
                cfg, always_continue_policy(),
                difficulty_vector=store.difficulty_vectors.get(qid),
                budget=store.budget).run(source_trace)
        except Exception as exc:                                  # noqa: BLE001
            problems.append(f"CRASH  {qid}: {type(exc).__name__}: {exc}")
            continue

        # things that would make the app misrepresent the system
        if not adaptive.answer.strip():
            problems.append(f"BLANK ANSWER  {qid}")
        if not adaptive.decisions:
            problems.append(f"NO DECISIONS  {qid}")
        if adaptive.tokens_used > full.tokens_used:
            problems.append(
                f"COST  {qid}: early stop used {adaptive.tokens_used} tokens, "
                f"more than full reasoning at {full.tokens_used}")
        if adaptive.stop_step >= len(adaptive.decisions):
            problems.append(f"STOP INDEX  {qid}: {adaptive.stop_step} of "
                            f"{len(adaptive.decisions)} decisions")
        verdict = store.correct_at.get((qid, adaptive.stop_step))
        if verdict is None:
            problems.append(f"NO VERDICT  {qid}: step {adaptive.stop_step} not in "
                            "the recorded trace, so the app cannot say right or wrong")
            continue
        if not str(row.question).strip():
            problems.append(f"EMPTY QUESTION  {qid}")

        correct += bool(verdict)
        right_by_source[source] += bool(verdict)

    print(f"\naccuracy over the audited set: {correct}/{len(ids)} "
          f"({100 * correct / max(len(ids), 1):.1f}%)")
    print("\nby source:")
    for source, n in sorted(by_source.items(), key=lambda kv: -kv[1]):
        print(f"  {source:<16} {right_by_source[source]:>3}/{n:<4} "
              f"{100 * right_by_source[source] / n:>5.1f}%")

    print(f"\ndefects: {len(problems)}")
    for line in problems[:25]:
        print("  " + line)
    if len(problems) > 25:
        print(f"  ... and {len(problems) - 25} more")

    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
