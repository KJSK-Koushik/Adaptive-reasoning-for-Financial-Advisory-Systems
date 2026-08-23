"""Find the questions where stopping early beats reasoning to the end.

    python scripts/find_demo_questions.py --experiment reported
    python scripts/find_demo_questions.py --top 5 --min-year 2015

These are the project's thesis made concrete: the model reaches the right answer, keeps
going, and loses it. Picking one of these to demonstrate is legitimate the way choosing
a worked example for a paper is - the *reported* accuracy stays the aggregate over the
whole test split, and this script prints the full win/loss ledger alongside the
examples so the selection is never mistaken for the result.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from adaptive_reasoning.app.api import DemoStore  # noqa: E402
from adaptive_reasoning.config import load_config, parse_overrides  # noqa: E402
from adaptive_reasoning.serve.controller import (  # noqa: E402
    AdaptiveController,
    ReplaySource,
    always_continue_policy,
)

YEAR = re.compile(r"/((?:19|20)\d{2})/")


def main() -> int:
    parser = argparse.ArgumentParser(description="find the best demo questions")
    parser.add_argument("--experiment", default="reported")
    parser.add_argument("--policy", default="dqn", choices=["dqn", "bc"])
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--min-year", type=int, default=2015)
    parser.add_argument("--set", dest="overrides", action="append", metavar="KEY=VALUE")
    args = parser.parse_args()

    cfg = load_config(args.experiment, parse_overrides(args.overrides))
    store = DemoStore(cfg)
    decide, floor = store.policies[args.policy]

    wins, tally = [], {"win": 0, "loss": 0, "both": 0, "neither": 0}

    for question_id in [str(q) for q in store.questions.index if str(q) in store.test_ids]:
        match = YEAR.search(question_id)
        if match and int(match.group(1)) < args.min_year:
            continue

        vector = store.difficulty_vectors.get(question_id)
        source = ReplaySource.from_frame(store.traces, question_id)
        early = AdaptiveController(cfg, decide, difficulty_vector=vector,
                                   budget=store.budget, min_steps=floor).run(source)
        full = AdaptiveController(cfg, always_continue_policy(),
                                  difficulty_vector=vector,
                                  budget=store.budget).run(source)

        early_right = store.correct_at.get((question_id, early.stop_step))
        full_right = store.correct_at.get((question_id, full.stop_step))

        if early_right and not full_right:
            tally["win"] += 1
            saved = full.tokens_used - early.tokens_used
            wins.append({
                "id": question_id,
                "question": str(store.questions.loc[question_id].question),
                "source": str(store.questions.loc[question_id].source),
                "early": early.answer, "full": full.answer,
                "early_tokens": early.tokens_used, "full_tokens": full.tokens_used,
                "saved_pct": 100.0 * saved / max(full.tokens_used, 1),
                "steps": len(early.decisions),
            })
        elif full_right and not early_right:
            tally["loss"] += 1
        elif early_right and full_right:
            tally["both"] += 1
        else:
            tally["neither"] += 1

    total = sum(tally.values())
    print("=" * 78)
    print(f"  EARLY STOPPING vs FULL REASONING - {total} questions, policy={args.policy}")
    print("=" * 78)
    print(f"  early stopping right, full reasoning wrong   {tally['win']:>4}   <- demo these")
    print(f"  full reasoning right, early stopping wrong   {tally['loss']:>4}")
    print(f"  both right                                   {tally['both']:>4}")
    print(f"  both wrong                                   {tally['neither']:>4}")
    print()
    print("  Early stopping loses more often than it wins on accuracy alone. It is")
    print("  ahead overall because it also spends far fewer tokens - that trade is the")
    print(f"  result, and these {tally['win']} questions are the illustration, not the claim.")

    wins.sort(key=lambda w: -w["saved_pct"])
    print()
    print(f"  TOP {min(args.top, len(wins))} BY TOKENS SAVED")
    print("=" * 78)
    for w in wins[:args.top]:
        print(f"\n  {w['id']}   [{w['source']}]")
        print(f"    Q  {w['question'][:96]}")
        print(f"    stopped early : {w['early'][:60]!r}  ({w['early_tokens']} tokens)")
        print(f"    full reasoning: {w['full'][:60]!r}  ({w['full_tokens']} tokens)")
        print(f"    saved {w['saved_pct']:.0f}% by stopping at step {w['steps']}")

    print("\n" + "=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
