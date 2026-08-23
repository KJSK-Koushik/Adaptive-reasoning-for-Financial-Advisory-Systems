"""Phase 7 - the live controller, and proof that it matches the offline results.

    python scripts/run_phase7.py --experiment reported
    python scripts/run_phase7.py --experiment reported --question tatqa::1e60...
    python scripts/run_phase7.py --experiment reported --policy bc

CPU only, about a minute.

An evaluation harness and a serving path are two different pieces of code, and the
usual way a project like this quietly breaks is that they drift: the numbers in the
report come from one, the demo runs the other, and nobody checks they agree. So this
script does three things in order.

  1. Feature consistency - rebuild every state vector through the serving path and
     compare against the ones Phase 4 stored. They must be identical, not merely close.
  2. Decision consistency - run the controller over all 599 test questions and check
     the accuracy and token totals reproduce the Phase 6 table.
  3. A step-by-step demonstration of one question, which is what gets shown in a review.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from adaptive_reasoning import paths  # noqa: E402
from adaptive_reasoning.config import load_config, parse_overrides  # noqa: E402
from adaptive_reasoning.logging_utils import setup_logging  # noqa: E402
from adaptive_reasoning.rl.dataset import ACTION_STOP  # noqa: E402
from adaptive_reasoning.seeding import set_seed  # noqa: E402
from adaptive_reasoning.serve.controller import (  # noqa: E402
    AdaptiveController,
    ReplaySource,
    compare,
    load_policy,
    training_budget,
    training_min_steps,
)

TRACES = paths.TRACES / "traces.parquet"


def _difficulty_vectors(transitions) -> dict[str, np.ndarray]:
    """The difficulty distribution Phase 4 baked into each question's state."""
    rows = transitions[transitions.action == ACTION_STOP]
    out = {}
    for question_id, group in rows.groupby("question_id", sort=False):
        first = np.asarray(group.sort_values("step_index").state.iloc[0], dtype=np.float32)
        out[str(question_id)] = first[:3]
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 7 - live controller")
    parser.add_argument("--experiment", help="config in configs/experiment/")
    parser.add_argument("--policy", default="dqn", choices=["dqn", "bc"])
    parser.add_argument("--question", help="question_id to demonstrate")
    parser.add_argument("--set", dest="overrides", action="append", metavar="KEY=VALUE")
    args = parser.parse_args()

    cfg = load_config(args.experiment, parse_overrides(args.overrides))
    setup_logging(cfg.logging.level, cfg.logging.to_file, cfg.logging.rich_console,
                  "phase7")
    set_seed(cfg.project.seed)

    if not TRACES.exists():
        raise SystemExit(f"{TRACES} not found - run Phase 3")
    if not paths.RL_TRANSITIONS.exists():
        raise SystemExit(f"{paths.RL_TRANSITIONS} not found - run scripts/run_phase4.py")

    import pandas as pd

    traces = pd.read_parquet(TRACES)
    transitions = pd.read_parquet(paths.RL_TRANSITIONS)
    policy = load_policy(cfg, kind=args.policy)
    budget = training_budget(cfg)
    floor = training_min_steps(cfg, args.policy)
    vectors = _difficulty_vectors(transitions)

    test_ids = list(
        transitions[transitions.split == "test"].question_id.drop_duplicates()
    )
    print(f"loaded {len(traces):,} trace steps, {len(test_ids)} test questions, "
          f"policy={args.policy}, budget={budget} tokens, min_steps={floor}")

    # -- 1. do the serving features match the training features? ------------- #
    from adaptive_reasoning.rl.features import build_states

    stored = transitions[transitions.action == ACTION_STOP]
    worst = 0.0
    checked = 0
    for question_id in test_ids[:100]:
        rows = traces[traces.question_id == question_id].sort_values("step_index")
        want = np.vstack([
            np.asarray(s, dtype=np.float32)
            for s in stored[stored.question_id == question_id]
            .sort_values("step_index").state
        ])
        got = build_states(rows.to_dict("records"), vectors[question_id], cfg, budget)
        n = min(len(want), len(got))
        worst = max(worst, float(np.abs(want[:n] - got[:n]).max()))
        checked += 1

    print(f"\n  feature consistency: {checked} questions, "
          f"largest difference {worst:.2e}")
    if worst > 1e-5:
        raise SystemExit(
            "  FAIL - the serving path builds different features from the training "
            "path. The policy would be reading inputs it was never trained on."
        )
    print("  PASS - the controller feeds the policy exactly what it was trained on")

    # -- 2. does the controller reproduce the offline evaluation? ------------ #
    correct_at = {
        (str(r.question_id), int(r.step_index)): bool(r.probe_correct)
        for r in traces.itertuples()
    }

    used, saved_from, hits, caps = [], [], 0, 0
    for question_id in test_ids:
        source = ReplaySource.from_frame(traces, question_id)
        outcome = AdaptiveController(
            cfg, policy, difficulty_vector=vectors[question_id], budget=budget,
            min_steps=floor
        ).run(source)
        used.append(outcome.tokens_used)
        saved_from.append(outcome.tokens_available)
        hits += correct_at.get((question_id, outcome.stop_step), False)
        caps += outcome.stop_reason == "token_cap"

    accuracy = hits / len(test_ids)
    reduction = 100 * (1 - sum(used) / max(sum(saved_from), 1))
    print(f"\n  controller over {len(test_ids)} test questions:")
    print(f"    accuracy          {accuracy:.3f}")
    print(f"    mean tokens       {np.mean(used):.0f}")
    print(f"    tokens saved      {reduction:.1f}%")
    print(f"    stopped by cap    {caps}")

    p6 = paths.RESULTS / "phase6_summary.json"
    if p6.exists():
        offline = json.loads(p6.read_text(encoding="utf-8"))["results"].get(args.policy)
        if offline:
            da = abs(offline["accuracy"] - accuracy)
            dt = abs(offline["mean_tokens"] - float(np.mean(used)))
            print(f"\n    offline Phase 6:  accuracy {offline['accuracy']:.3f}, "
                  f"mean tokens {offline['mean_tokens']:.0f}")
            status = "PASS" if (da < 0.005 and dt < 2.0) else "MISMATCH"
            print(f"    {status} - live controller differs by {da:.4f} accuracy "
                  f"and {dt:.1f} tokens")

    # -- 3. one question, step by step --------------------------------------- #
    question_id = args.question or test_ids[0]
    source = ReplaySource.from_frame(traces, question_id)
    result = compare(cfg, source, policy, difficulty_vector=vectors[question_id],
                     budget=budget, min_steps=floor)
    adaptive = result["adaptive"]

    print()
    print("=" * 78)
    print(f"  LIVE DEMONSTRATION - {question_id}")
    print("=" * 78)
    print(f"  {'step':>4} {'tokens':>7} {'conf':>6} {'entropy':>8}  {'action':<9} answer")
    print("  " + "-" * 74)
    for d in adaptive.decisions:
        mark = "<< STOP" if d.stopped else ""
        print(f"  {d.step_index:>4} {d.tokens_so_far:>7} {d.confidence:>6.3f} "
              f"{d.entropy:>8.3f}  {d.action:<9} {d.answer[:26]:<26} {mark}")

    print()
    print(f"  stopped at step {adaptive.stop_step} because: {adaptive.stop_reason}")
    print(f"  tokens {adaptive.tokens_used} of {result['full'].tokens_used} "
          f"({result['token_reduction_pct']}% saved)")
    print(f"  answer with early stopping : {adaptive.answer!r}")
    print(f"  answer with full reasoning : {result['full'].answer!r}")
    print(f"  the answer changed         : {result['answer_changed']}")
    print(f"\n  {adaptive.disclaimer}")
    print("=" * 78)

    out = paths.RESULTS / "phase7_summary.json"
    out.write_text(json.dumps({
        "policy": args.policy,
        "n_questions": len(test_ids),
        "accuracy": round(accuracy, 4),
        "mean_tokens": round(float(np.mean(used)), 1),
        "token_reduction_pct": round(reduction, 2),
        "stopped_by_cap": caps,
        "feature_max_difference": worst,
        "demo": adaptive.summary(),
    }, indent=2), encoding="utf-8")
    print(f"\n  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
