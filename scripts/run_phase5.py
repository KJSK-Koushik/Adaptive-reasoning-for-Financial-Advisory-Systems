"""Phase 5 - train the stopping policies.

    python scripts/run_phase5.py
    python scripts/run_phase5.py --set rl.difficulty_source=none   # ablation

CPU only. The DQN takes a few minutes, behaviour cloning about twenty seconds.
Requires artifacts/traces/transitions.parquet from Phase 4.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

from adaptive_reasoning import paths  # noqa: E402
from adaptive_reasoning.config import load_config, parse_overrides  # noqa: E402
from adaptive_reasoning.logging_utils import setup_logging  # noqa: E402
from adaptive_reasoning.rl import dqn as dqn_mod  # noqa: E402
from adaptive_reasoning.rl import rollout as roll  # noqa: E402
from adaptive_reasoning.rl.bc import BehaviourCloning, evaluate_bc  # noqa: E402
from adaptive_reasoning.seeding import set_seed  # noqa: E402

BC_MODEL = paths.MODELS / "behaviour_cloning.joblib"


def _row(name: str, metrics: dict) -> str:
    return (
        f"  {name:<22} {metrics.get('accuracy', 0):>8.3f} "
        f"{metrics.get('mean_tokens', 0):>9.0f} "
        f"{metrics.get('token_reduction_pct', 0):>9.1f} "
        f"{metrics.get('mean_stop_step', 0):>8.2f}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 5 - train stopping policies")
    parser.add_argument("--experiment", help="config in configs/experiment/")
    parser.add_argument("--skip-bc", action="store_true", help="DQN only")
    parser.add_argument(
        "--sweep", nargs="*", type=float, metavar="M",
        help="train one DQN per reward cost_multiplier and report the "
             "accuracy-versus-cost frontier. Default: 0.1 0.25 0.5 1 2 4",
    )
    parser.add_argument(
        "--set", dest="overrides", action="append", metavar="KEY=VALUE",
        help="config override (repeatable)",
    )
    args = parser.parse_args()

    cfg = load_config(args.experiment, parse_overrides(args.overrides))
    setup_logging(cfg.logging.level, cfg.logging.to_file, cfg.logging.rich_console, "phase5")
    set_seed(cfg.project.seed, deterministic_torch=True)
    paths.ensure_dirs()

    if not paths.RL_TRANSITIONS.exists():
        raise SystemExit(
            f"{paths.RL_TRANSITIONS} not found - run scripts/run_phase4.py first"
        )

    import pandas as pd

    frame = pd.read_parquet(paths.RL_TRANSITIONS)
    train_frame = frame[frame.split == "train"]

    train_traces = roll.load_traces(frame, cfg.rl.state_dim, split="train")
    val_traces = roll.load_traces(frame, cfg.rl.state_dim, split="val")
    test_traces = roll.load_traces(frame, cfg.rl.state_dim, split="test")
    print(f"traces: {len(train_traces)} train, {len(val_traces)} val, {len(test_traces)} test")

    min_steps = cfg.serve.min_steps_before_stop
    results: dict[str, dict] = {}

    # -- reference points ---------------------------------------------------- #
    results["full_reasoning"] = roll.evaluate(test_traces, roll.always_continue(), 0)
    results["oracle"] = roll.evaluate(test_traces, roll.oracle, 0, per_trace=True)

    def fixed_step(k: int):
        return lambda state, index: index >= k

    # -- behaviour cloning --------------------------------------------------- #
    if not args.skip_bc:
        bc = BehaviourCloning(cfg)
        stats = bc.fit(train_traces)
        print(f"behaviour cloning: {stats['n_states']:,} states, "
              f"{stats['stop_label_pct']}% labelled STOP, "
              f"train agreement {stats['train_agreement']:.3f}")
        bc.save(BC_MODEL)
        results["behaviour_cloning"] = evaluate_bc(bc, test_traces, min_steps)

    # -- frontier sweep ------------------------------------------------------ #
    if args.sweep is not None:
        from adaptive_reasoning.difficulty.from_traces import observed_budget
        from adaptive_reasoning.rl.dataset import recompute_rewards
        from adaptive_reasoning.traces.runner import TRACE_SUMMARY

        multipliers = args.sweep or [0.1, 0.25, 0.5, 1.0, 2.0, 4.0]
        budget = observed_budget(pd.read_parquet(TRACE_SUMMARY), cfg)
        frontier = []

        for multiplier in multipliers:
            swept = load_config(
                args.experiment,
                parse_overrides(args.overrides)
                | {"rl": {"reward": {"cost_multiplier": multiplier}}},
            )
            rewritten = recompute_rewards(train_frame, swept, budget)
            net, _, _ = dqn_mod.train(rewritten, swept, val_traces=val_traces)
            metrics = roll.evaluate(test_traces, dqn_mod.greedy_policy(net), min_steps)
            metrics["cost_multiplier"] = multiplier
            frontier.append(metrics)
            print(f"  multiplier {multiplier:>5}: acc {metrics['accuracy']:.3f}  "
                  f"tokens {metrics['mean_tokens']:.0f}  "
                  f"saved {metrics['token_reduction_pct']:.1f}%")

        print()
        print("=" * 68)
        print("  ACCURACY vs COST FRONTIER (test set)")
        print("=" * 68)
        print(f"  {'multiplier':>10} {'accuracy':>9} {'tokens':>8} {'saved %':>9}")
        print("  " + "-" * 40)
        for row in frontier:
            print(f"  {row['cost_multiplier']:>10} {row['accuracy']:>9.3f} "
                  f"{row['mean_tokens']:>8.0f} {row['token_reduction_pct']:>9.1f}")
        print(f"\n  full reasoning: acc {results['full_reasoning']['accuracy']:.3f}, "
              f"{results['full_reasoning']['mean_tokens']:.0f} tokens")
        print(f"  oracle:         acc {results['oracle']['accuracy']:.3f}, "
              f"{results['oracle']['mean_tokens']:.0f} tokens")

        out = paths.RESULTS / "phase5_frontier.json"
        out.write_text(json.dumps({"frontier": frontier,
                                   "full_reasoning": results["full_reasoning"],
                                   "oracle": results["oracle"]}, indent=2), encoding="utf-8")
        print(f"\n  wrote {out}")
        print("=" * 68)
        return 0

    # -- DQN ----------------------------------------------------------------- #
    network, history, best_val = dqn_mod.train(train_frame, cfg, val_traces=val_traces)
    torch.save(
        {
            "state_dict": network.state_dict(),
            "state_dim": cfg.rl.state_dim,
            "hidden_sizes": cfg.rl.dqn.hidden_sizes,
            "state_features": cfg.rl.state_features,
            "difficulty_source": cfg.rl.difficulty_source,
        },
        paths.DQN_POLICY,
    )
    # The floor on how much reasoning must happen before the policy may stop is a real
    # hyperparameter, not just a safety net: correctness climbs until roughly step 6
    # and then plateaus, so a policy allowed to stop at step 0 answers before the model
    # has had a chance. Tuned on validation, reported on test.
    policy = dqn_mod.greedy_policy(network)
    floor_search = []
    for floor in range(0, 9):
        metrics = roll.evaluate(val_traces, policy, min_steps=floor)
        floor_search.append((dqn_mod.selection_score(metrics, cfg), floor, metrics))
    best_floor = max(floor_search)[1]
    print(f"min_steps_before_stop tuned on validation: {best_floor}")

    results["dqn"] = roll.evaluate(test_traces, policy, best_floor)
    results["dqn"]["min_steps"] = best_floor

    # The comparison that actually matters: a fixed stopping point matched to the same
    # token budget. If the learned policy cannot beat that, it is not earning its place.
    matched = min(
        range(0, 16),
        key=lambda k: abs(
            roll.evaluate(test_traces, fixed_step(k))["token_reduction_pct"]
            - results["dqn"]["token_reduction_pct"]
        ),
    )
    results["fixed_step_matched"] = roll.evaluate(test_traces, fixed_step(matched))
    results["fixed_step_matched"]["stop_at"] = matched

    balance = dqn_mod.action_balance(
        network, __import__("numpy").vstack([t.states for t in test_traces])
    )

    # -- report -------------------------------------------------------------- #
    print()
    print("=" * 68)
    print("  PHASE 5 - TEST SET")
    print("=" * 68)
    print(f"  {'policy':<22} {'accuracy':>8} {'tokens':>9} {'saved %':>9} {'stop@':>8}")
    print("  " + "-" * 62)
    for name in ("full_reasoning", "fixed_step_matched", "behaviour_cloning",
                 "dqn", "oracle"):
        if name in results:
            print(_row(name, results[name]))

    fixed, dqn_r = results.get("fixed_step_matched"), results["dqn"]
    if fixed:
        gain = dqn_r["accuracy"] - fixed["accuracy"]
        print(f"\n  At matched cost (~{dqn_r['token_reduction_pct']:.0f}% saved), the DQN is")
        print(f"  {gain:+.3f} accuracy versus stopping at a fixed step {fixed['stop_at']}.")

    print("\n  DQN action balance:")
    print(f"    stops on {balance['stop_pct']}% of states, "
          f"continues on {balance['continue_pct']}%")
    print(f"    mean Q(stop) {balance['mean_q_stop']:+.3f}  "
          f"Q(continue) {balance['mean_q_continue']:+.3f}")

    if balance["stop_pct"] > 98 or balance["continue_pct"] > 98:
        print("\n  WARNING: the policy has collapsed to a single action. Retune the")
        print("  reward token_cost values - it is not making a real decision.")

    print("\n  DQN by difficulty tier (should stop sooner on easy):")
    dqn_metrics = results["dqn"]
    for tier in ("easy", "medium", "hard"):
        if f"{tier}_mean_tokens" in dqn_metrics:
            print(f"    {tier:<8} {dqn_metrics[f'{tier}_mean_tokens']:>7.0f} tokens  "
                  f"acc {dqn_metrics[f'{tier}_accuracy']:.3f}  "
                  f"n={dqn_metrics[f'{tier}_n']}")

    # How much of the available headroom the policy captured.
    full, oracle_m = results["full_reasoning"], results["oracle"]
    span = oracle_m["token_reduction_pct"] - full["token_reduction_pct"]
    if span > 0:
        captured = 100 * (dqn_metrics["token_reduction_pct"] - full["token_reduction_pct"]) / span
        print(f"\n  DQN captured {captured:.0f}% of the oracle's available token saving.")

    payload = {
        "results": results,
        "action_balance": balance,
        "best_val": best_val,
        "history": {
            "steps": history.steps, "loss": history.loss,
            "val_accuracy": history.val_accuracy, "val_tokens": history.val_tokens,
            "val_score": history.val_score,
        },
        "difficulty_source": cfg.rl.difficulty_source,
    }
    out = paths.RESULTS / "phase5_summary.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"\n  wrote {paths.DQN_POLICY}")
    print(f"  wrote {out}")
    print("=" * 68)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
