"""Phase 9 - ablations. What is difficulty-awareness actually contributing?

    python scripts/run_phase9.py --experiment reported

CPU only, roughly 40 minutes. Requires the Phase 3 traces.

The project's claim is that conditioning on question difficulty is what makes the
learned policy work. Every result so far compares the *whole system* against baselines,
which cannot separate "difficulty-awareness helps" from "any learned policy helps". So
each ingredient is removed in turn and the system retrained from scratch.

  full            difficulty in the state (classifier prediction) and in the reward
                  (measured label). The reported system.
  no_state        state difficulty replaced by a flat 1/3 vector. The reward still
                  varies by tier, so the agent is charged difficulty-aware prices it
                  cannot see.
  no_reward       one token price for every tier, set to the frequency-weighted mean
                  of the three (0.247) so that *overall* cost pressure is unchanged and
                  only the ratios between tiers disappear. Without that the ablation
                  would confound "no difficulty" with "cheaper tokens".
  neither         both removed - a difficulty-blind DQN.
  no_answer_shape the four answer-shape features removed, to size the other change
                  made late in the project.

Each variant is trained and then evaluated *at its own operating point*, and also
against a fixed-step rule matched to that point, because a variant that merely stops
later would otherwise look better for the wrong reason.
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
from adaptive_reasoning.eval import baselines as bl  # noqa: E402
from adaptive_reasoning.eval import metrics as mt  # noqa: E402
from adaptive_reasoning.logging_utils import setup_logging  # noqa: E402
from adaptive_reasoning.rl import dataset as ds  # noqa: E402
from adaptive_reasoning.rl import dqn as dqn_mod  # noqa: E402
from adaptive_reasoning.rl import rollout as roll  # noqa: E402
from adaptive_reasoning.seeding import set_seed  # noqa: E402

#: Frequency-weighted mean of the three tier prices over the transition table, so a
#: uniform reward costs the same in aggregate as the difficulty-aware one.
UNIFORM_COST = 0.247

VARIANTS = {
    "full": {},
    "no_state": {"rl": {"difficulty_source": "none"}},
    "no_reward": {"rl": {"reward": {"token_cost": {
        "easy": UNIFORM_COST, "medium": UNIFORM_COST, "hard": UNIFORM_COST}}}},
    "neither": {"rl": {"difficulty_source": "none", "reward": {"token_cost": {
        "easy": UNIFORM_COST, "medium": UNIFORM_COST, "hard": UNIFORM_COST}}}},
    # Diagnostic, not deployable: the *measured* difficulty label in the state. A
    # live system cannot have this - it is what Phase 2's classifier is trying to
    # approximate - so it is an upper bound on what any difficulty signal could buy.
    # If even this does not beat the difficulty-blind policy, the idea does not work
    # here and a better classifier would not rescue it.
    "oracle_difficulty": {"rl": {"difficulty_source": "true"}},
    "no_answer_shape": {"rl": {"state_features": [
        "difficulty_easy", "difficulty_medium", "difficulty_hard", "confidence",
        "min_token_confidence", "entropy", "entropy_slope", "token_ratio",
        "step_index_norm", "delta_confidence", "answer_stability", "progress_cue",
        "doubt_cue", "steps_since_answer_change"]}},
}

DESCRIPTIONS = {
    "full": "the reported system",
    "no_state": "policy cannot see difficulty",
    "no_reward": "one token price for every tier",
    "neither": "difficulty-blind entirely",
    "oracle_difficulty": "PERFECT difficulty in the state (not deployable)",
    "no_answer_shape": "without the answer-shape features",
}


def merge(base: dict, extra: dict) -> dict:
    """Deep-merge override dicts."""
    out = dict(base)
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = merge(out[key], value)
        else:
            out[key] = value
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 9 - ablations")
    parser.add_argument("--experiment", default="reported")
    parser.add_argument("--only", nargs="*", help="run just these variants")
    parser.add_argument("--set", dest="overrides", action="append", metavar="KEY=VALUE")
    args = parser.parse_args()

    setup_logging("WARNING", False, True, "phase9")
    base_overrides = parse_overrides(args.overrides)

    names = args.only or list(VARIANTS)
    results: dict[str, dict] = {}
    correct: dict[str, object] = {}

    for name in names:
        cfg = load_config(args.experiment, merge(base_overrides, VARIANTS[name]))
        set_seed(cfg.project.seed, deterministic_torch=True)

        print(f"\n=== {name} - {DESCRIPTIONS[name]} ===")
        print(f"    state_dim {cfg.rl.state_dim}, difficulty_source "
              f"{cfg.rl.difficulty_source}, token_cost "
              f"{cfg.rl.reward.token_cost.easy}/{cfg.rl.reward.token_cost.medium}/"
              f"{cfg.rl.reward.token_cost.hard}")

        frame, _ = ds.build_frame(cfg)
        train_frame = frame[frame.split == "train"]
        val = roll.load_traces(frame, cfg.rl.state_dim, split="val")
        test = roll.load_traces(frame, cfg.rl.state_dim, split="test")

        network, _, _ = dqn_mod.train(train_frame, cfg, val_traces=val)
        policy = dqn_mod.greedy_policy(network)

        # The step floor is tuned on validation, exactly as Phase 5 does it, so no
        # variant is handed an advantage the reported system did not get.
        floor = max(
            (dqn_mod.selection_score(roll.evaluate(val, policy, f), cfg), f)
             for f in range(0, 9))[1]

        metrics, vector = None, None
        rollouts = [roll.rollout(t, policy, floor) for t in test]
        import numpy as np

        vector = np.array([r.correct for r in rollouts], dtype=bool)
        metrics = roll.evaluate(test, policy, floor)

        # a fixed rule pulled to this variant's own cost
        target = metrics["token_reduction_pct"]
        k = min(range(0, 17), key=lambda k: abs(
            roll.evaluate(test, bl.fixed_step(k))["token_reduction_pct"] - target))
        fixed = roll.evaluate(test, bl.fixed_step(k))
        fixed_vec = np.array(
            [r.correct for r in (roll.rollout(t, bl.fixed_step(k)) for t in test)],
            dtype=bool)

        metrics.update({
            "min_steps": floor,
            "state_dim": cfg.rl.state_dim,
            "fixed_step_at": k,
            "fixed_step_accuracy": fixed["accuracy"],
            "fixed_step_reduction_pct": fixed["token_reduction_pct"],
            "margin_over_fixed": round(metrics["accuracy"] - fixed["accuracy"], 4),
            "vs_fixed": {**mt.paired_bootstrap(vector, fixed_vec,
                                               cfg.eval.bootstrap_samples,
                                               cfg.project.seed),
                         **mt.mcnemar(vector, fixed_vec)},
        })
        results[name] = metrics
        correct[name] = vector
        print(f"    accuracy {metrics['accuracy']:.3f}  tokens "
              f"{metrics['mean_tokens']:.0f}  saved {target:.1f}%  "
              f"margin over fixed step {metrics['margin_over_fixed']:+.3f}")

        del network
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # -- what each ingredient was worth -------------------------------------- #
    comparisons = {}
    if "full" in results:
        for name in results:
            if name == "full":
                continue
            cfg = load_config(args.experiment)
            comparisons[f"full_vs_{name}"] = {
                **mt.paired_bootstrap(correct["full"], correct[name],
                                      cfg.eval.bootstrap_samples, cfg.project.seed),
                **mt.mcnemar(correct["full"], correct[name]),
            }

    print()
    print("=" * 88)
    print("  PHASE 9 - ABLATIONS, 599 HELD-OUT TEST QUESTIONS")
    print("=" * 88)
    print(f"  {'variant':<18}{'accuracy':>10}{'tokens':>9}{'saved %':>9}"
          f"{'vs fixed':>10}{'p':>10}  what was removed")
    print("  " + "-" * 84)
    for name in names:
        if name not in results:
            continue
        m = results[name]
        print(f"  {name:<18}{m['accuracy']:>10.3f}{m['mean_tokens']:>9.0f}"
              f"{m['token_reduction_pct']:>9.1f}{m['margin_over_fixed']:>+10.3f}"
              f"{m['vs_fixed']['p_value']:>10.4f}  {DESCRIPTIONS[name]}")

    if comparisons:
        print()
        print("  THE FULL SYSTEM AGAINST EACH ABLATION (accuracy, paired)")
        print("  " + "-" * 84)
        for key, c in comparisons.items():
            name = key.replace("full_vs_", "")
            verdict = "significant" if c["p_value"] < 0.05 else "not significant"
            print(f"  vs {name:<18} {c['difference']:+.3f}  "
                  f"CI [{c['ci_low']:+.3f}, {c['ci_high']:+.3f}]  "
                  f"p={c['p_value']:.4f}  {verdict}")

    out = paths.RESULTS / "phase9_summary.json"
    out.write_text(json.dumps({"results": results, "comparisons": comparisons,
                               "uniform_token_cost": UNIFORM_COST}, indent=2),
                   encoding="utf-8")
    print(f"\n  wrote {out}")
    print("=" * 88)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
