"""Phase 6 - evaluate every stopping policy on one footing.

    python scripts/run_phase6.py --experiment reported

CPU only, about a minute. Requires artifacts/traces/transitions.parquet (Phase 4) and
the models written by Phase 5.

Phase 5 answered "does the DQN beat a fixed step?". Phase 6 answers the harder
question: does it beat every reasonable rule someone might write instead, at the same
token cost, by a margin larger than the noise?

Thresholds for the baselines are tuned on validation with the same objective used to
select the DQN checkpoint, so no policy gets a tuning advantage. Matched-cost variants
are then chosen on test to sit as close as possible to the DQN's token budget - that
choice favours the baselines, which is the conservative direction.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from adaptive_reasoning import paths  # noqa: E402
from adaptive_reasoning.config import load_config, parse_overrides  # noqa: E402
from adaptive_reasoning.eval import baselines as bl  # noqa: E402
from adaptive_reasoning.eval import metrics as mt  # noqa: E402
from adaptive_reasoning.logging_utils import setup_logging  # noqa: E402
from adaptive_reasoning.rl import dqn as dqn_mod  # noqa: E402
from adaptive_reasoning.rl import rollout as roll  # noqa: E402
from adaptive_reasoning.rl.bc import BehaviourCloning  # noqa: E402
from adaptive_reasoning.seeding import set_seed  # noqa: E402

BC_MODEL = paths.MODELS / "behaviour_cloning.joblib"


def measure(traces, decide, min_steps=0, per_trace=False):
    """Metrics plus the per-question correctness vector the paired tests need."""
    rollouts = [
        roll.rollout(t, decide(t) if per_trace else decide, min_steps) for t in traces
    ]
    correct = np.array([r.correct for r in rollouts], dtype=bool)
    return roll.evaluate(traces, decide, min_steps, per_trace=per_trace), correct


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 6 - baselines and evaluation")
    parser.add_argument("--experiment", help="config in configs/experiment/")
    parser.add_argument("--set", dest="overrides", action="append", metavar="KEY=VALUE")
    args = parser.parse_args()

    cfg = load_config(args.experiment, parse_overrides(args.overrides))
    setup_logging(cfg.logging.level, cfg.logging.to_file, cfg.logging.rich_console,
                  "phase6")
    set_seed(cfg.project.seed)
    paths.ensure_dirs()

    if not paths.RL_TRANSITIONS.exists():
        raise SystemExit(f"{paths.RL_TRANSITIONS} not found - run scripts/run_phase4.py")
    if not paths.DQN_POLICY.exists():
        raise SystemExit(f"{paths.DQN_POLICY} not found - run scripts/run_phase5.py")

    import pandas as pd

    frame = pd.read_parquet(paths.RL_TRANSITIONS)
    val = roll.load_traces(frame, cfg.rl.state_dim, split="val")
    test = roll.load_traces(frame, cfg.rl.state_dim, split="test")
    print(f"traces: {len(val)} validation, {len(test)} test")

    floor = cfg.serve.min_steps_before_stop

    def score(metrics):
        return dqn_mod.selection_score(metrics, cfg)

    # -- the learned policies ------------------------------------------------ #
    blob = torch.load(paths.DQN_POLICY, map_location="cpu", weights_only=False)
    network = dqn_mod.QNetwork(blob["state_dim"], blob["hidden_sizes"])
    network.load_state_dict(blob["state_dict"])
    network.eval()
    dqn_policy = dqn_mod.greedy_policy(network)

    # Phase 5 tuned this floor on validation; reuse it rather than re-tuning on test.
    dqn_floor = 0
    p5 = paths.RESULTS / "phase5_summary.json"
    if p5.exists():
        dqn_floor = json.loads(p5.read_text(encoding="utf-8"))["results"]["dqn"].get(
            "min_steps", 0)

    results: dict[str, dict] = {}
    correct: dict[str, np.ndarray] = {}

    def record(name, metrics, vector, **extra):
        metrics.update(extra)
        hits = int(vector.sum())
        low, high = mt.wilson_interval(hits, len(vector))
        metrics["accuracy_ci"] = [round(low, 4), round(high, 4)]
        metrics.update(mt.token_cost_model(metrics["mean_tokens"]))
        results[name] = metrics
        correct[name] = vector

    record("full_reasoning", *measure(test, bl.full_reasoning()))
    record("oracle", *measure(test, roll.oracle, per_trace=True))
    record("dqn", *measure(test, dqn_policy, dqn_floor), min_steps=dqn_floor)

    if BC_MODEL.exists():
        bc = BehaviourCloning.load(cfg, BC_MODEL)
        record("behaviour_cloning",
               *measure(test, bc.decide_batch, floor, per_trace=True))

    # -- tuned baselines ----------------------------------------------------- #
    conf_i = bl.feature_index(cfg, "confidence")
    ent_i = bl.feature_index(cfg, "entropy")
    stab_i = bl.feature_index(cfg, "steps_since_answer_change")
    max_steps = cfg.traces.max_steps

    tau_c, _ = bl.tune_threshold(
        val, lambda t: bl.confidence_threshold(t, conf_i), cfg.eval.confidence_tau,
        score, floor)
    record("confidence_threshold",
           *measure(test, bl.confidence_threshold(tau_c, conf_i), floor), tau=tau_c)

    tau_e, _ = bl.tune_threshold(
        val, lambda t: bl.entropy_threshold(t, ent_i), cfg.eval.entropy_tau,
        score, floor)
    record("entropy_threshold",
           *measure(test, bl.entropy_threshold(tau_e, ent_i), floor), tau=tau_e)

    n_stab, _ = bl.tune_threshold(
        val, lambda n: bl.answer_stability(int(n), stab_i, max_steps),
        [1, 2, 3, 4, 5], score, floor)
    record("answer_stability",
           *measure(test, bl.answer_stability(int(n_stab), stab_i, max_steps), floor),
           n_stable=int(n_stab))

    budget, _ = bl.tune_threshold(
        val, lambda b: bl.fixed_budget(int(b)), cfg.eval.fixed_budget_tokens,
        score, floor, per_trace=True)
    record("fixed_budget",
           *measure(test, bl.fixed_budget(int(budget)), floor, per_trace=True),
           budget_tokens=int(budget))


    # -- matched-cost comparison --------------------------------------------- #
    # Accuracy at a policy's own natural operating point is not comparable across
    # policies - a rule that stops later will look better simply because it spent
    # more. So every rival is re-tuned to land as close as possible to the DQN's
    # token budget, and only those matched numbers support a claim.
    target = results["dqn"]["token_reduction_pct"]

    def match_cost(build, grid, per_trace=False):
        """Grid value whose test token_reduction sits closest to the DQN's."""
        return min(grid, key=lambda v: abs(
            roll.evaluate(test, build(v), per_trace=per_trace)["token_reduction_pct"]
            - target))

    matched_k = match_cost(bl.fixed_step, range(0, 17))
    record("fixed_step_matched", *measure(test, bl.fixed_step(matched_k)),
           stop_at=matched_k)

    budget_grid = sorted(set(list(cfg.eval.fixed_budget_tokens)
                             + list(range(64, 480, 16))))
    matched_b = match_cost(lambda b: bl.fixed_budget(int(b)), budget_grid,
                           per_trace=True)
    record("fixed_budget_matched",
           *measure(test, bl.fixed_budget(int(matched_b)), per_trace=True),
           budget_tokens=int(matched_b))

    conf_grid = [round(0.40 + 0.02 * i, 2) for i in range(31)]
    matched_c = match_cost(lambda t: bl.confidence_threshold(t, conf_i), conf_grid)
    record("confidence_matched",
           *measure(test, bl.confidence_threshold(matched_c, conf_i)), tau=matched_c)

    ent_grid = [round(0.05 * i, 2) for i in range(1, 41)]
    matched_e = match_cost(lambda t: bl.entropy_threshold(t, ent_i), ent_grid)
    record("entropy_matched",
           *measure(test, bl.entropy_threshold(matched_e, ent_i)), tau=matched_e)

    prob_grid = [round(0.05 * i, 2) for i in range(1, 20)]
    matched_p = match_cost(lambda p: bl.random_stop(p, cfg.project.seed), prob_grid,
                           per_trace=True)
    record("random_matched",
           *measure(test, bl.random_stop(matched_p, cfg.project.seed), per_trace=True),
           probability=matched_p)

    stab_grid = list(range(1, 9))
    matched_s = match_cost(
        lambda n: bl.answer_stability(int(n), stab_i, max_steps), stab_grid)
    record("stability_matched",
           *measure(test, bl.answer_stability(int(matched_s), stab_i, max_steps)),
           n_stable=int(matched_s))

    # A family that cannot be pushed to the DQN's budget is not a fair rival at
    # that budget, and saying so is more honest than quietly comparing across costs.
    TOLERANCE = 3.0
    for name in ("fixed_step_matched", "fixed_budget_matched", "confidence_matched",
                 "entropy_matched", "stability_matched", "random_matched"):
        if name in results:
            gap = results[name]["token_reduction_pct"] - target
            results[name]["cost_gap_pct"] = round(gap, 2)
            results[name]["cost_matched"] = bool(abs(gap) <= TOLERANCE)

    # -- significance -------------------------------------------------------- #
    comparisons = {}
    for rival in ("fixed_step_matched", "fixed_budget_matched", "confidence_matched",
                  "entropy_matched", "stability_matched", "random_matched",
                  "behaviour_cloning"):
        if rival in correct:
            comparisons[f"dqn_vs_{rival}"] = {
                **mt.paired_bootstrap(correct["dqn"], correct[rival],
                                      cfg.eval.bootstrap_samples, cfg.project.seed),
                **mt.mcnemar(correct["dqn"], correct[rival]),
            }

    # -- report -------------------------------------------------------------- #
    order = ["full_reasoning", "entropy_threshold", "confidence_threshold",
             "answer_stability", "fixed_budget", "behaviour_cloning", "oracle",
             "random_matched", "fixed_step_matched", "fixed_budget_matched",
             "confidence_matched", "entropy_matched", "stability_matched", "dqn"]
    natural = {"full_reasoning", "entropy_threshold", "confidence_threshold",
               "answer_stability", "fixed_budget", "behaviour_cloning", "oracle"}

    print()
    print("=" * 86)
    print("  PHASE 6 - ALL POLICIES, 599 HELD-OUT TEST QUESTIONS")
    print("=" * 86)
    print(f"  {'policy':<24}{'accuracy':>10}{'95% CI':>16}{'tokens':>9}"
          f"{'saved %':>9}{'latency':>10}{'energy':>9}")
    print("  " + "-" * 82)
    banner_done = False
    for name in order:
        if name not in results:
            continue
        if name not in natural and not banner_done:
            print()
            print(f"  at the DQN's cost ({target:.0f}% of tokens saved):")
            banner_done = True
        m = results[name]
        ci = f"[{m['accuracy_ci'][0]:.3f},{m['accuracy_ci'][1]:.3f}]"
        flag = "" if m.get("cost_matched", True) else "  << cannot reach this cost"
        print(f"  {name:<24}{m['accuracy']:>10.3f}{ci:>16}{m['mean_tokens']:>9.0f}"
              f"{m['token_reduction_pct']:>9.1f}{m['latency_seconds']:>9.2f}s"
              f"{m['energy_joules']:>8.0f}J{flag}")

    print()
    print("  DQN VERSUS EACH RIVAL - paired bootstrap and exact McNemar")
    print("  " + "-" * 82)
    for key, c in comparisons.items():
        rival = key.replace("dqn_vs_", "")
        verdict = "significant" if c["p_value"] < 0.05 else "not significant"
        gap = results.get(rival, {}).get("cost_gap_pct", 0.0)
        if not results.get(rival, {}).get("cost_matched", True):
            side = "spends less" if gap > 0 else "spends more"
            verdict += f" (unmatched: rival {side}, {gap:+.1f} pts of budget)"
        print(f"  vs {rival:<24} {c['difference']:+.3f}  "
              f"CI [{c['ci_low']:+.3f}, {c['ci_high']:+.3f}]  "
              f"p={c['p_value']:.4f}  {verdict}")

    payload = {
        "results": results,
        "comparisons": comparisons,
        "tuned": {"confidence_tau": tau_c, "entropy_tau": tau_e,
                  "stability_steps": int(n_stab), "fixed_budget": int(budget)},
        "matched": {"target_token_reduction_pct": target,
                    "fixed_step": matched_k, "fixed_budget": int(matched_b),
                    "confidence_tau": matched_c, "entropy_tau": matched_e,
                    "stability_steps": int(matched_s), "random_p": matched_p},
        "config": {"selection": cfg.rl.selection.objective,
                   "difficulty_source": cfg.rl.difficulty_source},
    }
    out = paths.RESULTS / "phase6_summary.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\n  wrote {out}")
    print("=" * 86)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
