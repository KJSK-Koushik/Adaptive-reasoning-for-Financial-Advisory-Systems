"""Phase 4 - build the offline RL dataset from reasoning traces.

    python scripts/run_phase4.py
    python scripts/run_phase4.py --set rl.difficulty_source=none    # ablation

CPU only, seconds to run. Produces artifacts/traces/transitions.parquet.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from adaptive_reasoning import paths  # noqa: E402
from adaptive_reasoning.config import load_config, parse_overrides  # noqa: E402
from adaptive_reasoning.logging_utils import setup_logging  # noqa: E402
from adaptive_reasoning.rl import dataset  # noqa: E402
from adaptive_reasoning.seeding import set_seed  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 4 - offline RL dataset")
    parser.add_argument("--experiment", help="config in configs/experiment/")
    parser.add_argument("--dry-run", action="store_true", help="build but do not write")
    parser.add_argument(
        "--set", dest="overrides", action="append", metavar="KEY=VALUE",
        help="config override, e.g. --set rl.difficulty_source=none (repeatable)",
    )
    args = parser.parse_args()

    cfg = load_config(args.experiment, parse_overrides(args.overrides))
    setup_logging(cfg.logging.level, cfg.logging.to_file, cfg.logging.rich_console, "phase4")
    set_seed(cfg.project.seed)
    paths.ensure_dirs()

    stats = dataset.build(cfg, write=not args.dry_run)

    print()
    print("=" * 68)
    print("  PHASE 4 SUMMARY")
    print("=" * 68)
    print(f"  traces                  {stats['n_traces']:,}")
    print(f"  transitions             {stats['n_transitions']:,}")
    print(f"  state dimension         {stats['state_dim']}")
    print(f"  token budget            {stats['token_budget']}")
    print(f"  difficulty source       {cfg.rl.difficulty_source}")
    print()
    print(f"  mean STOP reward        {stats['stop_reward_mean']:+.4f}")
    print(f"  mean CONTINUE reward    {stats['continue_reward_mean']:+.4f}")
    print(f"  mean oracle return      {stats['mean_oracle_return']:+.4f}")
    print("\n  transitions by split:")
    for split, count in sorted(stats["by_split"].items()):
        print(f"    {split:<10} {count:>9,}")
    print("\n  transitions by difficulty:")
    for tier, count in sorted(stats["by_difficulty"].items(), key=lambda kv: str(kv[0])):
        print(f"    {str(tier):<10} {count:>9,}")

    if not args.dry_run:
        print(f"\n  wrote {paths.RL_TRANSITIONS}")
    print("=" * 68)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
