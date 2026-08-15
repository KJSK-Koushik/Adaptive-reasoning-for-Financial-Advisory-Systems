"""Phase 3 - reasoning trace generation.

Run the pilot gate FIRST; this script refuses to start if the gate has not passed.

    python scripts/run_pilot.py
    python scripts/run_phase3.py

    python scripts/run_phase3.py --experiment smoke --limit 4   # local plumbing check
    python scripts/run_phase3.py --skip-gate                    # only if you know why
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from adaptive_reasoning import paths  # noqa: E402
from adaptive_reasoning.config import load_config, parse_overrides  # noqa: E402
from adaptive_reasoning.data.io import load_unified, stratified_sample  # noqa: E402
from adaptive_reasoning.logging_utils import setup_logging  # noqa: E402
from adaptive_reasoning.seeding import set_seed  # noqa: E402
from adaptive_reasoning.traces import runner  # noqa: E402


def _gate_passed() -> bool | None:
    """Read the pilot verdict. ``None`` means the pilot has never been run."""
    path = paths.RESULTS / "phase3_pilot.json"
    if not path.exists():
        return None
    try:
        return bool(json.loads(path.read_text(encoding="utf-8")).get("passed"))
    except (json.JSONDecodeError, OSError):
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 3 - generate reasoning traces")
    parser.add_argument("--experiment", help="config in configs/experiment/")
    parser.add_argument("--limit", type=int, help="cap the number of questions")
    parser.add_argument("--no-resume", action="store_true", help="ignore existing shards")
    parser.add_argument("--skip-gate", action="store_true", help="bypass the pilot gate")
    parser.add_argument(
        "--set", dest="overrides", action="append", metavar="KEY=VALUE",
        help="config override, e.g. --set llm.batch_size=32 (repeatable)",
    )
    args = parser.parse_args()

    cfg = load_config(args.experiment, parse_overrides(args.overrides))
    setup_logging(cfg.logging.level, cfg.logging.to_file, cfg.logging.rich_console, "phase3")
    set_seed(cfg.project.seed)
    paths.ensure_dirs()

    if not args.skip_gate:
        gate = _gate_passed()
        if gate is None:
            print("The pilot has not been run. Phase 3 costs hours of GPU time and is")
            print("worthless if the model does not reason at length.\n")
            print("    python scripts/run_pilot.py\n")
            print("Use --skip-gate to override.")
            return 2
        if not gate:
            print("The pilot gate FAILED - see artifacts/results/phase3_pilot.json.")
            print("Fix the model or prompt before spending GPU time, or use --skip-gate.")
            return 2

    records = load_unified()
    n = args.limit or cfg.traces.n_questions
    if n:
        records = stratified_sample(records, n, cfg.project.seed, by="source")

    print(f"tracing {len(records)} questions with {cfg.llm.model_id}")
    summary = runner.run(records, cfg, resume=not args.no_resume)

    print()
    print("=" * 68)
    print("  PHASE 3 SUMMARY")
    print("=" * 68)
    for key, value in summary.items():
        print(f"  {key:<28} {value}")
    if "oracle_token_saving_pct" in summary:
        print()
        print(f"  The oracle stops {summary['oracle_token_saving_pct']}% earlier than full")
        print("  reasoning. That is the headroom the DQN is competing for - a policy")
        print("  capturing most of it is a strong result.")
    print(f"\n  wrote {paths.TRACE_DATASET}")
    print("=" * 68)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
