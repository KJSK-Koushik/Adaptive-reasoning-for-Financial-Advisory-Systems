"""Pre-flight check that gates Phase 3.

Run this FIRST on whatever GPU will do the real work. It takes a couple of minutes and
refuses to let Phase 3 proceed if the model does not actually reason at length.

    python scripts/run_pilot.py                      # default model, 50 questions
    python scripts/run_pilot.py --model <hf-id>      # try a different model
    python scripts/run_pilot.py --experiment smoke   # tiny model, CPU (expected to FAIL)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from adaptive_reasoning import paths, pilot  # noqa: E402
from adaptive_reasoning.config import load_config, parse_overrides  # noqa: E402
from adaptive_reasoning.data.io import load_unified, stratified_sample  # noqa: E402
from adaptive_reasoning.llm import ReasoningLLM  # noqa: E402
from adaptive_reasoning.logging_utils import setup_logging  # noqa: E402
from adaptive_reasoning.seeding import set_seed  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 3 pre-flight pilot")
    parser.add_argument("--experiment", help="config in configs/experiment/")
    parser.add_argument("--model", help="override llm.model_id")
    parser.add_argument("--n", type=int, help="override the number of pilot questions")
    parser.add_argument(
        "--set", dest="overrides", action="append", metavar="KEY=VALUE",
        help="config override, e.g. --set llm.batch_size=32 (repeatable)",
    )
    args = parser.parse_args()

    cfg = load_config(args.experiment, parse_overrides(args.overrides))
    setup_logging(cfg.logging.level, cfg.logging.to_file, cfg.logging.rich_console, "pilot")
    set_seed(cfg.project.seed)
    paths.ensure_dirs()

    n = args.n or cfg.traces.pilot.n_questions
    records = stratified_sample(load_unified(), n, cfg.project.seed, by="source")

    llm = ReasoningLLM(cfg, model_id=args.model)
    report = pilot.run_pilot(records, cfg, llm=llm)
    pilot.write_report(report)
    pilot.print_report(report)

    return 0 if report.passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
