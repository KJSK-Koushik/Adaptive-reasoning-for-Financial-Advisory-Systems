"""Re-grade recorded traces after a change to the grading rules.

    python scripts/regrade_traces.py --dry-run
    python scripts/regrade_traces.py

The probe answers the model produced are recorded, so correctness can be recomputed
without a GPU. Only ``probe_correct`` and the derived summary change; not one token of
generated text is touched.

Run this whenever ``grading.py`` changes, then rerun Phases 4-7 - every downstream
number depends on which answers count as right.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from adaptive_reasoning import paths  # noqa: E402
from adaptive_reasoning.grading import is_correct  # noqa: E402
from adaptive_reasoning.logging_utils import get_logger, setup_logging  # noqa: E402
from adaptive_reasoning.traces.runner import TRACE_SUMMARY  # noqa: E402

log = get_logger("regrade")


def main() -> int:
    parser = argparse.ArgumentParser(description="re-grade recorded traces")
    parser.add_argument("--dry-run", action="store_true",
                        help="report what would change and write nothing")
    args = parser.parse_args()
    setup_logging("INFO", False, True, "regrade")

    import pandas as pd

    if not paths.TRACE_DATASET.exists():
        raise SystemExit(f"{paths.TRACE_DATASET} not found - run Phase 3 first")

    traces = pd.read_parquet(paths.TRACE_DATASET)
    unified = pd.read_parquet(paths.UNIFIED_DATASET).set_index("id")
    gold = unified.gold_answer.to_dict()
    answer_type = unified.answer_type.to_dict()

    missing = set(traces.question_id.astype(str)) - set(gold)
    if missing:
        raise SystemExit(
            f"{len(missing)} traced questions are absent from the unified dataset, "
            "so they cannot be graded - regenerate unified.parquet first"
        )

    was = traces.probe_correct.astype(bool).to_numpy()
    now = [
        is_correct(str(row.probe_answer or ""), str(gold[str(row.question_id)]),
                   str(answer_type[str(row.question_id)]))
        for row in traces.itertuples()
    ]
    traces["probe_correct"] = now
    now_arr = traces.probe_correct.to_numpy()

    gained = int((~was & now_arr).sum())
    lost = int((was & ~now_arr).sum())
    print(f"  probe rows            {len(traces):,}")
    print(f"  newly correct         {gained:,}")
    print(f"  no longer correct     {lost:,}")
    print(f"  net change            {gained - lost:+,}")

    # Rebuild the per-question summary, which is derived entirely from probe_correct.
    summary = pd.read_parquet(TRACE_SUMMARY) if TRACE_SUMMARY.exists() else None
    rebuilt = []
    for question_id, group in traces.groupby("question_id", sort=False):
        group = group.sort_values("step_index")
        hits = group.index[group.probe_correct].tolist()
        earliest = (int(group.step_index[group.probe_correct].iloc[0])
                    if hits else None)
        rebuilt.append({
            "question_id": str(question_id),
            "total_tokens": int(group.tokens_so_far.iloc[-1]),
            "final_answer": group.probe_answer.iloc[-1],
            "final_correct": bool(group.probe_correct.iloc[-1]),
            "n_steps": len(group),
            "earliest_correct_step": earliest,
            "oracle_tokens": (int(group.tokens_so_far.iloc[earliest])
                              if earliest is not None else None),
        })
    rebuilt = pd.DataFrame(rebuilt)

    if summary is not None and "difficulty" in summary.columns:
        rebuilt = rebuilt.merge(
            summary[["question_id", "difficulty"]], on="question_id", how="left")

    n = len(rebuilt)
    print(f"\n  questions             {n:,}")
    if summary is not None:
        print(f"  final accuracy was    {summary.final_correct.mean():.4f}")
    print(f"  final accuracy now    {rebuilt.final_correct.mean():.4f}")
    solvable = rebuilt.earliest_correct_step.notna().mean()
    print(f"  ever-correct now      {solvable:.4f}")

    if args.dry_run:
        print("\n  dry run - nothing written")
        return 0

    for path in (paths.TRACE_DATASET, TRACE_SUMMARY):
        if path.exists():
            backup = path.with_suffix(path.suffix + ".pre_regrade")
            if not backup.exists():
                shutil.copy2(path, backup)
                print(f"\n  backed up {path.name} -> {backup.name}")

    traces.to_parquet(paths.TRACE_DATASET, index=False)
    rebuilt.to_parquet(TRACE_SUMMARY, index=False)
    print(f"  wrote {paths.TRACE_DATASET}")
    print(f"  wrote {TRACE_SUMMARY}")
    print("\n  now rerun: run_phase4 -> run_phase5 -> run_phase6 -> run_phase7")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
