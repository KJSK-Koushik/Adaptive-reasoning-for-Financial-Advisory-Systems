"""Rebuild traces.parquet from checkpoint shards after an interrupted run.

    python scripts/consolidate_shards.py --dry-run
    python scripts/consolidate_shards.py

Phase 3 writes a shard every ``traces.checkpoint_every`` questions and consolidates
them into traces.parquet only when the whole run finishes. A run killed by Kaggle's
12-hour limit therefore leaves a complete, usable set of shards and no final file.

This recovers those shards. A partial run is still a valid dataset - it is simply
smaller, and the splits are recomputed over whatever was actually traced, so nothing
downstream sees questions that were never generated.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from adaptive_reasoning import paths  # noqa: E402
from adaptive_reasoning.traces.runner import SHARD_DIR, TRACE_SUMMARY, consolidate  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="rebuild traces from checkpoint shards")
    ap.add_argument("--dry-run", action="store_true", help="report and write nothing")
    ap.add_argument("--shards", type=Path, default=None,
                    help="shard directory, if not artifacts/traces/_shards")
    args = ap.parse_args()

    import pandas as pd

    source = args.shards or SHARD_DIR
    if source != SHARD_DIR:
        if not source.exists():
            raise SystemExit(f"{source} does not exist")
        # Clear first. A leftover shard from an earlier or smaller run would be merged
        # in silently, putting questions traced under a different configuration into
        # the same dataset.
        stale = sorted(SHARD_DIR.glob("*.parquet")) if SHARD_DIR.exists() else []
        if stale:
            print(f"removing {len(stale)} shard files already in {SHARD_DIR}")
            for f in stale:
                f.unlink()
        SHARD_DIR.mkdir(parents=True, exist_ok=True)
        copied = sorted(source.glob("*.parquet"))
        for f in copied:
            shutil.copy2(f, SHARD_DIR / f.name)
        print(f"copied {len(copied)} shard files from {source}")

    steps = sorted(SHARD_DIR.glob("steps_*.parquet"))
    summaries = sorted(SHARD_DIR.glob("summary_*.parquet"))
    if not steps:
        raise SystemExit(
            f"no shards in {SHARD_DIR}. Download the notebook output and place the "
            f"_shards directory there, or pass --shards <path>."
        )

    frames = []
    unreadable = 0
    for shard in summaries:
        try:
            frames.append(pd.read_parquet(shard))
        except Exception:                                        # noqa: BLE001
            unreadable += 1               # a shard interrupted mid-write

    summary = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    print(f"  shard files       {len(steps)} steps, {len(summaries)} summaries")
    if unreadable:
        print(f"  unreadable        {unreadable} (interrupted mid-write, skipped)")
    print(f"  questions traced  {len(summary):,}")

    if len(summary):
        trunc = (summary.total_tokens >= summary.total_tokens.max()).mean()
        print(f"  final accuracy    {summary.final_correct.mean() * 100:.1f}%")
        print(f"  ever correct      "
              f"{summary.earliest_correct_step.notna().mean() * 100:.1f}%")
        print(f"  mean tokens       {summary.total_tokens.mean():.0f}")
        print(f"  mean steps        {summary.n_steps.mean():.1f}")
        print(f"  at the cap        {trunc * 100:.1f}%")

    if args.dry_run:
        print("\n  dry run - nothing written")
        return 0

    for path in (paths.TRACE_DATASET, TRACE_SUMMARY):
        if path.exists():
            backup = path.with_suffix(path.suffix + ".before_consolidate")
            if not backup.exists():
                shutil.copy2(path, backup)
                print(f"  backed up {path.name} -> {backup.name}")

    consolidate()
    print(f"\n  wrote {paths.TRACE_DATASET}")
    print(f"  wrote {TRACE_SUMMARY}")
    print("\n  now rerun: run_phase4 -> run_phase5 -> run_phase6 -> run_phase7")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
