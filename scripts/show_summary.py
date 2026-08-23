"""Print a one-screen summary of everything the pipeline has produced so far.

    python scripts/show_summary.py

Reads only the JSON summaries written by phases 1-5, so it is instant and safe to
run at any time. Useful for reviews and for checking that a rerun actually changed
what you expected it to change.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from adaptive_reasoning import paths  # noqa: E402


def _load(name: str) -> dict | None:
    path = paths.RESULTS / name
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    p1 = _load("phase1_summary.json")
    p3 = _load("phase3_summary.json")
    p4 = _load("phase4_summary.json")
    p5 = _load("phase5_summary.json")

    if p1:
        print("PHASE 1  unified dataset")
        print(f"  {p1['total']:,} questions from {len(p1['by_source'])} sources, "
              f"{p1['duplicates_removed']} duplicates removed")
        for source, count in sorted(p1["by_source"].items(), key=lambda kv: -kv[1]):
            print(f"    {source:<14} {count:>6,}")
        split = p1["by_split"]
        print(f"  splits: train {split['train']:,}  val {split['val']:,}  "
              f"test {split['test']:,}")
        print()

    if p3:
        print("PHASE 3  reasoning traces (GPU, DeepSeek-R1-Distill-Qwen-1.5B)")
        print(f"  traces generated       {p3['n_traces']:,}")
        print(f"  final answer correct   {p3['final_accuracy'] * 100:.1f}%")
        print(f"  correct at some point  {p3['solvable_fraction'] * 100:.1f}%"
              f"   <-- the overthinking gap")
        print(f"  mean reasoning tokens  {p3['mean_total_tokens']:.0f} over "
              f"{p3['mean_steps']:.1f} steps")
        print(f"  oracle stop would use  {p3['oracle_mean_tokens']:.0f} tokens "
              f"({p3['oracle_token_saving_pct']:.1f}% saved)")
        print()

    if p4:
        print("PHASE 4  offline RL dataset")
        print(f"  transitions            {p4['n_transitions']:,}")
        print(f"  state dimension        {p4['state_dim']}")
        split = p4["by_split"]
        print(f"  by split               train {split['train']:,}  "
              f"val {split['val']:,}  test {split['test']:,}")
        print()

    if p5:
        print("PHASE 5  stopping policies (599 held-out test questions)")
        print(f"  {'policy':<22} {'accuracy':>9} {'tokens':>8} {'saved %':>9}")
        for name in ("full_reasoning", "fixed_step_matched", "behaviour_cloning",
                     "dqn", "oracle"):
            row = p5["results"].get(name)
            if row:
                print(f"  {name:<22} {row['accuracy']:>9.3f} "
                      f"{row['mean_tokens']:>8.0f} {row['token_reduction_pct']:>9.1f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
