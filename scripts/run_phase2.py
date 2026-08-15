"""Phase 2 - difficulty labelling and classifier.

Three stages, run separately because the middle one needs a GPU:

    python scripts/run_phase2.py --stage sample --experiment smoke --limit 12
    python scripts/run_phase2.py --stage label
    python scripts/run_phase2.py --stage train

``sample`` is the GPU-bound step (Phase 2b) and is bundled with Phase 3 on Kaggle.
``label`` and ``train`` are CPU-only and take seconds.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from adaptive_reasoning import paths  # noqa: E402
from adaptive_reasoning.config import load_config, parse_overrides  # noqa: E402
from adaptive_reasoning.data import build as build_mod  # noqa: E402
from adaptive_reasoning.data.io import load_unified, stratified_sample  # noqa: E402
from adaptive_reasoning.difficulty import classifier as clf_mod  # noqa: E402
from adaptive_reasoning.difficulty import from_traces, labeling, sampling  # noqa: E402
from adaptive_reasoning.logging_utils import setup_logging  # noqa: E402
from adaptive_reasoning.schema import Difficulty  # noqa: E402
from adaptive_reasoning.seeding import set_seed  # noqa: E402


def stage_sample(cfg, args) -> int:
    """Run k samples per question. GPU-bound."""
    records = load_unified()
    n = args.limit or cfg.difficulty.n_questions
    if n:
        records = stratified_sample(records, n, cfg.project.seed, by="source")

    print(f"sampling {len(records)} questions, k={cfg.difficulty.k_samples}")
    outcomes = sampling.sample_questions(records, cfg, resume=not args.no_resume)
    sampling.write_manifest(cfg, len(records), len(outcomes))
    print(f"produced {len(outcomes)} sample outcomes -> {sampling.SAMPLES_PATH}")
    return 0


def _verdicts_from_traces(cfg):
    """Derive labels from Phase 3 traces - no GPU needed."""
    import pandas as pd

    from adaptive_reasoning.traces.runner import TRACE_SUMMARY

    steps = pd.read_parquet(paths.TRACE_DATASET)
    summary = pd.read_parquet(TRACE_SUMMARY)
    print(f"deriving difficulty from {len(summary):,} traces ({len(steps):,} probe points)")
    return from_traces.label_all_from_traces(steps, summary, cfg)


def _verdicts_from_samples(cfg):
    """Derive labels from k independent samples - the original, GPU-expensive route."""
    outcomes = sampling.load_outcomes()
    if not outcomes:
        return None
    print(f"deriving difficulty from {len(outcomes):,} sample outcomes")
    return labeling.label_all(outcomes, cfg)


def stage_label(cfg, args) -> int:
    """Turn trace probes (or k samples) into difficulty labels and write them back."""
    import pandas as pd

    from adaptive_reasoning.traces.runner import TRACE_SUMMARY

    source = args.source
    if source == "auto":
        have_traces = paths.TRACE_DATASET.exists() and TRACE_SUMMARY.exists()
        source = "traces" if have_traces else "samples"

    if source == "traces":
        if not (paths.TRACE_DATASET.exists() and TRACE_SUMMARY.exists()):
            raise SystemExit(
                f"no traces at {paths.TRACE_DATASET} - run scripts/run_phase3.py first"
            )
        verdicts = _verdicts_from_traces(cfg)
    else:
        verdicts = _verdicts_from_samples(cfg)
        if verdicts is None:
            raise SystemExit(
                f"no sample outcomes at {sampling.SAMPLES_PATH} - run --stage sample "
                f"first, or use --source traces"
            )

    dist = labeling.distribution(verdicts)

    frame = pd.DataFrame(
        [
            {
                "question_id": v.question_id,
                "difficulty": str(v.difficulty),
                "difficulty_score": v.score,
                "pass_rate": v.pass_rate,
                "n_samples": v.n_samples,
                "median_reasoning_tokens": v.median_reasoning_tokens,
                "answer_diversity": v.answer_diversity,
                "reason": v.reason,
            }
            for v in verdicts
        ]
    )
    paths.DIFFICULTY_LABELS.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(paths.DIFFICULTY_LABELS, index=False)

    # Merge into the unified dataset so downstream phases see one file. Only
    # `difficulty`/`difficulty_score` are touched - `difficulty_prior` is a different
    # notion and is left intact for the Phase 9 comparison.
    unified = pd.read_parquet(paths.UNIFIED_DATASET)
    measured = frame[["question_id", "difficulty", "difficulty_score"]]
    merged = (
        unified.drop(columns=["difficulty", "difficulty_score"], errors="ignore")
        .merge(measured, left_on="id", right_on="question_id", how="left")
        .drop(columns=["question_id"])
    )

    # Re-split over the labelled subset. Phase 1 split all ~30k questions, but only
    # these few thousand were traced, and the sample did not inherit the proportions.
    merged = build_mod.resplit_subset(merged, merged["difficulty"].notna(), cfg)

    merged.to_parquet(paths.UNIFIED_DATASET, index=False)

    print("=" * 68)
    print("  DIFFICULTY LABELS")
    print("=" * 68)
    print(f"  source             {source}")
    print(f"  labelled           {len(verdicts):,} questions")
    for tier in ("easy", "medium", "hard"):
        count = dist.get(tier, 0)
        pct = 100 * count / max(len(verdicts), 1)
        print(f"    {tier:<8} {count:>7,}  ({pct:.1f}%)")

    label = "mean correct-probe ratio" if source == "traces" else "mean pass rate"
    print(f"\n  {label:<26} {frame.pass_rate.mean():.3f}")
    print(f"  median tokens              {frame.median_reasoning_tokens.median():.0f}")
    print(f"  mean answer instability    {frame.answer_diversity.mean():.3f}")

    # A degenerate distribution means the thresholds need re-tuning. Cheap to spot now,
    # expensive to discover after a classifier and a DQN have been trained on it.
    # Both checks matter: a dominant tier and a *collapsed* one are different failures,
    # and the first real run had 78% hard with medium down at 7.5%.
    total = max(len(verdicts), 1)
    shares = {tier: dist.get(tier, 0) / total for tier in ("easy", "medium", "hard")}
    dominant = max(shares.values())
    starved = [tier for tier, share in shares.items() if share < 0.10]

    if dominant > 0.70 or starved:
        print("\n  WARNING: the tier distribution looks degenerate.")
        if dominant > 0.70:
            worst = max(shares, key=shares.get)
            print(f"    - {dominant:.0%} of questions are '{worst}'")
        for tier in starved:
            print(f"    - '{tier}' has only {shares[tier]:.1%} of questions")
        print("  Re-tune difficulty.from_traces thresholds against the observed")
        print("  distribution and re-run - this is a seconds-long CPU job, no GPU.")

    print(f"\n  wrote {paths.DIFFICULTY_LABELS}")
    print(f"  merged into {paths.UNIFIED_DATASET}")
    print("=" * 68)
    return 0


def stage_train(cfg, args) -> int:
    """Train the text-only difficulty classifier."""
    import pandas as pd

    frame = pd.read_parquet(paths.UNIFIED_DATASET)
    labelled = frame[frame.difficulty.notna()]
    if labelled.empty:
        raise SystemExit("no labelled rows - run --stage label first")

    train = labelled[labelled.split == "train"]
    evaluation = labelled[labelled.split.isin(["val", "test"])]
    if train.empty:
        train, evaluation = labelled, labelled

    model = clf_mod.DifficultyClassifier(cfg)
    report = model.fit(
        train.question.tolist(),
        train.context.fillna("").tolist(),
        [Difficulty(d) for d in train.difficulty],
        eval_split=(
            evaluation.question.tolist(),
            evaluation.context.fillna("").tolist(),
            [Difficulty(d) for d in evaluation.difficulty],
        ) if not evaluation.empty else None,
    )
    model.save()
    clf_mod.write_report(report)

    print("=" * 68)
    print("  DIFFICULTY CLASSIFIER")
    print("=" * 68)
    print(f"  trained on         {report.n_train:,}")
    print(f"  evaluated on       {report.n_eval:,}")
    print(f"  accuracy           {report.accuracy:.3f}")
    print(f"  macro F1           {report.macro_f1:.3f}")
    print(f"  majority baseline  {report.baseline_majority:.3f}  <- must beat this")
    print("\n  per class:")
    for name, m in report.per_class.items():
        print(f"    {name:<8} P={m['precision']:.3f}  R={m['recall']:.3f}  "
              f"F1={m['f1']:.3f}  n={m['support']}")
    print("\n  confusion (rows = true, cols = predicted, order easy/medium/hard):")
    for name, row in zip(["easy", "medium", "hard"], report.confusion, strict=True):
        print(f"    {name:<8} {row}")
    print(f"\n  wrote {paths.DIFFICULTY_MODEL}")
    print("=" * 68)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 2 - difficulty labelling")
    parser.add_argument("--stage", required=True, choices=["sample", "label", "train"])
    parser.add_argument("--experiment", help="config in configs/experiment/")
    parser.add_argument("--limit", type=int, help="cap questions (sampling only)")
    parser.add_argument("--no-resume", action="store_true", help="ignore existing shards")
    parser.add_argument(
        "--source", choices=["auto", "traces", "samples"], default="auto",
        help="where difficulty comes from (label stage): Phase 3 traces (free) or "
             "k independent samples (GPU-expensive). Default picks traces if present.",
    )
    parser.add_argument(
        "--set", dest="overrides", action="append", metavar="KEY=VALUE",
        help="config override, e.g. --set difficulty.k_samples=3 (repeatable)",
    )
    args = parser.parse_args()

    cfg = load_config(args.experiment, parse_overrides(args.overrides))
    setup_logging(cfg.logging.level, cfg.logging.to_file, cfg.logging.rich_console,
                  f"phase2-{args.stage}")
    set_seed(cfg.project.seed)
    paths.ensure_dirs()

    return {"sample": stage_sample, "label": stage_label, "train": stage_train}[args.stage](cfg, args)


if __name__ == "__main__":
    raise SystemExit(main())
