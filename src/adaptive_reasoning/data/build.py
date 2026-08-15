"""Phase 1 builder: load every source, unify, deduplicate, sample and split.

Output is ``data/processed/unified.parquet`` plus a JSON summary in
``artifacts/results/phase1_summary.json``.

Two details worth knowing:

**Deduplication is on the question+context pair**, not the id. FinQA and ConvFinQA
draw on the same underlying filings, and TAT-QA repeats question phrasings across
tables. Near-identical items appearing in both train and test would inflate every
number downstream.

**Splitting is stratified by (source, domain, answer_type)** rather than by
difficulty, because difficulty does not exist yet - Phase 2 assigns it. Stratifying on
source keeps the easy/hard mix comparable across splits in the meantime.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter, defaultdict

from .. import paths
from ..config import Config
from ..logging_utils import get_logger
from ..schema import UNIFIED_COLUMNS, QARecord, Split
from . import synthetic
from .adapters import ADAPTERS

log = get_logger("data.build")


def _fingerprint(record: QARecord) -> str:
    """Hash of the normalised question+context, used for deduplication."""
    key = f"{record.question.strip().lower()}||{record.context.strip().lower()}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()


def load_all(cfg: Config) -> tuple[list[QARecord], dict[str, int]]:
    """Run every adapter plus the synthetic generator. Missing sources are skipped."""
    records: list[QARecord] = []
    counts: dict[str, int] = {}

    for name, module in ADAPTERS.items():
        try:
            loaded = module.load(cfg)
        except Exception as exc:  # noqa: BLE001 - one bad source must not kill the run
            log.error("adapter %s failed: %s: %s", name, type(exc).__name__, exc)
            loaded = []
        counts[name] = len(loaded)
        records.extend(loaded)

    loaded = synthetic.load(cfg)
    counts["synthetic"] = len(loaded)
    records.extend(loaded)

    return records, counts


def deduplicate(records: list[QARecord]) -> tuple[list[QARecord], int]:
    seen: set[str] = set()
    kept: list[QARecord] = []
    for record in records:
        fp = _fingerprint(record)
        if fp in seen:
            continue
        seen.add(fp)
        kept.append(record)
    return kept, len(records) - len(kept)


def subsample(records: list[QARecord], cfg: Config) -> list[QARecord]:
    """Apply per-source caps from ``data.sample_sizes``. ``None`` keeps everything."""
    rng = random.Random(cfg.project.seed)
    by_source: dict[str, list[QARecord]] = defaultdict(list)
    for record in records:
        by_source[record.source].append(record)

    out: list[QARecord] = []
    for source, items in by_source.items():
        cap = cfg.data.sample_sizes.get(source)
        if cap is not None and len(items) > cap:
            items = rng.sample(items, cap)
            log.info("%s: capped to %d", source, cap)
        out.extend(items)
    return out


def assign_splits(records: list[QARecord], cfg: Config) -> list[QARecord]:
    """Stratified train/val/test assignment.

    Difficulty is not available yet (Phase 2 produces it), so we stratify on the
    fields that exist and that correlate with it.
    """
    rng = random.Random(cfg.project.seed)
    strata: dict[tuple, list[QARecord]] = defaultdict(list)
    for record in records:
        strata[(record.source, record.domain, record.answer_type)].append(record)

    train_frac, val_frac = cfg.data.splits.train, cfg.data.splits.val
    for items in strata.values():
        rng.shuffle(items)
        n = len(items)
        n_train = int(n * train_frac)
        n_val = int(n * (train_frac + val_frac)) - n_train
        for i, record in enumerate(items):
            if i < n_train:
                record.split = Split.TRAIN
            elif i < n_train + n_val:
                record.split = Split.VAL
            else:
                record.split = Split.TEST
    return records


def resplit_subset(frame, mask, cfg: Config, salt: str = "resplit"):
    """Reassign train/val/test over a subset of rows, stratified by source+difficulty.

    Phase 1 splits all ~30k questions, but only a few thousand get traced, and that
    sample is not guaranteed to inherit the proportions - the first real run produced
    an 87/7/5 split of the traced subset. Every phase from 5 onward sees only traced
    questions, so the split that matters is the one *over them*.

    Reassigning here also keeps the difficulty classifier and the DQN on the same
    split, so a question the classifier trained on cannot end up in the policy's test
    set carrying an optimistic difficulty prediction.
    """
    import random

    rng = random.Random(f"{cfg.project.seed}:{salt}")
    subset = frame[mask]
    if subset.empty:
        return frame

    strata: dict[tuple, list[int]] = defaultdict(list)
    for index, row in subset.iterrows():
        strata[(row.get("source"), row.get("difficulty"))].append(index)

    train_frac, val_frac = cfg.data.splits.train, cfg.data.splits.val
    assignments: dict[int, str] = {}

    for indices in strata.values():
        rng.shuffle(indices)
        n = len(indices)
        n_train = int(round(n * train_frac))
        n_val = int(round(n * (train_frac + val_frac))) - n_train
        for position, index in enumerate(indices):
            if position < n_train:
                assignments[index] = "train"
            elif position < n_train + n_val:
                assignments[index] = "val"
            else:
                assignments[index] = "test"

    frame = frame.copy()
    for index, split in assignments.items():
        frame.at[index, "split"] = split

    counts = Counter(assignments.values())
    log.info("re-split %d labelled questions: %s", len(assignments), dict(counts))
    return frame


def to_dataframe(records: list[QARecord]):
    import pandas as pd

    frame = pd.DataFrame([r.model_dump() for r in records])
    for column in UNIFIED_COLUMNS:
        if column not in frame.columns:
            frame[column] = None
    return frame[UNIFIED_COLUMNS]


def summarise(records: list[QARecord]) -> dict:
    return {
        "total": len(records),
        "by_source": dict(Counter(r.source for r in records)),
        "by_domain": dict(Counter(str(r.domain) for r in records)),
        "by_answer_type": dict(Counter(str(r.answer_type) for r in records)),
        "by_split": dict(Counter(str(r.split) for r in records)),
        "mean_question_chars": round(
            sum(len(r.question) for r in records) / max(len(records), 1), 1
        ),
        "mean_context_chars": round(
            sum(len(r.context) for r in records) / max(len(records), 1), 1
        ),
    }


def build(cfg: Config, write: bool = True) -> tuple[list[QARecord], dict]:
    """Run the whole Phase 1 pipeline."""
    records, raw_counts = load_all(cfg)
    log.info("loaded %d raw records from %d sources", len(records), len(raw_counts))

    records, n_dupes = deduplicate(records)
    log.info("deduplicated: removed %d, %d remain", n_dupes, len(records))

    records = subsample(records, cfg)
    records = assign_splits(records, cfg)

    summary = summarise(records)
    summary["raw_counts"] = raw_counts
    summary["duplicates_removed"] = n_dupes

    if write:
        paths.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
        frame = to_dataframe(records)
        frame.to_parquet(paths.UNIFIED_DATASET, index=False)
        log.info("wrote %s (%d rows)", paths.UNIFIED_DATASET, len(frame))

        paths.RESULTS.mkdir(parents=True, exist_ok=True)
        summary_path = paths.RESULTS / "phase1_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        log.info("wrote %s", summary_path)

    return records, summary
