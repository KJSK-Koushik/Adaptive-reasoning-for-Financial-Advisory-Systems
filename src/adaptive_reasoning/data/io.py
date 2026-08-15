"""Reading the unified dataset back into typed records.

Used by Phases 2, 3 and 6. Parquet round-trips list columns as numpy arrays and
missing values as ``NaN``, neither of which pydantic accepts, so the conversion is
done once here rather than being reinvented (differently) in each phase.
"""

from __future__ import annotations

from pathlib import Path

from .. import paths
from ..logging_utils import get_logger
from ..schema import QARecord

log = get_logger("data.io")

_OPTIONAL = {"difficulty", "difficulty_score", "split"}


def _scalar(value):
    """Normalise a parquet cell into something pydantic accepts."""
    import numpy as np
    import pandas as pd

    if value is None:
        return None
    if isinstance(value, (np.ndarray, list, tuple)):
        return [str(v) for v in value]
    if isinstance(value, float) and pd.isna(value):
        return None
    if value is pd.NaT:
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def load_unified(path: Path | None = None, require_difficulty: bool = False) -> list[QARecord]:
    """Load ``unified.parquet`` as :class:`QARecord` objects."""
    import pandas as pd

    path = path or paths.UNIFIED_DATASET
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found - run `python scripts/run_phase1.py` first"
        )

    frame = pd.read_parquet(path)
    if require_difficulty:
        before = len(frame)
        frame = frame[frame["difficulty"].notna()]
        if frame.empty:
            raise ValueError(
                f"{path} has no difficulty labels - run "
                f"`python scripts/run_phase2.py --stage label` first"
            )
        log.info("using %d of %d rows that carry a difficulty label", len(frame), before)

    records = []
    for row in frame.to_dict("records"):
        clean = {key: _scalar(value) for key, value in row.items()}
        clean["answer_options"] = clean.get("answer_options") or []
        records.append(QARecord(**clean))

    log.info("loaded %d records from %s", len(records), path.name)
    return records


def stratified_sample(
    records: list[QARecord], n: int, seed: int, by: str = "source",
    salt: str = "stratified_sample",
) -> list[QARecord]:
    """Take ``n`` records spread evenly across the values of ``by``.

    Used to choose which questions get sampled (Phase 2) and traced (Phase 3). An
    unstratified sample would be dominated by FinQA and would leave the policy with
    almost no easy-tier examples to learn the "stop immediately" behaviour from.

    .. note::
       The RNG is salted rather than seeded directly with ``seed``. ``assign_splits``
       in Phase 1 also shuffles group-by-group from ``random.Random(seed)``; with the
       same seed and the same iteration order, the questions ranked first for sampling
       were the same ones ranked first for the train split. The first real trace run
       came out 87.5% train / 7.4% val / 5.1% test instead of 70/15/15, leaving only
       205 test questions. Different streams, no correlation.
    """
    import random

    if n >= len(records):
        return list(records)

    rng = random.Random(f"{seed}:{salt}")
    groups: dict[str, list[QARecord]] = {}
    for record in records:
        groups.setdefault(str(getattr(record, by)), []).append(record)

    per_group = max(1, n // len(groups))
    picked: list[QARecord] = []
    leftovers: list[QARecord] = []

    for items in groups.values():
        rng.shuffle(items)
        picked.extend(items[:per_group])
        leftovers.extend(items[per_group:])

    # Top up from the leftovers when small groups could not fill their quota.
    if len(picked) < n:
        rng.shuffle(leftovers)
        picked.extend(leftovers[: n - len(picked)])

    rng.shuffle(picked)
    return picked[:n]
