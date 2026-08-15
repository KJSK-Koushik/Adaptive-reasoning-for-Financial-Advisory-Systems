"""FinQA adapter.

Three things about the real data drove the decisions here.

**1. ``qa.answer`` is unreliable; ``qa.exe_ans`` is the gold.**
The human-written ``answer`` string is inconsistently scaled and sometimes empty.
Example from ``train.json[0]``: ``answer='380'`` while ``exe_ans=3.8`` and the program
is ``divide(3.8, 1)``. The executed answer is what the official FinQA metric
("execution accuracy") uses, so that is what we grade against.

**2. Percentages are stored as fractions.** ``answer='53%'`` has ``exe_ans=0.53232``.
Our grader already accepts a 100x discrepancy, which covers both this and the case
above. Documented as a deliberate leniency in docs/DATASETS.md.

**3. 124 of 6,251 training rows have a non-float ``exe_ans``** - almost all
``'yes'``/``'no'``. Those become categorical questions rather than being discarded.

Context uses ``qa.gold_inds`` (the annotated evidence rows) when
``data.use_gold_evidence`` is set. This is the standard FinQA "gold evidence"
setting: it isolates reasoning from retrieval, which is exactly our research
question, and it cuts prompt length by roughly an order of magnitude.
"""

from __future__ import annotations

import json

from ... import paths
from ...config import Config
from ...logging_utils import get_logger
from ...schema import AnswerType, Domain, QARecord
from ..text_utils import build_context, clean, format_number, render_table

log = get_logger("data.finqa")

SPLIT_FILES = ["train.json", "dev.json", "test.json"]

_YES_NO = {"yes", "no", "true", "false"}


def _gold_evidence(qa: dict) -> str:
    """Join the annotated evidence rows, table rows first."""
    gold = qa.get("gold_inds") or {}
    if not gold:
        return ""
    table_rows, text_rows = [], []
    for key, value in gold.items():
        (table_rows if key.startswith("table") else text_rows).append(clean(str(value)))
    return "\n".join(table_rows + text_rows)


def _record(item: dict, cfg: Config) -> QARecord | None:
    qa = item.get("qa") or {}
    question = clean(qa.get("question", ""))
    if not question:
        return None

    exe_ans = qa.get("exe_ans")

    if isinstance(exe_ans, bool):
        gold, answer_type, options = ("yes" if exe_ans else "no"), AnswerType.CATEGORICAL, ["yes", "no"]
    elif isinstance(exe_ans, (int, float)):
        gold, answer_type, options = format_number(float(exe_ans)), AnswerType.NUMERIC, []
    elif isinstance(exe_ans, str) and exe_ans.strip().lower() in _YES_NO:
        normalised = exe_ans.strip().lower()
        gold = "yes" if normalised in {"yes", "true"} else "no"
        answer_type, options = AnswerType.CATEGORICAL, ["yes", "no"]
    else:
        # Unparseable answer - cannot be graded, so it cannot carry an RL reward.
        return None

    if cfg.data.use_gold_evidence and (evidence := _gold_evidence(qa)):
        context = evidence[: cfg.data.max_context_chars]
    else:
        narrative = " ".join(item.get("pre_text", []) + item.get("post_text", []))
        table = render_table(item.get("table") or [])
        context = build_context(narrative, table, cfg.data.max_context_chars)

    return QARecord(
        id=f"finqa::{item.get('id', '')}",
        source="finqa",
        domain=Domain.REPORT_QA,
        question=question,
        context=context,
        gold_answer=gold,
        answer_type=answer_type,
        answer_options=options,
    )


def load(cfg: Config) -> list[QARecord]:
    """Load every FinQA split. We re-split ourselves, so all three files are used."""
    folder = paths.RAW_SOURCES["finqa"]
    records: list[QARecord] = []
    skipped = 0

    for name in SPLIT_FILES:
        path = folder / name
        if not path.exists():
            log.warning("FinQA file missing: %s (run the download step)", path.name)
            continue
        items = json.loads(path.read_text(encoding="utf-8"))
        for item in items:
            record = _record(item, cfg)
            if record is None:
                skipped += 1
            else:
                records.append(record)
        log.info("finqa/%s: %d items", name, len(items))

    log.info("finqa: %d usable records (%d skipped as ungradeable)", len(records), skipped)
    return records
