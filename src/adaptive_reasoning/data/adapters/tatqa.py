"""TAT-QA adapter.

One row in the raw file is a *table plus paragraphs plus a list of questions*, so
the file is exploded into one record per question.

Answer-type handling, based on the actual train split (13,251 questions):

* ``arithmetic`` (5,553) - numeric, kept.
* ``count`` (305) - numeric, kept.
* ``span`` (5,737) - a single span. Kept **only when it parses as a number**, which
  covers the "what was X in 2019?" lookups. Text spans are dropped: grading free text
  against a reasoning model's phrasing is unreliable, and unreliable grading poisons
  the RL reward.
* ``multi-span`` (1,656) - dropped, same reason.

TAT-QA stores a separate ``scale`` field (``thousand``/``million``/``percent``/...)
and the official metric scores value and scale separately. We fold the scale into the
question as an explicit instruction ("Give your answer in thousand.") so the target is
unambiguous and a single numeric comparison suffices.

Only train and dev are downloaded - the public test split ships without answers.
"""

from __future__ import annotations

import json

from ... import paths
from ...config import Config
from ...grading import extract_number
from ...logging_utils import get_logger
from ...schema import AnswerType, Domain, QARecord
from ..text_utils import build_context, clean, format_number, render_table

log = get_logger("data.tatqa")

SPLIT_FILES = ["tatqa_dataset_train.json", "tatqa_dataset_dev.json"]

_SCALE_HINT = {
    "thousand": "Give your answer in thousands.",
    "million": "Give your answer in millions.",
    "billion": "Give your answer in billions.",
    "percent": "Give your answer as a percentage.",
}


def _answer_value(answer, answer_type: str) -> float | None:
    """Reduce a TAT-QA answer to a single number, or ``None`` if it is not numeric."""
    if isinstance(answer, list):
        if len(answer) != 1:
            return None          # multi-span
        answer = answer[0]
    if isinstance(answer, bool):
        return None
    if isinstance(answer, (int, float)):
        return float(answer)
    if isinstance(answer, str):
        return extract_number(answer, prefer="first")
    return None


def _records_for_table(row: dict, cfg: Config) -> list[QARecord]:
    table_block = render_table((row.get("table") or {}).get("table") or [])
    narrative = " ".join(
        clean(p.get("text", "")) for p in sorted(
            row.get("paragraphs") or [], key=lambda p: p.get("order", 0)
        )
    )
    context = build_context(narrative, table_block, cfg.data.max_context_chars)

    out: list[QARecord] = []
    for q in row.get("questions") or []:
        answer_type = q.get("answer_type", "")
        if answer_type == "multi-span":
            continue

        value = _answer_value(q.get("answer"), answer_type)
        if value is None:
            continue        # text span, or unparseable

        question = clean(q.get("question", ""))
        if not question:
            continue

        if hint := _SCALE_HINT.get(q.get("scale") or ""):
            question = f"{question} {hint}"

        out.append(
            QARecord(
                id=f"tatqa::{q.get('uid', '')}",
                source="tatqa",
                domain=Domain.REPORT_QA,
                question=question,
                context=context,
                gold_answer=format_number(value),
                answer_type=AnswerType.NUMERIC,
            )
        )
    return out


def load(cfg: Config) -> list[QARecord]:
    folder = paths.RAW_SOURCES["tatqa"]
    records: list[QARecord] = []

    for name in SPLIT_FILES:
        path = folder / name
        if not path.exists():
            log.warning("TAT-QA file missing: %s (run the download step)", path.name)
            continue
        rows = json.loads(path.read_text(encoding="utf-8"))
        before = len(records)
        for row in rows:
            records.extend(_records_for_table(row, cfg))
        log.info("tatqa/%s: %d tables -> %d records", name, len(rows), len(records) - before)

    log.info("tatqa: %d usable records", len(records))
    return records
