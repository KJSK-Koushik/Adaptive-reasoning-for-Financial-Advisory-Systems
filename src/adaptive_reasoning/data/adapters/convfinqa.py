"""ConvFinQA adapter - conversational, multi-turn numerical reasoning (Hard tier).

Structure verified against the real download (``scripts/run_phase1.py --inspect
convfinqa``). The release ships each split twice:

* ``train.json`` / ``dev.json`` - **conversation-level**, 3,037 / 421 items. Each item
  holds ``annotation.dialogue_break`` (the decomposed turn questions) and
  ``annotation.exe_ans_list`` (one answer per turn). Lengths agree on 421/421 dev
  conversations, giving 1,490 dev turns and ~11,104 train turns.
* ``train_turn.json`` / ``dev_turn.json`` - **turn-level**, one item per turn, with
  ``annotation.cur_dial`` (the dialogue up to and including this turn) and
  ``annotation.exe_ans`` (this turn's answer).

.. warning::
   **Do not grade against ``qa.exe_ans``.** The ``qa`` block is inherited from the
   original FinQA example the conversation was built from, and is *identical across
   every turn*. In ``dev_turn.json`` the first conversation has five turns whose true
   answers are 60.94, 25.14, 35.8, 25.14 and 1.42403, while ``qa.exe_ans`` reads
   1.42403 for all five. Using it would assign the final answer to every turn - the
   labels would look plausible and be wrong, which is the worst kind of data bug.
   ``annotation.exe_ans`` / ``exe_ans_list`` are the per-turn answers.

``test_private.json`` and ``test_turn_private.json`` are excluded: they carry only
``dialogue_break`` / ``cur_dial`` with no answers, being held out for the leaderboard.

Turns are emitted individually, with the preceding turns and their answers prepended
to the context. Later turns are genuinely hard - "and how much does that change
represent in relation to this 2005 value?" is meaningless without the history.
"""

from __future__ import annotations

import json

from ... import paths
from ...config import Config
from ...logging_utils import get_logger
from ...schema import AnswerType, Domain, QARecord
from ..text_utils import build_context, clean, format_number, render_table

log = get_logger("data.convfinqa")

#: Conversation-level splits that carry answers. The turn-level files contain the same
#: content re-exploded, so loading both would duplicate every question.
SPLIT_FILES = ["train.json", "dev.json"]


def _gold(value) -> tuple[str, AnswerType, list[str]] | None:
    """Normalise one ConvFinQA answer into (gold, type, options)."""
    if isinstance(value, bool):
        return ("yes" if value else "no", AnswerType.CATEGORICAL, ["yes", "no"])
    if isinstance(value, (int, float)):
        return (format_number(float(value)), AnswerType.NUMERIC, [])
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"yes", "no"}:
            return (text, AnswerType.CATEGORICAL, ["yes", "no"])
    return None


def _base_context(item: dict, cfg: Config) -> str:
    """Report context, preferring the annotated gold evidence rows when enabled."""
    if cfg.data.use_gold_evidence:
        gold_inds = (item.get("qa") or {}).get("gold_inds") or {}
        if gold_inds:
            table_rows, text_rows = [], []
            for key, value in gold_inds.items():
                (table_rows if key.startswith("table") else text_rows).append(
                    clean(str(value))
                )
            return "\n".join(table_rows + text_rows)[: cfg.data.max_context_chars]

    narrative = " ".join(item.get("pre_text", []) + item.get("post_text", []))
    table = render_table(item.get("table") or [])
    return build_context(narrative, table, cfg.data.max_context_chars)


def _turns(item: dict) -> list[tuple[str, object]]:
    """Extract (question, answer) pairs, handling both file layouts.

    Returns an empty list when the item carries no answers.
    """
    ann = item.get("annotation") or {}

    # Turn-level layout: one turn per item, dialogue history in cur_dial.
    if "cur_dial" in ann:
        dialogue = ann.get("cur_dial") or []
        if not dialogue or "exe_ans" not in ann:
            return []
        # Only the final entry is this item's question; earlier ones are history whose
        # answers live in exe_ans_list.
        history_answers = ann.get("exe_ans_list") or []
        pairs: list[tuple[str, object]] = []
        for i, question in enumerate(dialogue[:-1]):
            answer = history_answers[i] if i < len(history_answers) else None
            pairs.append((question, answer))
        pairs.append((dialogue[-1], ann["exe_ans"]))
        return pairs

    # Conversation-level layout: the whole decomposed dialogue in one item.
    questions = ann.get("dialogue_break") or []
    answers = ann.get("exe_ans_list") or []
    if not questions or len(answers) < len(questions):
        return []
    return list(zip(questions, answers, strict=False))


def _records(item: dict, cfg: Config, fallback_id: int) -> list[QARecord]:
    pairs = _turns(item)
    if not pairs:
        return []

    base = _base_context(item, cfg)
    conv_id = item.get("id", fallback_id)

    out: list[QARecord] = []
    history: list[tuple[str, str]] = []

    for turn, (raw_question, raw_answer) in enumerate(pairs):
        question = clean(str(raw_question))
        parsed = _gold(raw_answer)
        if not question:
            continue
        if parsed is None:
            # Unanswerable turn: keep it as history so later turns still make sense.
            history.append((question, str(raw_answer)))
            continue
        gold, answer_type, options = parsed

        context = base
        if history:
            prior = "\n".join(f"Q: {q}\nA: {a}" for q, a in history)
            context = f"{base}\n\nEarlier in this conversation:\n{prior}"

        out.append(
            QARecord(
                id=f"convfinqa::{conv_id}::turn{turn}",
                source="convfinqa",
                domain=Domain.REPORT_QA,
                question=question,
                # Allow headroom over max_context_chars for the dialogue history, which
                # is short but essential - truncating it makes later turns unanswerable.
                context=context[: cfg.data.max_context_chars + 1200],
                gold_answer=gold,
                answer_type=answer_type,
                answer_options=options,
            )
        )
        history.append((question, gold))

    return out


def load(cfg: Config) -> list[QARecord]:
    folder = paths.RAW_SOURCES["convfinqa"]
    present = [folder / n for n in SPLIT_FILES if (folder / n).exists()]
    if not present:
        log.warning(
            "ConvFinQA not found in %s - skipping. Clone "
            "https://github.com/czyssrs/ConvFinQA and unzip data.zip there.",
            folder,
        )
        return []

    records: list[QARecord] = []
    for path in present:
        items = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(items, list) or not items:
            log.warning("convfinqa/%s: unexpected top-level type, skipping", path.name)
            continue

        before = len(records)
        for i, item in enumerate(items):
            records.extend(_records(item, cfg, i))

        produced = len(records) - before
        if produced == 0:
            raise ValueError(
                f"convfinqa/{path.name}: parsed 0 turns from {len(items)} items. "
                f"Expected 'annotation.dialogue_break' + 'annotation.exe_ans_list' "
                f"(conversation-level) or 'annotation.cur_dial' + 'annotation.exe_ans' "
                f"(turn-level); annotation keys were "
                f"{sorted(items[0].get('annotation') or {})}."
            )
        log.info("convfinqa/%s: %d conversations -> %d turns", path.name, len(items), produced)

    log.info("convfinqa: %d usable records", len(records))
    return records
