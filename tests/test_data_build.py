from __future__ import annotations

import pytest

from adaptive_reasoning.config import load_config
from adaptive_reasoning.data import build
from adaptive_reasoning.data.text_utils import build_context, render_table, truncate_context
from adaptive_reasoning.schema import AnswerType, Domain, QARecord, Split


def _record(rid: str, question: str, context: str = "", source: str = "synthetic") -> QARecord:
    return QARecord(
        id=rid,
        source=source,
        domain=Domain.INVESTMENT,
        question=question,
        context=context,
        gold_answer="1",
        answer_type=AnswerType.NUMERIC,
    )


# --------------------------------------------------------------------------- #
# text_utils
# --------------------------------------------------------------------------- #
def test_render_table_adds_a_header_separator():
    out = render_table([["a", "b"], ["1", "2"]])
    assert out.splitlines() == ["a | b", "--- | ---", "1 | 2"]


def test_render_table_flags_truncation_rather_than_hiding_it():
    rows = [["h"]] + [[str(i)] for i in range(40)]
    out = render_table(rows, max_rows=5)
    assert "further rows omitted" in out


def test_render_table_handles_empty():
    assert render_table([]) == ""


def test_truncate_context_keeps_the_end():
    text = "alpha. " * 200 + "THE IMPORTANT TAIL"
    out = truncate_context(text, 100)
    assert "THE IMPORTANT TAIL" in out
    assert len(out) <= 100


def test_build_context_never_sacrifices_the_table():
    table = render_table([["year", "value"], ["2019", "100"]])
    narrative = "x" * 5000
    out = build_context(narrative, table, max_chars=200)
    assert table in out


# --------------------------------------------------------------------------- #
# deduplication
# --------------------------------------------------------------------------- #
def test_deduplicate_matches_on_question_and_context_not_id():
    records = [
        _record("a", "What is 2+2?", "ctx"),
        _record("b", "what is 2+2?", "CTX"),   # different id, same content
        _record("c", "What is 3+3?", "ctx"),
    ]
    kept, removed = build.deduplicate(records)
    assert removed == 1
    assert len(kept) == 2


def test_deduplicate_keeps_same_question_with_different_context():
    records = [
        _record("a", "What was revenue?", "report A"),
        _record("b", "What was revenue?", "report B"),
    ]
    kept, removed = build.deduplicate(records)
    assert removed == 0
    assert len(kept) == 2


# --------------------------------------------------------------------------- #
# sampling and splitting
# --------------------------------------------------------------------------- #
def test_subsample_respects_per_source_caps():
    cfg = load_config(overrides={"data": {"sample_sizes": {"synthetic": 10}}})
    records = [_record(str(i), f"q{i}") for i in range(100)]
    out = build.subsample(records, cfg)
    assert len(out) == 10


def test_subsample_keeps_everything_when_cap_is_null():
    cfg = load_config(overrides={"data": {"sample_sizes": {"finqa": None}}})
    records = [_record(str(i), f"q{i}", source="finqa") for i in range(50)]
    assert len(build.subsample(records, cfg)) == 50


def test_assign_splits_covers_every_record():
    cfg = load_config()
    records = [_record(str(i), f"q{i}") for i in range(1000)]
    build.assign_splits(records, cfg)
    assert all(r.split is not None for r in records)


def test_assign_splits_respects_the_configured_ratios():
    cfg = load_config()
    records = [_record(str(i), f"q{i}") for i in range(1000)]
    build.assign_splits(records, cfg)
    counts = {s: sum(1 for r in records if r.split == s) for s in Split}
    assert counts[Split.TRAIN] == pytest.approx(700, abs=20)
    assert counts[Split.VAL] == pytest.approx(150, abs=20)
    assert counts[Split.TEST] == pytest.approx(150, abs=20)


def test_assign_splits_is_stratified_by_source():
    """Every source must appear in every split, or evaluation is biased."""
    cfg = load_config()
    records = [
        _record(f"{src}{i}", f"q{src}{i}", source=src)
        for src in ("finqa", "phrasebank", "synthetic")
        for i in range(200)
    ]
    build.assign_splits(records, cfg)
    for src in ("finqa", "phrasebank", "synthetic"):
        present = {r.split for r in records if r.source == src}
        assert present == set(Split), f"{src} missing from some split: {present}"


def test_assign_splits_is_deterministic():
    cfg = load_config()
    a = [_record(str(i), f"q{i}") for i in range(300)]
    b = [_record(str(i), f"q{i}") for i in range(300)]
    build.assign_splits(a, cfg)
    build.assign_splits(b, cfg)
    assert [r.split for r in a] == [r.split for r in b]


# --------------------------------------------------------------------------- #
# frame conversion
# --------------------------------------------------------------------------- #
def test_to_dataframe_has_the_declared_column_order():
    from adaptive_reasoning.schema import UNIFIED_COLUMNS

    frame = build.to_dataframe([_record("a", "q")])
    assert list(frame.columns) == UNIFIED_COLUMNS


def test_summarise_counts_by_source():
    records = [_record("a", "q1"), _record("b", "q2", source="finqa")]
    summary = build.summarise(records)
    assert summary["total"] == 2
    assert summary["by_source"] == {"synthetic": 1, "finqa": 1}
