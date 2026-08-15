from __future__ import annotations

from collections import Counter

import pytest

from adaptive_reasoning.data.build import to_dataframe
from adaptive_reasoning.data.io import load_unified, stratified_sample
from adaptive_reasoning.schema import AnswerType, Difficulty, Domain, QARecord, Split


def _record(i: int, source: str = "finqa", difficulty=None) -> QARecord:
    return QARecord(
        id=f"{source}::{i}",
        source=source,
        domain=Domain.REPORT_QA,
        question=f"question {i}?",
        context="ctx",
        gold_answer="1",
        answer_type=AnswerType.NUMERIC,
        answer_options=[],
        difficulty=difficulty,
        split=Split.TRAIN,
    )


@pytest.fixture
def parquet(tmp_path):
    def _write(records):
        path = tmp_path / "unified.parquet"
        to_dataframe(records).to_parquet(path, index=False)
        return path
    return _write


def test_round_trip_preserves_records(parquet):
    original = [_record(0), _record(1)]
    path = parquet(original)
    loaded = load_unified(path)
    assert [r.id for r in loaded] == [r.id for r in original]
    assert loaded[0].question == "question 0?"


def test_list_column_survives_parquet(parquet):
    """Parquet returns list columns as numpy arrays, which pydantic rejects."""
    record = _record(0)
    record.answer_options = ["yes", "no"]
    record.answer_type = AnswerType.CATEGORICAL
    loaded = load_unified(parquet([record]))
    assert loaded[0].answer_options == ["yes", "no"]


def test_missing_difficulty_becomes_none_not_nan(parquet):
    """Parquet writes an absent difficulty as NaN, which is not a valid enum."""
    loaded = load_unified(parquet([_record(0)]))
    assert loaded[0].difficulty is None
    assert loaded[0].difficulty_score is None


def test_present_difficulty_is_preserved(parquet):
    loaded = load_unified(parquet([_record(0, difficulty=Difficulty.HARD)]))
    assert loaded[0].difficulty == Difficulty.HARD


def test_require_difficulty_filters_unlabelled(parquet):
    path = parquet([_record(0), _record(1, difficulty=Difficulty.EASY)])
    assert len(load_unified(path)) == 2
    assert len(load_unified(path, require_difficulty=True)) == 1


def test_require_difficulty_raises_when_nothing_is_labelled(parquet):
    with pytest.raises(ValueError, match="no difficulty labels"):
        load_unified(parquet([_record(0)]), require_difficulty=True)


def test_missing_file_raises_with_guidance(tmp_path):
    with pytest.raises(FileNotFoundError, match="run_phase1"):
        load_unified(tmp_path / "absent.parquet")


# --------------------------------------------------------------------------- #
# stratified sampling
# --------------------------------------------------------------------------- #
def _mixed() -> list[QARecord]:
    records = []
    for source, count in [("finqa", 500), ("phrasebank", 100), ("german_credit", 20)]:
        records.extend(_record(i, source) for i in range(count))
    return records


def test_stratified_sample_returns_the_requested_count():
    assert len(stratified_sample(_mixed(), 90, seed=1)) == 90


def test_stratified_sample_covers_every_source():
    """An unstratified sample would be dominated by FinQA and starve the easy tier."""
    picked = stratified_sample(_mixed(), 90, seed=1)
    assert set(Counter(r.source for r in picked)) == {"finqa", "phrasebank", "german_credit"}


def test_stratified_sample_tops_up_when_a_group_is_too_small():
    """german_credit has only 20 records but the even quota asks for 40."""
    picked = stratified_sample(_mixed(), 120, seed=1)
    assert len(picked) == 120


def test_stratified_sample_returns_everything_when_n_exceeds_size():
    records = _mixed()
    assert len(stratified_sample(records, 10_000, seed=1)) == len(records)


def test_stratified_sample_is_deterministic():
    first = [r.id for r in stratified_sample(_mixed(), 60, seed=7)]
    second = [r.id for r in stratified_sample(_mixed(), 60, seed=7)]
    assert first == second


def test_stratified_sample_has_no_duplicates():
    picked = stratified_sample(_mixed(), 200, seed=3)
    assert len({r.id for r in picked}) == len(picked)
