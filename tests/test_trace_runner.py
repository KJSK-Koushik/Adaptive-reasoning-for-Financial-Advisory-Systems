from __future__ import annotations

import pytest

from adaptive_reasoning.schema import TRACE_STEP_COLUMNS, Difficulty, Trace, TraceStep
from adaptive_reasoning.traces.runner import _steps_frame, _summary_frame


def _step(qid, index, tokens, correct) -> TraceStep:
    return TraceStep(
        question_id=qid, step_index=index, tokens_so_far=tokens, step_text="...",
        probe_answer="12.4" if correct else "99", probe_correct=correct,
        confidence=0.8, min_token_confidence=0.6, entropy=0.3, answer_changed=False,
    )


def _trace(qid="q1", pattern=(False, True, True), total=160) -> Trace:
    steps = [_step(qid, i, (i + 1) * 40, c) for i, c in enumerate(pattern)]
    return Trace(
        question_id=qid, difficulty=Difficulty.MEDIUM, total_tokens=total,
        final_answer="12.4", final_correct=pattern[-1], steps=steps,
    )


def test_steps_frame_has_the_declared_columns():
    frame = _steps_frame([_trace()])
    assert list(frame.columns) == TRACE_STEP_COLUMNS


def test_steps_frame_flattens_every_step():
    frame = _steps_frame([_trace("a"), _trace("b")])
    assert len(frame) == 6
    assert set(frame.question_id) == {"a", "b"}


def test_summary_records_the_oracle_stopping_point():
    """oracle_tokens is the cost of stopping at the earliest correct step."""
    frame = _summary_frame([_trace(pattern=(False, True, True))])
    row = frame.iloc[0]
    assert row.earliest_correct_step == 1
    assert row.oracle_tokens == 80          # step 1 -> (1+1)*40


def test_summary_handles_a_never_correct_trace():
    frame = _summary_frame([_trace(pattern=(False, False))])
    row = frame.iloc[0]
    assert row.earliest_correct_step is None
    assert row.oracle_tokens is None


def test_summary_oracle_equals_total_when_only_the_last_step_is_correct():
    frame = _summary_frame([_trace(pattern=(False, False, True))])
    assert frame.iloc[0].oracle_tokens == 120


def test_summary_counts_steps():
    frame = _summary_frame([_trace(pattern=(False, True, True, True))])
    assert frame.iloc[0].n_steps == 4


@pytest.mark.parametrize(
    ("pattern", "expected"),
    [((True,), 0), ((False, True), 1), ((False, False, True), 2), ((False, False), None)],
)
def test_earliest_correct_step_matches_the_pattern(pattern, expected):
    assert _trace(pattern=pattern).earliest_correct_step == expected


# --------------------------------------------------------------------------- #
# wall-clock budget
# --------------------------------------------------------------------------- #
class TestWallClockBudget:
    """A hosted notebook killed at its time limit saves nothing. Stopping
    ourselves, flushing and consolidating turns a lost run into a smaller one."""

    def test_budget_defaults_to_unlimited(self):
        from adaptive_reasoning.config import load_config

        assert load_config().traces.max_wall_seconds is None

    def test_budget_is_configurable(self, tmp_path):
        from adaptive_reasoning.config import load_config

        cfg = load_config(overrides={"traces": {"max_wall_seconds": 42}})
        assert cfg.traces.max_wall_seconds == 42

    def test_run_stops_when_the_budget_is_exhausted(self, monkeypatch, tmp_path):
        """With a zero budget nothing is generated, and the run still returns."""
        import adaptive_reasoning.traces.runner as runner
        from adaptive_reasoning.config import load_config
        from adaptive_reasoning.schema import AnswerType, Domain, QARecord

        monkeypatch.setattr(runner, "SHARD_DIR", tmp_path / "_shards")
        monkeypatch.setattr(runner, "completed_ids", lambda: set())
        monkeypatch.setattr(runner, "consolidate", lambda: (tmp_path / "a", tmp_path / "b"))
        monkeypatch.setattr(runner, "summarise", lambda: {"n_traces": 0})

        class _Generator:
            def __init__(self, *a, **k):
                pass

            def generate(self, records, probe=True):
                raise AssertionError("generation should not start on a zero budget")

        monkeypatch.setattr(runner, "TraceGenerator", _Generator)
        monkeypatch.setattr(runner, "ReasoningLLM", lambda cfg: object())

        cfg = load_config(overrides={"traces": {"max_wall_seconds": 0}})
        records = [
            QARecord(id=f"q{i}", source="synthetic", domain=Domain.INVESTMENT,
                     question="q", context="", gold_answer="1",
                     answer_type=AnswerType.NUMERIC)
            for i in range(4)
        ]
        out = runner.run(records, cfg, resume=False)
        assert out["stopped_early"] is True
        assert "wall_seconds" in out


# --------------------------------------------------------------------------- #
# probe agreement
# --------------------------------------------------------------------------- #
class TestProbeAgreement:
    """The forced probe and the model's own final answer measure the same thing at
    the same point. When they diverge the probe is capturing something else, and
    every reward in the RL dataset is computed from it wrongly."""

    def _trace(self, probe_correct, final_correct):
        from adaptive_reasoning.schema import Trace, TraceStep

        step = TraceStep(
            question_id="q", step_index=0, tokens_so_far=48, step_text="",
            probe_answer="x", probe_correct=probe_correct, confidence=0.5,
            min_token_confidence=0.5, entropy=0.5, answer_changed=False,
            is_terminal=True,
        )
        return Trace(question_id="q", total_tokens=48, steps=[step],
                     final_answer="x", final_correct=final_correct)

    def test_full_agreement(self):
        from adaptive_reasoning.traces.runner import _agreement_of

        traces = [self._trace(True, True), self._trace(False, False)]
        assert _agreement_of(traces) == 1.0

    def test_total_disagreement(self):
        from adaptive_reasoning.traces.runner import _agreement_of

        traces = [self._trace(False, True), self._trace(True, False)]
        assert _agreement_of(traces) == 0.0

    def test_the_contaminated_case_falls_below_the_floor(self):
        """Run 3 measured 64.8%; the floor is 90%."""
        from adaptive_reasoning.traces.runner import MIN_PROBE_AGREEMENT, _agreement_of

        traces = [self._trace(False, True)] * 35 + [self._trace(True, True)] * 65
        agreement = _agreement_of(traces)
        assert agreement == pytest.approx(0.65)
        assert agreement < MIN_PROBE_AGREEMENT

    def test_empty_input_is_treated_as_agreeing(self):
        from adaptive_reasoning.traces.runner import _agreement_of

        assert _agreement_of([]) == 1.0
