"""Tests for the Phase 3 pre-flight gate.

The scenario in ``test_gate_fails_on_a_non_reasoning_model`` is not hypothetical: the
Phase 2 smoke run on Qwen2.5-0.5B-Instruct produced a median of 3 reasoning tokens.
Without this gate, that behaviour would have consumed twelve GPU-hours and produced
traces with nothing to learn from.
"""

from __future__ import annotations

import pytest

from adaptive_reasoning.config import load_config
from adaptive_reasoning.llm import Generation
from adaptive_reasoning.pilot import PilotReport, _apply_gate, _project_cost, run_pilot
from adaptive_reasoning.schema import AnswerType, Domain, QARecord


def _report(**overrides) -> PilotReport:
    base = {
        "model_id": "test/model", "device": "cpu", "n_questions": 50,
        "median_reasoning_tokens": 300.0, "mean_reasoning_tokens": 320.0,
        "p90_reasoning_tokens": 600.0, "max_reasoning_tokens": 900,
        "think_tag_rate": 1.0, "answer_rate": 0.95, "truncation_rate": 0.05,
        "wall_seconds": 120.0, "tokens_per_second": 600.0, "accuracy": 0.5,
    }
    base.update(overrides)
    return PilotReport(**base)


@pytest.fixture
def cfg():
    return load_config()


def test_gate_passes_for_a_reasoning_model(cfg):
    report = _report()
    _apply_gate(report, cfg)
    assert report.passed
    assert report.failures == []


def test_gate_fails_on_a_non_reasoning_model(cfg):
    """Median 3 tokens - exactly what the 0.5B smoke model produced."""
    report = _report(median_reasoning_tokens=3.0)
    _apply_gate(report, cfg)
    assert not report.passed
    assert any("not reasoning" in f for f in report.failures)
    assert any("DeepSeek-R1-Distill" in f for f in report.failures)


def test_gate_fails_when_answers_are_missing(cfg):
    report = _report(answer_rate=0.2)
    _apply_gate(report, cfg)
    assert not report.passed
    assert any("produced an answer" in f for f in report.failures)


def test_gate_reports_every_failure_not_just_the_first(cfg):
    report = _report(median_reasoning_tokens=2.0, answer_rate=0.1)
    _apply_gate(report, cfg)
    assert len(report.failures) == 2


def test_gate_thresholds_are_config_driven():
    lenient = load_config(overrides={"traces": {"pilot": {"min_median_reasoning_tokens": 1}}})
    strict = load_config(overrides={"traces": {"pilot": {"min_median_reasoning_tokens": 500}}})
    a, b = _report(median_reasoning_tokens=100.0), _report(median_reasoning_tokens=100.0)
    _apply_gate(a, lenient)
    _apply_gate(b, strict)
    assert a.passed and not b.passed


def test_cost_projection_scales_with_throughput(cfg):
    fast, slow = _report(tokens_per_second=1200.0), _report(tokens_per_second=600.0)
    _project_cost(fast, cfg)
    _project_cost(slow, cfg)
    assert slow.projected_phase3_hours == pytest.approx(2 * fast.projected_phase3_hours, rel=0.01)


def test_cost_projection_handles_zero_throughput(cfg):
    report = _report(tokens_per_second=0.0)
    _project_cost(report, cfg)
    assert report.projected_phase3_hours == 0.0


# --------------------------------------------------------------------------- #
# end to end with a stubbed model
# --------------------------------------------------------------------------- #
class _StubLLM:
    """Minimal stand-in so the pilot can be exercised without loading a model."""

    def __init__(self, text: str, tokenizer):
        self.model_id = "stub"
        self.hw = type("hw", (), {"backend": "cpu"})()
        self.tokenizer = tokenizer
        self._text = text

    def render(self, messages):
        return messages[-1]["content"]

    def batched(self, prompts, **kwargs):
        for i, _ in enumerate(prompts):
            yield i, Generation(text=self._text, n_prompt_tokens=10,
                                n_generated_tokens=50, finished=True)


class _WordTokenizer:
    def encode(self, text, add_special_tokens=False):
        return text.split()


def _records(n=4):
    return [
        QARecord(id=f"q{i}", source="synthetic", domain=Domain.INVESTMENT,
                 question="What is 5% of 200?", gold_answer="10",
                 answer_type=AnswerType.NUMERIC)
        for i in range(n)
    ]


def test_run_pilot_detects_a_reasoning_model(cfg):
    text = "<think>" + "step " * 200 + "</think>\nFinal answer: 10"
    report = run_pilot(_records(), cfg, llm=_StubLLM(text, _WordTokenizer()))
    assert report.think_tag_rate == 1.0
    assert report.answer_rate == 1.0
    assert report.accuracy == 1.0
    assert report.passed


def test_run_pilot_detects_a_shortcutting_model(cfg):
    report = run_pilot(_records(), cfg, llm=_StubLLM("Final answer: 10", _WordTokenizer()))
    assert report.median_reasoning_tokens < 10
    assert not report.passed


def test_run_pilot_measures_throughput(cfg):
    report = run_pilot(_records(8), cfg, llm=_StubLLM("Final answer: 10", _WordTokenizer()))
    assert report.tokens_per_second > 0
    assert report.wall_seconds >= 0
