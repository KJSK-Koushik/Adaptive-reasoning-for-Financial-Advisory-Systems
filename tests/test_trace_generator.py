"""Trace generator tests.

``test_probing_does_not_perturb_reasoning`` is the one that matters. Probing appends
tokens to the live KV cache and then crops it back; if the restore is wrong, the model
carries on from a corrupted state and *every trace in the dataset is silently wrong* -
with no crash and no obviously broken output. It would surface only as a DQN that
refuses to learn, weeks later.
"""

from __future__ import annotations

import pytest
import torch

from adaptive_reasoning.config import load_config
from adaptive_reasoning.schema import AnswerType, Domain, QARecord
from adaptive_reasoning.traces.generator import _entropy, _sample, crop_cache

SMOKE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


# --------------------------------------------------------------------------- #
# pure helpers - no model needed
# --------------------------------------------------------------------------- #
def test_entropy_is_zero_for_a_certain_distribution():
    logits = torch.tensor([[0.0, 100.0, 0.0]])
    assert float(_entropy(logits)) == pytest.approx(0.0, abs=1e-5)


def test_entropy_is_maximal_for_a_uniform_distribution():
    logits = torch.zeros(1, 8)
    assert float(_entropy(logits)) == pytest.approx(torch.tensor(8.0).log().item(), abs=1e-5)


def test_entropy_is_per_sequence():
    logits = torch.stack([torch.zeros(8), torch.tensor([100.0] + [0.0] * 7)])
    values = _entropy(logits)
    assert values.shape == (2,)
    assert values[0] > values[1]


def test_sample_is_greedy_at_zero_temperature():
    logits = torch.tensor([[1.0, 5.0, 2.0], [9.0, 0.0, 0.0]])
    assert _sample(logits, temperature=0.0, top_p=1.0).tolist() == [1, 0]


def test_sample_respects_top_p():
    """With a dominant token and tight top_p, only that token may be drawn."""
    logits = torch.tensor([[10.0, 0.0, 0.0, 0.0]])
    picks = {int(_sample(logits, temperature=1.0, top_p=0.5)) for _ in range(50)}
    assert picks == {0}


def test_sample_returns_one_token_per_sequence():
    logits = torch.randn(5, 100)
    assert _sample(logits, temperature=0.8, top_p=0.95).shape == (5,)


# --------------------------------------------------------------------------- #
# cache cropping
# --------------------------------------------------------------------------- #
class _FakeLayer:
    def __init__(self, length):
        self.keys = torch.randn(1, 2, length, 4)
        self.values = torch.randn(1, 2, length, 4)


class _FakeCache:
    """No ``crop`` method, exercising the manual-slice fallback."""

    def __init__(self, length, n_layers=3):
        self.layers = [_FakeLayer(length) for _ in range(n_layers)]

    def get_seq_length(self):
        return self.layers[0].keys.shape[2]


def test_crop_cache_fallback_shortens_every_layer():
    cache = _FakeCache(20)
    crop_cache(cache, 12)
    assert cache.get_seq_length() == 12
    assert all(layer.values.shape[2] == 12 for layer in cache.layers)


def test_crop_cache_preserves_the_retained_prefix():
    cache = _FakeCache(20)
    expected = cache.layers[0].keys[:, :, :12, :].clone()
    crop_cache(cache, 12)
    assert torch.equal(cache.layers[0].keys, expected)


def test_crop_cache_is_a_noop_when_already_short_enough():
    cache = _FakeCache(8)
    crop_cache(cache, 12)
    assert cache.get_seq_length() == 8


def test_crop_cache_rejects_an_uncroppable_cache():
    class Opaque:
        def get_seq_length(self):
            return 10

    with pytest.raises(RuntimeError, match="croppable KV cache"):
        crop_cache(Opaque(), 5)


# --------------------------------------------------------------------------- #
# end to end with a real (small) model
# --------------------------------------------------------------------------- #
def _records(n=2):
    return [
        QARecord(id=f"q{i}", source="synthetic", domain=Domain.INVESTMENT,
                 question=f"What is {i + 3} plus {i + 4}?", gold_answer=str(2 * i + 7),
                 answer_type=AnswerType.NUMERIC)
        for i in range(n)
    ]


@pytest.fixture(scope="module")
def generator():
    """Real model, greedy decoding, short budget - a few seconds on CPU."""
    from adaptive_reasoning.llm import ReasoningLLM
    from adaptive_reasoning.traces.generator import TraceGenerator

    cfg = load_config(overrides={
        "llm": {"model_id": SMOKE_MODEL, "dtype": "float32",
                "max_new_tokens": 48, "temperature": 0.0},
        "traces": {"step_tokens": 12, "max_steps": 4, "probe_max_tokens": 6},
    })
    llm = ReasoningLLM(cfg)
    return TraceGenerator(llm, cfg)


@pytest.mark.slow
def test_probing_does_not_perturb_reasoning(generator):
    """The whole design rests on this: probe, crop the cache, carry on unchanged.

    Greedy decoding makes both runs deterministic, so the generated text must match
    exactly. Any drift means the cache restore is broken and every trace is corrupt.
    """
    records = _records()
    with_probes = generator.generate(records, probe=True)
    without_probes = generator.generate(records, probe=False)

    for probed, clean in zip(with_probes, without_probes, strict=True):
        assert probed.total_tokens == clean.total_tokens, "probing changed the token count"
        assert probed.final_answer == clean.final_answer, "probing changed the final answer"


@pytest.mark.slow
def test_generate_produces_steps(generator):
    traces = generator.generate(_records(), probe=True)
    assert all(t.steps for t in traces)
    assert all(s.step_index == i for t in traces for i, s in enumerate(t.steps))


@pytest.mark.slow
def test_tokens_so_far_increases_monotonically(generator):
    for trace in generator.generate(_records(), probe=True):
        counts = [s.tokens_so_far for s in trace.steps]
        assert counts == sorted(counts)
        assert all(c > 0 for c in counts)


@pytest.mark.slow
def test_confidence_and_entropy_are_in_range(generator):
    for trace in generator.generate(_records(), probe=True):
        for step in trace.steps:
            assert 0.0 <= step.confidence <= 1.0
            assert 0.0 <= step.min_token_confidence <= step.confidence + 1e-6
            assert step.entropy >= 0.0


@pytest.mark.slow
def test_last_step_is_marked_terminal(generator):
    for trace in generator.generate(_records(), probe=True):
        assert trace.steps[-1].is_terminal


@pytest.mark.slow
def test_short_answer_still_yields_a_terminal_step(generator):
    """A model that answers before the first probe boundary must still produce a step.

    Regression: the first implementation emitted zero steps whenever generation ended
    inside the first ``step_tokens`` window. A trace with no steps gives the policy no
    decision to learn from - and that happens precisely on the easy questions where
    learning to stop immediately matters most.
    """
    record = QARecord(
        id="trivial", source="synthetic", domain=Domain.INVESTMENT,
        question="What is 2 plus 2?", gold_answer="4", answer_type=AnswerType.NUMERIC,
    )
    trace = generator.generate([record], probe=True)[0]
    assert len(trace.steps) >= 1
    assert trace.steps[-1].is_terminal
    assert trace.steps[-1].tokens_so_far == trace.total_tokens


@pytest.mark.slow
def test_step_count_never_exceeds_max_steps(generator):
    for trace in generator.generate(_records(), probe=True):
        assert len(trace.steps) <= generator.cfg.traces.max_steps


@pytest.mark.slow
def test_batching_matches_single_sequence_generation(generator):
    """Left padding must not change what a sequence generates."""
    records = _records(2)
    batched = generator.generate(records, probe=True)
    alone = [generator.generate([r], probe=True)[0] for r in records]
    for a, b in zip(batched, alone, strict=True):
        assert a.final_answer == b.final_answer
