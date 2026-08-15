"""Regression tests for the prefill OOM.

The first Kaggle run died at question 2080 with::

    logits = self.lm_head(hidden_states[:, slice_indices, :])
    torch.OutOfMemoryError: Tried to allocate 10.12 GiB

The prefill was projecting *every* prompt position through the vocabulary head -
batch 32 x ~1,100 tokens x 152k vocab x 2 bytes - and all but the last row was thrown
away immediately. These tests pin the fix.
"""

from __future__ import annotations

import pytest

from adaptive_reasoning.config import load_config
from adaptive_reasoning.schema import AnswerType, Domain, QARecord
from adaptive_reasoning.traces.generator import _logits_to_keep_arg

SMOKE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def test_detects_the_supported_spelling():
    class NewStyle:
        def forward(self, input_ids, logits_to_keep=0):
            ...

    class OldStyle:
        def forward(self, input_ids, num_logits_to_keep=0):
            ...

    class Neither:
        def forward(self, input_ids):
            ...

    assert _logits_to_keep_arg(NewStyle()) == "logits_to_keep"
    assert _logits_to_keep_arg(OldStyle()) == "num_logits_to_keep"
    assert _logits_to_keep_arg(Neither()) is None


def test_handles_an_uninspectable_forward():
    class Odd:
        forward = print       # builtin, no usable signature

    assert _logits_to_keep_arg(Odd()) in (None, "logits_to_keep", "num_logits_to_keep")


@pytest.fixture(scope="module")
def generator():
    from adaptive_reasoning.llm import ReasoningLLM
    from adaptive_reasoning.traces.generator import TraceGenerator

    cfg = load_config(overrides={
        "llm": {"model_id": SMOKE_MODEL, "dtype": "float32",
                "max_new_tokens": 24, "temperature": 0.0},
        "traces": {"step_tokens": 8, "max_steps": 3, "probe_max_tokens": 4},
    })
    return TraceGenerator(ReasoningLLM(cfg), cfg)


@pytest.mark.slow
def test_real_model_supports_the_optimisation(generator):
    """If this ever returns None on a supported model, prefill memory silently blows up."""
    assert generator._logits_to_keep_arg == "logits_to_keep"


@pytest.mark.slow
def test_forward_still_returns_one_logits_row_per_sequence(generator):
    import torch

    tokenizer = generator.tokenizer
    encoded = tokenizer(["What is 2 plus 2?", "Name a colour."],
                        return_tensors="pt", padding=True)
    ids = encoded.input_ids.to(generator.device)
    mask = encoded.attention_mask.to(generator.device)
    positions = (mask.cumsum(dim=1) - 1).clamp(min=0)

    with torch.no_grad():
        logits, cache = generator._forward(ids, mask, None, positions)

    assert logits.shape == (2, generator.model.config.vocab_size)
    assert cache.get_seq_length() == ids.shape[1]


@pytest.mark.slow
def test_generation_is_unchanged_by_the_optimisation(generator):
    """Trimming the logits rows must not alter what the model produces."""
    records = [
        QARecord(id="m1", source="synthetic", domain=Domain.INVESTMENT,
                 question="What is 6 plus 7?", gold_answer="13",
                 answer_type=AnswerType.NUMERIC),
    ]
    with_opt = generator.generate(records, probe=False)[0]

    generator._logits_to_keep_arg = None      # force the old, wasteful path
    try:
        without_opt = generator.generate(records, probe=False)[0]
    finally:
        generator._logits_to_keep_arg = "logits_to_keep"

    assert with_opt.total_tokens == without_opt.total_tokens
    assert with_opt.final_answer == without_opt.final_answer
