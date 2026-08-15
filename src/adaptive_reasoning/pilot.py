"""Pre-flight pilot for Phase 3.

Phase 3 costs roughly twelve GPU-hours. Two assumptions underpin it, and if either is
wrong the entire run produces unusable data:

**1. The model reasons at length.** The whole project is about stopping reasoning
early. If the model answers in three tokens there is nothing to stop, no confidence
trajectory to observe, and no signal for the DQN to learn from. This is not
hypothetical - the Phase 2 smoke run on ``Qwen2.5-0.5B-Instruct`` produced a *median of
3 reasoning tokens*, and strengthening the prompt did not help. That model would have
yielded twelve hours of worthless traces.

**2. The throughput estimates are right.** Every figure in ``docs/PHASE3_REMOTE.md`` is
an educated guess until something measures it on the actual hardware.

So Phase 3 runs this first and refuses to proceed if the gate fails. Cheap insurance:
50 questions is a couple of minutes on a GPU.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field

from . import paths
from .config import Config
from .grading import is_correct
from .llm import ReasoningLLM
from .logging_utils import get_logger
from .prompting import build_messages, count_reasoning_tokens, extract_answer
from .schema import QARecord

log = get_logger("pilot")


@dataclass
class PilotReport:
    model_id: str
    device: str
    n_questions: int

    # Reasoning behaviour - assumption 1.
    median_reasoning_tokens: float
    mean_reasoning_tokens: float
    p90_reasoning_tokens: float
    max_reasoning_tokens: int
    think_tag_rate: float          # fraction of generations containing <think>
    answer_rate: float             # fraction that produced a non-empty answer
    truncation_rate: float         # fraction that hit the token cap

    # Throughput - assumption 2.
    wall_seconds: float
    tokens_per_second: float
    accuracy: float

    passed: bool = False
    failures: list[str] = field(default_factory=list)
    projected_phase3_hours: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


def _percentile(values: list[int], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(int(q * (len(ordered) - 1)), len(ordered) - 1)
    return float(ordered[idx])


def _median(values: list[int]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    return float(ordered[mid]) if len(ordered) % 2 else (ordered[mid - 1] + ordered[mid]) / 2


def run_pilot(records: list[QARecord], cfg: Config, llm: ReasoningLLM | None = None) -> PilotReport:
    """Generate one full-length completion per record and measure what happened."""
    llm = llm or ReasoningLLM(cfg)
    tokenizer = llm.tokenizer

    prompts = [llm.render(build_messages(r)) for r in records]
    log.info("pilot: %d questions on %s", len(records), llm.model_id)

    start = time.perf_counter()
    generations = [g for _, g in llm.batched(
        prompts,
        max_new_tokens=cfg.llm.max_new_tokens,
        temperature=cfg.llm.temperature,
        do_sample=True,
    )]
    wall = time.perf_counter() - start

    reasoning_tokens: list[int] = []
    n_think = n_answered = n_truncated = n_correct = 0
    total_generated = 0

    for record, gen in zip(records, generations, strict=True):
        reasoning_tokens.append(count_reasoning_tokens(gen.text, tokenizer))
        total_generated += gen.n_generated_tokens
        if "<think>" in gen.text or "</think>" in gen.text:
            n_think += 1
        if not gen.finished:
            n_truncated += 1
        answer = extract_answer(gen.text)
        if answer.strip():
            n_answered += 1
            if is_correct(answer, record.gold_answer, str(record.answer_type),
                          cfg.data.numeric_tolerance):
                n_correct += 1

    n = len(records)
    report = PilotReport(
        model_id=llm.model_id,
        device=llm.hw.backend,
        n_questions=n,
        median_reasoning_tokens=_median(reasoning_tokens),
        mean_reasoning_tokens=round(sum(reasoning_tokens) / max(n, 1), 1),
        p90_reasoning_tokens=_percentile(reasoning_tokens, 0.9),
        max_reasoning_tokens=max(reasoning_tokens) if reasoning_tokens else 0,
        think_tag_rate=round(n_think / max(n, 1), 3),
        answer_rate=round(n_answered / max(n, 1), 3),
        truncation_rate=round(n_truncated / max(n, 1), 3),
        wall_seconds=round(wall, 1),
        tokens_per_second=round(total_generated / wall, 1) if wall > 0 else 0.0,
        accuracy=round(n_correct / max(n, 1), 3),
    )

    _apply_gate(report, cfg)
    _project_cost(report, cfg)
    return report


def _apply_gate(report: PilotReport, cfg: Config) -> None:
    """Decide whether Phase 3 is allowed to proceed."""
    gate = cfg.traces.pilot
    failures: list[str] = []

    if report.median_reasoning_tokens < gate.min_median_reasoning_tokens:
        failures.append(
            f"median reasoning length is {report.median_reasoning_tokens:.0f} tokens, "
            f"below the required {gate.min_median_reasoning_tokens}. The model is not "
            f"reasoning, so there is nothing for a stopping policy to cut short. "
            f"Use a reasoning-tuned model (e.g. a DeepSeek-R1-Distill) rather than a "
            f"plain instruction model."
        )

    if report.answer_rate < gate.min_answer_rate:
        failures.append(
            f"only {report.answer_rate:.0%} of generations produced an answer, below "
            f"the required {gate.min_answer_rate:.0%}. Check the prompt format and "
            f"that max_new_tokens is large enough to finish."
        )

    report.failures = failures
    report.passed = not failures


def _project_cost(report: PilotReport, cfg: Config) -> None:
    """Replace the guessed GPU-hour estimates with a measured projection."""
    if report.tokens_per_second <= 0:
        return

    # Per question: one full reasoning pass, plus a short probe at each step boundary.
    per_question = report.mean_reasoning_tokens + (
        cfg.traces.max_steps * cfg.traces.probe_max_tokens
    )
    trace_tokens = per_question * cfg.traces.n_questions

    n_difficulty = cfg.difficulty.n_questions or cfg.traces.n_questions
    difficulty_tokens = (
        min(report.mean_reasoning_tokens, cfg.difficulty.sample_max_new_tokens)
        * cfg.difficulty.k_samples
        * n_difficulty
    )

    report.projected_phase3_hours = round(
        (trace_tokens + difficulty_tokens) / report.tokens_per_second / 3600, 2
    )


def write_report(report: PilotReport) -> None:
    path = paths.RESULTS / "phase3_pilot.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    log.info("wrote %s", path)


def print_report(report: PilotReport) -> None:
    print("=" * 68)
    print("  PHASE 3 PILOT")
    print("=" * 68)
    print(f"  model                    {report.model_id}")
    print(f"  device                   {report.device}")
    print(f"  questions                {report.n_questions}")
    print("\n  -- does the model reason? --")
    print(f"  median reasoning tokens  {report.median_reasoning_tokens:.0f}")
    print(f"  mean                     {report.mean_reasoning_tokens:.0f}")
    print(f"  p90                      {report.p90_reasoning_tokens:.0f}")
    print(f"  max                      {report.max_reasoning_tokens}")
    print(f"  <think> tag rate         {report.think_tag_rate:.0%}")
    print(f"  answer rate              {report.answer_rate:.0%}")
    print(f"  truncated at cap         {report.truncation_rate:.0%}")
    print(f"  accuracy                 {report.accuracy:.1%}")
    print("\n  -- throughput --")
    print(f"  wall time                {report.wall_seconds:.0f}s")
    print(f"  tokens/second            {report.tokens_per_second:.1f}")
    print(f"  projected Phase 3        {report.projected_phase3_hours:.1f} GPU-hours")
    print()
    if report.passed:
        print("  GATE PASSED - Phase 3 may proceed.")
    else:
        print("  GATE FAILED - do not run Phase 3:")
        for failure in report.failures:
            print(f"    * {failure}")
    print("=" * 68)
