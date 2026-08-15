"""Phase 3 driver: generate traces across the dataset, with restart safety.

Runs on a free Kaggle session, so it assumes it will be killed at some point.
Checkpoint shards are written every ``traces.checkpoint_every`` questions and completed
question ids are skipped on restart, making a dropped session cost minutes rather than
hours.
"""

from __future__ import annotations

import json
from pathlib import Path

from .. import paths
from ..config import Config
from ..llm import ReasoningLLM
from ..logging_utils import get_logger
from ..schema import TRACE_STEP_COLUMNS, QARecord, Trace
from .generator import TraceGenerator

log = get_logger("traces.runner")

SHARD_DIR = paths.TRACES / "_shards"
TRACE_SUMMARY = paths.TRACES / "trace_summary.parquet"


def completed_ids() -> set[str]:
    """Question ids already traced in an earlier session."""
    if not SHARD_DIR.exists():
        return set()
    import pandas as pd

    done: set[str] = set()
    for shard in sorted(SHARD_DIR.glob("steps_*.parquet")):
        try:
            done.update(pd.read_parquet(shard, columns=["question_id"])["question_id"])
        except Exception as exc:  # noqa: BLE001 - a half-written shard must not be fatal
            log.warning("ignoring unreadable shard %s: %s", shard.name, exc)
    return done


def _steps_frame(traces: list[Trace]):
    import pandas as pd

    rows = [step.model_dump() for trace in traces for step in trace.steps]
    frame = pd.DataFrame(rows)
    return frame[TRACE_STEP_COLUMNS] if not frame.empty else frame


def _summary_frame(traces: list[Trace]):
    import pandas as pd

    return pd.DataFrame(
        [
            {
                "question_id": t.question_id,
                "difficulty": str(t.difficulty) if t.difficulty else None,
                "total_tokens": t.total_tokens,
                "final_answer": t.final_answer,
                "final_correct": t.final_correct,
                "n_steps": len(t.steps),
                "earliest_correct_step": t.earliest_correct_step,
                # Tokens that could have been saved by stopping at the earliest correct
                # step - the headroom the DQN is competing for.
                "oracle_tokens": (
                    t.steps[t.earliest_correct_step].tokens_so_far
                    if t.earliest_correct_step is not None else None
                ),
            }
            for t in traces
        ]
    )


def _flush(traces: list[Trace], index: int) -> None:
    SHARD_DIR.mkdir(parents=True, exist_ok=True)
    _steps_frame(traces).to_parquet(SHARD_DIR / f"steps_{index:05d}.parquet", index=False)
    _summary_frame(traces).to_parquet(SHARD_DIR / f"summary_{index:05d}.parquet", index=False)
    log.info("checkpoint %05d: %d traces", index, len(traces))


def consolidate() -> tuple[Path, Path]:
    """Merge shards into ``traces.parquet`` and ``trace_summary.parquet``."""
    import pandas as pd

    steps = sorted(SHARD_DIR.glob("steps_*.parquet")) if SHARD_DIR.exists() else []
    summaries = sorted(SHARD_DIR.glob("summary_*.parquet")) if SHARD_DIR.exists() else []
    if not steps:
        log.warning("no shards to consolidate")
        return paths.TRACE_DATASET, TRACE_SUMMARY

    paths.TRACES.mkdir(parents=True, exist_ok=True)
    pd.concat([pd.read_parquet(s) for s in steps], ignore_index=True).to_parquet(
        paths.TRACE_DATASET, index=False
    )
    pd.concat([pd.read_parquet(s) for s in summaries], ignore_index=True).to_parquet(
        TRACE_SUMMARY, index=False
    )
    log.info("consolidated %d shards -> %s", len(steps), paths.TRACE_DATASET.name)
    return paths.TRACE_DATASET, TRACE_SUMMARY


def run(records: list[QARecord], cfg: Config, llm: ReasoningLLM | None = None,
        resume: bool = True) -> dict:
    """Generate traces for every record. Returns a summary dictionary."""
    llm = llm or ReasoningLLM(cfg)
    generator = TraceGenerator(llm, cfg)

    done = completed_ids() if resume else set()
    todo = [r for r in records if r.id not in done]
    if done:
        log.info("resuming: %d already traced, %d remaining", len(done), len(todo))
    if not todo:
        log.info("nothing to trace")
        return {"traced": 0, "skipped": len(done)}

    batch_size = cfg.llm.batch_size
    shard_index = len(list(SHARD_DIR.glob("steps_*.parquet"))) if SHARD_DIR.exists() else 0
    pending: list[Trace] = []
    produced = 0

    for start in range(0, len(todo), batch_size):
        chunk = todo[start : start + batch_size]
        traces = generator.generate(chunk, probe=True)
        pending.extend(traces)
        produced += len(traces)

        if len(pending) >= cfg.traces.checkpoint_every:
            _flush(pending, shard_index)
            shard_index += 1
            pending = []

        if produced % max(batch_size * 5, 1) == 0:
            log.info("  %d/%d traced", produced, len(todo))

    if pending:
        _flush(pending, shard_index)

    consolidate()
    return summarise()


def summarise() -> dict:
    """Headline statistics over the consolidated traces."""
    import pandas as pd

    if not TRACE_SUMMARY.exists():
        return {}

    frame = pd.read_parquet(TRACE_SUMMARY)
    solved = frame[frame.earliest_correct_step.notna()]

    summary = {
        "n_traces": int(len(frame)),
        "final_accuracy": round(float(frame.final_correct.mean()), 4),
        "mean_total_tokens": round(float(frame.total_tokens.mean()), 1),
        "mean_steps": round(float(frame.n_steps.mean()), 2),
        "solvable_fraction": round(float(len(solved) / max(len(frame), 1)), 4),
    }

    if not solved.empty:
        summary["oracle_mean_tokens"] = round(float(solved.oracle_tokens.mean()), 1)
        summary["oracle_token_saving_pct"] = round(
            100 * (1 - solved.oracle_tokens.mean() / solved.total_tokens.mean()), 1
        )
        summary["mean_earliest_correct_step"] = round(
            float(solved.earliest_correct_step.mean()), 2
        )

    path = paths.RESULTS / "phase3_summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
