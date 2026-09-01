"""Phase 2b - run the reasoning model ``k`` times per question.

This is the GPU-bound half of Phase 2 and, per ``docs/PHASE3_REMOTE.md``, it is folded
into the same remote job as Phase 3 rather than run separately.

Design points that matter when this runs on a free Kaggle session:

* **Checkpointing.** Results are flushed to a parquet shard every
  ``traces.checkpoint_every`` questions, and completed question ids are skipped on
  restart. A killed session costs minutes, not hours.
* **Flattened batching.** All ``k`` samples of all questions go into one flat list
  before batching, so a batch is always full. Sampling one question ``k`` times per
  batch would waste most of the GPU when ``k`` is small.
* **Reduced token cap.** ``difficulty.sample_max_new_tokens`` is lower than the full
  reasoning budget: we only need to know *whether* the model arrives, not to watch the
  whole journey.
"""

from __future__ import annotations

import json
from pathlib import Path

from .. import paths
from ..config import Config
from ..grading import is_correct
from ..llm import ReasoningLLM
from ..logging_utils import get_logger
from ..prompting import build_messages, count_reasoning_tokens, extract_answer
from ..schema import QARecord
from .labeling import SampleOutcome

log = get_logger("difficulty.sampling")

SAMPLES_PATH = paths.DATA_PROCESSED / "difficulty_samples.parquet"
_CHECKPOINT_DIR = paths.DATA_PROCESSED / "_difficulty_shards"


def _completed_ids(shard_dir: Path) -> set[str]:
    """Question ids already sampled in a previous session."""
    if not shard_dir.exists():
        return set()
    import pandas as pd

    done: set[str] = set()
    for shard in sorted(shard_dir.glob("shard_*.parquet")):
        try:
            done.update(pd.read_parquet(shard, columns=["question_id"])["question_id"].tolist())
        except Exception as exc:  # noqa: BLE001 - a half-written shard must not be fatal
            log.warning("ignoring unreadable shard %s: %s", shard.name, exc)
    return done


def _flush(rows: list[dict], shard_dir: Path, index: int) -> None:
    import pandas as pd

    shard_dir.mkdir(parents=True, exist_ok=True)
    path = shard_dir / f"shard_{index:05d}.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)
    log.info("checkpoint: wrote %s (%d rows)", path.name, len(rows))


def sample_questions(
    records: list[QARecord],
    cfg: Config,
    llm: ReasoningLLM | None = None,
    resume: bool = True,
) -> list[SampleOutcome]:
    """Run ``k`` sampled generations for each record and grade every one."""
    llm = llm or ReasoningLLM(cfg)
    k = cfg.difficulty.k_samples
    max_new = cfg.difficulty.sample_max_new_tokens

    done = _completed_ids(_CHECKPOINT_DIR) if resume else set()
    todo = [r for r in records if r.id not in done]
    if done:
        log.info("resuming: %d questions already sampled, %d remaining", len(done), len(todo))
    if not todo:
        log.info("nothing to sample")
        return load_outcomes()

    # Flatten to (record, sample_index) so every batch is full.
    prompts: list[str] = []
    owners: list[QARecord] = []
    for record in todo:
        rendered = llm.render(build_messages(record, cfg.prompting.few_shot))
        for _ in range(k):
            prompts.append(rendered)
            owners.append(record)

    log.info(
        "sampling %d questions x k=%d = %d generations (cap %d tokens)",
        len(todo), k, len(prompts), max_new,
    )

    rows: list[dict] = []
    pending: list[dict] = []
    shard_index = len(list(_CHECKPOINT_DIR.glob("shard_*.parquet"))) if _CHECKPOINT_DIR.exists() else 0
    checkpoint_every = max(1, cfg.traces.checkpoint_every) * k
    tokenizer = llm.tokenizer

    for i, generation in llm.batched(
        prompts,
        max_new_tokens=max_new,
        temperature=cfg.difficulty.sampling_temperature,
        do_sample=True,
    ):
        record = owners[i]
        answer = extract_answer(generation.text)
        row = {
            "question_id": record.id,
            "source": record.source,
            "answer": answer,
            "correct": is_correct(
                answer, record.gold_answer, str(record.answer_type), cfg.data.numeric_tolerance
            ),
            "reasoning_tokens": count_reasoning_tokens(generation.text, tokenizer),
            "generated_tokens": generation.n_generated_tokens,
            "truncated": not generation.finished,
        }
        rows.append(row)
        pending.append(row)

        if len(pending) >= checkpoint_every:
            _flush(pending, _CHECKPOINT_DIR, shard_index)
            shard_index += 1
            pending = []

        if (i + 1) % (checkpoint_every) == 0:
            n_ok = sum(1 for r in rows if r["correct"])
            log.info("  %d/%d generations, running accuracy %.1f%%",
                     i + 1, len(prompts), 100 * n_ok / len(rows))

    if pending:
        _flush(pending, _CHECKPOINT_DIR, shard_index)

    consolidate()
    return load_outcomes()


def consolidate() -> Path:
    """Merge checkpoint shards into a single parquet file."""
    import pandas as pd

    shards = sorted(_CHECKPOINT_DIR.glob("shard_*.parquet")) if _CHECKPOINT_DIR.exists() else []
    if not shards:
        log.warning("no shards to consolidate")
        return SAMPLES_PATH

    frame = pd.concat([pd.read_parquet(s) for s in shards], ignore_index=True)
    SAMPLES_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(SAMPLES_PATH, index=False)
    log.info("consolidated %d shards -> %s (%d rows)", len(shards), SAMPLES_PATH.name, len(frame))
    return SAMPLES_PATH


def load_outcomes(path: Path | None = None) -> list[SampleOutcome]:
    """Read sampled outcomes back from disk."""
    import pandas as pd

    path = path or SAMPLES_PATH
    if not path.exists():
        return []
    frame = pd.read_parquet(path)
    return [
        SampleOutcome(
            question_id=row.question_id,
            answer=str(row.answer),
            correct=bool(row.correct),
            reasoning_tokens=int(row.reasoning_tokens),
        )
        for row in frame.itertuples()
    ]


def write_manifest(cfg: Config, n_questions: int, n_generations: int) -> None:
    """Record what produced these samples, so labels can be traced to a run."""
    manifest = {
        "model_id": cfg.llm.model_id,
        "k_samples": cfg.difficulty.k_samples,
        "temperature": cfg.difficulty.sampling_temperature,
        "max_new_tokens": cfg.difficulty.sample_max_new_tokens,
        "numeric_tolerance": cfg.data.numeric_tolerance,
        "n_questions": n_questions,
        "n_generations": n_generations,
    }
    path = paths.RESULTS / "difficulty_sampling_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log.info("wrote %s", path)
