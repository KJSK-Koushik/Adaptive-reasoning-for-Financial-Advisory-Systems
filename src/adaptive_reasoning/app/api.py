"""Phase 8 - the advisory service.

A FastAPI wrapper around the Phase 7 controller. It answers financial questions and,
for every answer, reports how much reasoning it skipped and whether skipping changed
the answer - which is the point of the project and not a detail to bury.

Answers are served in **replay mode**: the reasoning was generated during Phase 3 and
recorded, so the service returns exactly what the model produced, without a GPU. Every
response says ``mode: replay`` rather than implying live generation. The controller is
the same code either way, so nothing here is a mock-up of the real path - it *is* the
real path, fed from a recording.

Scope: the service performs analysis - computations, risk scores, fraud likelihood,
factual question answering over financial documents. It does not give personalised
investment advice, and every answer carries the disclaimer from ``serve.disclaimer``.
"""

from __future__ import annotations

import json
from functools import lru_cache

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from .. import paths
from ..config import Config, load_config
from ..logging_utils import get_logger
from ..rl.dataset import ACTION_STOP
from ..serve.controller import (
    AdaptiveController,
    ReplaySource,
    always_continue_policy,
    load_policy,
    training_budget,
    training_min_steps,
)

log = get_logger("app.api")


# --------------------------------------------------------------------------- #
# response models
# --------------------------------------------------------------------------- #
class QuestionSummary(BaseModel):
    id: str
    domain: str
    question: str
    answer_type: str
    difficulty: str | None = None
    has_trace: bool


class StepView(BaseModel):
    step: int
    action: str
    reason: str = ""
    answer: str
    tokens: int
    confidence: float
    entropy: float
    text: str = ""


class AnswerResponse(BaseModel):
    question_id: str
    question: str
    context: str = ""
    answer: str
    reference_answer: str = Field(description="the dataset's gold answer, for the demo only")
    correct: bool | None = None

    policy: str
    mode: str = "replay"
    stopped_at_step: int
    stop_reason: str

    tokens_used: int
    tokens_if_unrestricted: int
    tokens_saved: int
    token_reduction_pct: float
    latency_seconds: float
    latency_saved_seconds: float

    full_reasoning_answer: str
    full_reasoning_correct: bool | None = None
    answer_changed: bool
    early_stop_helped: bool | None = Field(
        default=None,
        description="True when stopping early was right and full reasoning was wrong",
    )

    steps: list[StepView] = Field(default_factory=list)
    disclaimer: str


class StatsResponse(BaseModel):
    dataset_questions: int
    traced_questions: int
    test_questions: int
    policies: list[str]
    evaluation: dict
    disclaimer: str


# --------------------------------------------------------------------------- #
# the data and models the service needs
# --------------------------------------------------------------------------- #
class DemoStore:
    """Everything loaded once at startup, so a request is pure computation."""

    def __init__(self, cfg: Config):
        import pandas as pd

        self.cfg = cfg
        self.budget = training_budget(cfg)

        if not paths.TRACE_DATASET.exists():
            raise FileNotFoundError(
                f"{paths.TRACE_DATASET} not found - run Phase 3 before serving")

        self.questions = pd.read_parquet(paths.UNIFIED_DATASET).set_index("id")
        self.traces = pd.read_parquet(paths.TRACE_DATASET)
        transitions = pd.read_parquet(paths.RL_TRANSITIONS)

        stops = transitions[transitions.action == ACTION_STOP]
        self.difficulty_vectors = {
            str(qid): np.asarray(g.sort_values("step_index").state.iloc[0],
                                 dtype=np.float32)[:3]
            for qid, g in stops.groupby("question_id", sort=False)
        }
        self.test_ids = set(
            transitions[transitions.split == "test"].question_id.astype(str))
        self.traced_ids = set(self.traces.question_id.astype(str))

        self.correct_at = {
            (str(r.question_id), int(r.step_index)): bool(r.probe_correct)
            for r in self.traces.itertuples()
        }

        self.policies = {}
        for kind in ("dqn", "bc"):
            try:
                self.policies[kind] = (load_policy(cfg, kind=kind),
                                       training_min_steps(cfg, kind))
            except Exception as exc:                      # noqa: BLE001
                log.warning("policy %s unavailable: %s", kind, exc)

        log.info("serving %d traced questions, policies: %s",
                 len(self.traced_ids), sorted(self.policies))

    def question(self, question_id: str):
        if question_id not in self.questions.index:
            raise HTTPException(404, f"unknown question_id {question_id!r}")
        return self.questions.loc[question_id]


@lru_cache(maxsize=1)
def get_store(experiment: str | None = None) -> DemoStore:
    return DemoStore(load_config(experiment))


# --------------------------------------------------------------------------- #
# the service
# --------------------------------------------------------------------------- #
def create_app(cfg: Config | None = None, experiment: str | None = "reported") -> FastAPI:
    cfg = cfg or load_config(experiment)
    app = FastAPI(
        title=cfg.app.title,
        version="0.1.0",
        description=(
            "Adaptive reasoning termination for financial question answering. "
            "Analysis only - not personalised investment advice."
        ),
    )

    def store() -> DemoStore:
        return get_store(experiment)

    @app.get("/health")
    def health() -> dict:
        s = store()
        return {"status": "ok", "traced_questions": len(s.traced_ids),
                "policies": sorted(s.policies), "budget_tokens": s.budget}

    @app.get("/questions", response_model=list[QuestionSummary])
    def questions(
        limit: int = Query(20, ge=1, le=200),
        domain: str | None = None,
        test_only: bool = True,
    ) -> list[QuestionSummary]:
        """Questions available to demonstrate. Traced questions only - the rest have
        no recorded reasoning to replay."""
        s = store()
        ids = s.test_ids if test_only else s.traced_ids
        frame = s.questions[s.questions.index.isin(ids)]
        if domain:
            frame = frame[frame.domain == domain]
        out = []
        for question_id, row in frame.head(limit).iterrows():
            out.append(QuestionSummary(
                id=str(question_id),
                domain=str(row.domain),
                question=str(row.question)[:300],
                answer_type=str(row.answer_type),
                difficulty=None if row.get("difficulty") is None
                else str(row.get("difficulty")),
                has_trace=str(question_id) in s.traced_ids,
            ))
        return out

    @app.post("/ask", response_model=AnswerResponse)
    def ask(question_id: str, policy: str = "dqn", include_steps: bool = True):
        """Answer one question, and report what stopping early cost or saved."""
        s = store()
        if policy not in s.policies:
            raise HTTPException(
                400, f"unknown policy {policy!r}, available: {sorted(s.policies)}")
        if question_id not in s.traced_ids:
            raise HTTPException(
                404, f"no recorded reasoning for {question_id!r} - "
                     "only traced questions can be replayed")

        row = s.question(question_id)
        decide, floor = s.policies[policy]
        vector = s.difficulty_vectors.get(question_id)
        source = ReplaySource.from_frame(s.traces, question_id)

        adaptive = AdaptiveController(
            s.cfg, decide, difficulty_vector=vector, budget=s.budget, min_steps=floor
        ).run(source)
        full = AdaptiveController(
            s.cfg, always_continue_policy(), difficulty_vector=vector, budget=s.budget
        ).run(source)

        was_right = s.correct_at.get((question_id, adaptive.stop_step))
        full_right = s.correct_at.get((question_id, full.stop_step))
        helped = None
        if was_right is not None and full_right is not None:
            helped = bool(was_right and not full_right)

        saved = full.tokens_used - adaptive.tokens_used
        per_second = 91.5                       # measured by the Phase 3 pilot

        return AnswerResponse(
            question_id=question_id,
            question=str(row.question),
            context=str(row.context or "")[:2000],
            answer=adaptive.answer,
            reference_answer=str(row.gold_answer),
            correct=was_right,
            policy=policy,
            stopped_at_step=adaptive.stop_step,
            stop_reason=adaptive.stop_reason,
            tokens_used=adaptive.tokens_used,
            tokens_if_unrestricted=full.tokens_used,
            tokens_saved=saved,
            token_reduction_pct=round(100.0 * saved / max(full.tokens_used, 1), 1),
            latency_seconds=round(adaptive.tokens_used / per_second, 2),
            latency_saved_seconds=round(saved / per_second, 2),
            full_reasoning_answer=full.answer,
            full_reasoning_correct=full_right,
            answer_changed=adaptive.answer.strip() != full.answer.strip(),
            early_stop_helped=helped,
            steps=[
                StepView(step=d.step_index, action=d.action, reason=d.reason,
                         answer=d.answer, tokens=d.tokens_so_far,
                         confidence=round(d.confidence, 4),
                         entropy=round(d.entropy, 4),
                         text=d.step_text if s.cfg.app.show_reasoning_trace else "")
                for d in adaptive.decisions
            ] if include_steps else [],
            disclaimer=adaptive.disclaimer,
        )

    @app.get("/stats", response_model=StatsResponse)
    def stats() -> StatsResponse:
        """Headline evaluation numbers, read from the phase summaries."""
        s = store()
        evaluation: dict = {}
        summary = paths.RESULTS / "phase6_summary.json"
        if summary.exists():
            data = json.loads(summary.read_text(encoding="utf-8"))["results"]
            for name in ("full_reasoning", "fixed_step_matched", "behaviour_cloning",
                         "dqn", "oracle"):
                if name in data:
                    evaluation[name] = {
                        "accuracy": data[name]["accuracy"],
                        "mean_tokens": data[name]["mean_tokens"],
                        "token_reduction_pct": data[name]["token_reduction_pct"],
                    }
        return StatsResponse(
            dataset_questions=len(s.questions),
            traced_questions=len(s.traced_ids),
            test_questions=len(s.test_ids),
            policies=sorted(s.policies),
            evaluation=evaluation,
            disclaimer=s.cfg.serve.disclaimer.strip(),
        )

    return app
