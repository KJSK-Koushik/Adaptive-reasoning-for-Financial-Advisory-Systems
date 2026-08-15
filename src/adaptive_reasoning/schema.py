"""Unified data schemas.

Every dataset - FinQA, TAT-QA, ConvFinQA, PhraseBank, PaySim, German Credit and
our synthetic generator - is converted into :class:`QARecord`. Nothing downstream
knows which source a question came from, which is what lets Phases 2-6 stay simple.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class Domain(StrEnum):
    INVESTMENT = "investment"      # synthetic finance math, portfolio questions
    REPORT_QA = "report_qa"        # FinQA, TAT-QA, ConvFinQA
    FRAUD = "fraud"                # PaySim
    RISK = "risk"                  # German Credit
    SENTIMENT = "sentiment"        # Financial PhraseBank


class AnswerType(StrEnum):
    NUMERIC = "numeric"            # graded with relative tolerance
    CATEGORICAL = "categorical"    # graded with normalised exact match


class Difficulty(StrEnum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class Split(StrEnum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class QARecord(BaseModel):
    """One question in the unified dataset. Phase 1 writes these to unified.parquet."""

    model_config = ConfigDict(extra="forbid", use_enum_values=True)

    id: str
    source: str                        # finqa | tatqa | convfinqa | phrasebank | paysim | ...
    domain: Domain
    question: str
    context: str = ""                  # table, report snippet, or transaction record
    gold_answer: str                   # always stored as a string; parsed per answer_type
    answer_type: AnswerType
    answer_options: list[str] = Field(default_factory=list)  # for categorical questions

    # Measured in Phase 2 by sampling the model k times. None until then.
    difficulty: Difficulty | None = None
    difficulty_score: float | None = None   # pass rate over k samples, 0..1

    # An *a priori* guess at difficulty, currently set only by the synthetic generator
    # from the number of arithmetic steps involved. Deliberately a separate field from
    # `difficulty`: intrinsic difficulty and model-perceived difficulty are different
    # things, and conflating them would teach the classifier the wrong notion. Keeping
    # both lets Phase 9 report how far apart they actually are.
    difficulty_prior: Difficulty | None = None

    split: Split | None = None

    def prompt(self) -> str:
        """Render the question the way the reasoning LLM will see it."""
        parts = []
        if self.context:
            parts.append(self.context.strip())
        parts.append(self.question.strip())
        if self.answer_type == AnswerType.CATEGORICAL and self.answer_options:
            parts.append("Options: " + ", ".join(self.answer_options))
        return "\n\n".join(parts)


class TraceStep(BaseModel):
    """One early-exit probe point inside a reasoning trace.

    This is the raw material for the RL dataset: it records what *would* have
    happened had the model stopped here.
    """

    model_config = ConfigDict(extra="forbid")

    question_id: str
    step_index: int
    tokens_so_far: int
    step_text: str                     # reasoning generated since the previous step

    # What we would have answered if forced to stop at this point.
    probe_answer: str
    probe_correct: bool

    # Signals that make up the DQN state.
    confidence: float                  # mean probability of the answer tokens
    min_token_confidence: float
    entropy: float                     # mean entropy over this step's tokens
    answer_changed: bool               # differs from the previous step's probe answer

    is_terminal: bool = False          # model emitted its own stop, or hit the budget


class Trace(BaseModel):
    """A full reasoning trace for one question."""

    model_config = ConfigDict(extra="forbid")

    question_id: str
    difficulty: Difficulty | None = None
    total_tokens: int
    final_answer: str
    final_correct: bool
    steps: list[TraceStep]

    @property
    def earliest_correct_step(self) -> int | None:
        """Index of the first step where stopping would have been correct.

        This defines the oracle baseline and the headroom the DQN is chasing.
        """
        for step in self.steps:
            if step.probe_correct:
                return step.step_index
        return None


# Column order for the unified parquet file, so the schema is stable on disk.
UNIFIED_COLUMNS = [
    "id",
    "source",
    "domain",
    "question",
    "context",
    "gold_answer",
    "answer_type",
    "answer_options",
    "difficulty",
    "difficulty_score",
    "difficulty_prior",
    "split",
]

TRACE_STEP_COLUMNS = [
    "question_id",
    "step_index",
    "tokens_so_far",
    "step_text",
    "probe_answer",
    "probe_correct",
    "confidence",
    "min_token_confidence",
    "entropy",
    "answer_changed",
    "is_terminal",
]
