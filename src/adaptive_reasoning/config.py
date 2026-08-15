"""Typed configuration loader.

``configs/default.yaml`` is the single source of truth. Experiment configs in
``configs/experiment/`` are deep-merged over it, so an experiment only needs to
state what it changes.

Validation happens at load time via pydantic, which means a typo in a YAML key
fails immediately with a clear message rather than silently doing the wrong
thing eight hours into a trace-generation run.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from . import paths


class _Base(BaseModel):
    # Reject unknown keys - catches YAML typos early.
    model_config = ConfigDict(extra="forbid")


# --------------------------------------------------------------------------- #
# Sections
# --------------------------------------------------------------------------- #
class ProjectCfg(_Base):
    name: str
    seed: int = 42
    device: Literal["cpu", "cuda", "xpu", "mps", "auto"] = "auto"


class SplitCfg(_Base):
    train: float
    val: float
    test: float
    stratify_by: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _sums_to_one(self):
        total = self.train + self.val + self.test
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"data.splits must sum to 1.0, got {total}")
        return self


class DataCfg(_Base):
    sample_sizes: dict[str, int | None]
    phrasebank_config: str
    paysim_fraud_ratio: float = Field(ge=0.0, le=1.0)
    splits: SplitCfg
    numeric_tolerance: float = Field(gt=0.0)
    max_context_chars: int = Field(gt=0)
    use_gold_evidence: bool = True
    exclude_protected_attributes: bool = True


class DifficultyClassifierCfg(_Base):
    encoder: str
    head: Literal["lightgbm", "logistic"]
    max_seq_length: int
    context_chars: int = Field(ge=0)
    n_estimators: int
    learning_rate: float
    class_weight: str


class FromTracesCfg(_Base):
    easy_min_correct_ratio: float = Field(ge=0.0, le=1.0)
    hard_max_correct_ratio: float = Field(ge=0.0, le=1.0)
    long_reasoning_ratio: float = Field(ge=0.0, le=1.0)
    unstable_answer_ratio: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _thresholds_ordered(self):
        if self.hard_max_correct_ratio >= self.easy_min_correct_ratio:
            raise ValueError(
                "difficulty.from_traces.hard_max_correct_ratio must be below "
                "easy_min_correct_ratio"
            )
        return self


class DifficultyCfg(_Base):
    k_samples: int = Field(ge=1)
    sampling_temperature: float
    sample_max_new_tokens: int = Field(gt=0)
    n_questions: int | None = None
    easy_min_pass_rate: float = Field(ge=0.0, le=1.0)
    hard_max_pass_rate: float = Field(ge=0.0, le=1.0)
    long_reasoning_token_threshold: int
    from_traces: FromTracesCfg
    classifier: DifficultyClassifierCfg

    @model_validator(mode="after")
    def _thresholds_ordered(self):
        if self.hard_max_pass_rate >= self.easy_min_pass_rate:
            raise ValueError(
                "difficulty.hard_max_pass_rate must be below easy_min_pass_rate"
            )
        return self


class LLMCfg(_Base):
    model_id: str
    dtype: Literal["float16", "bfloat16", "float32"]
    load_in_4bit: bool
    max_new_tokens: int
    temperature: float
    top_p: float
    batch_size: int


class TraceCaptureCfg(_Base):
    answer_confidence: bool
    min_token_confidence: bool
    entropy: bool
    top_k_logprobs: int


class PilotCfg(_Base):
    n_questions: int = Field(gt=0)
    min_median_reasoning_tokens: int = Field(ge=0)
    min_answer_rate: float = Field(ge=0.0, le=1.0)


class TracesCfg(_Base):
    n_questions: int
    step_tokens: int
    step_delimiters: list[str]
    max_steps: int
    probe_prompt: str
    probe_max_tokens: int
    capture: TraceCaptureCfg
    checkpoint_every: int
    output_format: Literal["parquet", "jsonl"]
    pilot: PilotCfg


class TokenCostCfg(_Base):
    easy: float
    medium: float
    hard: float

    def for_difficulty(self, difficulty: str) -> float:
        try:
            return getattr(self, difficulty)
        except AttributeError as exc:
            raise ValueError(f"unknown difficulty {difficulty!r}") from exc


class RewardCfg(_Base):
    correct_bonus: float
    incorrect_penalty: float
    token_cost: TokenCostCfg
    cost_multiplier: float = Field(default=1.0, ge=0.0)
    stability_bonus: float


class DQNCfg(_Base):
    hidden_sizes: list[int]
    gamma: float = Field(gt=0.0, le=1.0)
    learning_rate: float
    batch_size: int
    buffer_size: int
    target_update_interval: int
    train_steps: int
    epsilon_start: float
    epsilon_end: float
    epsilon_decay_steps: int
    double_dqn: bool
    grad_clip: float
    eval_every: int


class SelectionCfg(_Base):
    objective: Literal["weighted_score", "accuracy_at_budget"] = "weighted_score"
    min_token_reduction: float = Field(default=45.0, ge=0.0, le=100.0)
    weight: float = 0.30


class RLCfg(_Base):
    difficulty_source: Literal["predicted", "true", "none"] = "predicted"
    state_features: list[str]
    reward: RewardCfg
    selection: SelectionCfg = Field(default_factory=SelectionCfg)
    dqn: DQNCfg

    @property
    def state_dim(self) -> int:
        return len(self.state_features)


class EvalCfg(_Base):
    baselines: list[str]
    fixed_budget_tokens: list[int]
    confidence_tau: list[float]
    entropy_tau: list[float]
    metrics: list[str]
    measure_energy: bool
    bootstrap_samples: int


class ServeCfg(_Base):
    host: str
    port: int
    stream: bool
    hard_token_cap: int
    min_steps_before_stop: int
    disclaimer: str


class AppCfg(_Base):
    title: str
    show_reasoning_trace: bool
    show_savings_panel: bool
    compare_against: str


class LoggingCfg(_Base):
    level: str
    to_file: bool
    rich_console: bool


class Config(_Base):
    project: ProjectCfg
    data: DataCfg
    difficulty: DifficultyCfg
    llm: LLMCfg
    traces: TracesCfg
    rl: RLCfg
    eval: EvalCfg
    serve: ServeCfg
    app: AppCfg
    logging: LoggingCfg


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into ``base`` without mutating either."""
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"config file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def parse_overrides(assignments: list[str] | None) -> dict[str, Any]:
    """Turn ``["llm.batch_size=32", "traces.n_questions=4000"]`` into a nested dict.

    Values are parsed as YAML, so ``32`` becomes an int, ``true`` a bool, ``null`` None
    and ``[1,2]`` a list. This exists so a remote run can be retuned from the command
    line without rebuilding and re-uploading the whole package.
    """
    out: dict[str, Any] = {}
    for assignment in assignments or []:
        if "=" not in assignment:
            raise ValueError(f"override must be key.path=value, got {assignment!r}")
        path, _, raw = assignment.partition("=")
        value = yaml.safe_load(raw)

        cursor = out
        parts = path.strip().split(".")
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[parts[-1]] = value
    return out


def load_config(experiment: str | None = None, overrides: dict | None = None) -> Config:
    """Load the default config, optionally layering an experiment and dict overrides.

    Args:
        experiment: name of a file in ``configs/experiment/`` (with or without ``.yaml``).
        overrides: last-word dictionary applied on top, handy for tests and sweeps.
    """
    merged = _read_yaml(paths.CONFIGS / "default.yaml")

    if experiment:
        name = experiment if experiment.endswith(".yaml") else f"{experiment}.yaml"
        merged = _deep_merge(merged, _read_yaml(paths.EXPERIMENT_CONFIGS / name))

    if overrides:
        merged = _deep_merge(merged, overrides)

    return Config(**merged)
