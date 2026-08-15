"""Canonical project paths.

Everything resolves from the repository root so scripts behave the same whether
they are launched from the root, from ``scripts/``, or from a notebook.
"""

from __future__ import annotations

from pathlib import Path

# src/adaptive_reasoning/paths.py -> src/adaptive_reasoning -> src -> root
ROOT = Path(__file__).resolve().parents[2]

CONFIGS = ROOT / "configs"
EXPERIMENT_CONFIGS = CONFIGS / "experiment"
DOCS = ROOT / "docs"
SCRIPTS = ROOT / "scripts"

DATA = ROOT / "data"
DATA_RAW = DATA / "raw"
DATA_INTERIM = DATA / "interim"
DATA_PROCESSED = DATA / "processed"

# One directory per source dataset, matching docs/DATASETS.md.
RAW_SOURCES = {
    "finqa": DATA_RAW / "finqa",
    "tatqa": DATA_RAW / "tatqa",
    "convfinqa": DATA_RAW / "convfinqa",
    "phrasebank": DATA_RAW / "phrasebank",
    "paysim": DATA_RAW / "paysim",
    "german_credit": DATA_RAW / "german_credit",
}

ARTIFACTS = ROOT / "artifacts"
TRACES = ARTIFACTS / "traces"
MODELS = ARTIFACTS / "models"
RESULTS = ARTIFACTS / "results"
LOGS = ARTIFACTS / "logs"

# Well-known output files, named here so no phase has to hardcode a string.
UNIFIED_DATASET = DATA_PROCESSED / "unified.parquet"
SPLITS_DIR = DATA_PROCESSED / "splits"
DIFFICULTY_LABELS = DATA_PROCESSED / "difficulty_labels.parquet"
DIFFICULTY_MODEL = MODELS / "difficulty_clf.joblib"
TRACE_DATASET = TRACES / "traces.parquet"
RL_TRANSITIONS = TRACES / "transitions.parquet"
DQN_POLICY = MODELS / "stopping_policy.pt"

ALL_DIRS: tuple[Path, ...] = (
    CONFIGS,
    EXPERIMENT_CONFIGS,
    DOCS,
    DATA_RAW,
    DATA_INTERIM,
    DATA_PROCESSED,
    SPLITS_DIR,
    *RAW_SOURCES.values(),
    ARTIFACTS,
    TRACES,
    MODELS,
    RESULTS,
    LOGS,
)


def ensure_dirs() -> list[Path]:
    """Create every project directory. Returns the ones that were newly made."""
    created = []
    for d in ALL_DIRS:
        if not d.exists():
            d.mkdir(parents=True, exist_ok=True)
            created.append(d)
        gitkeep = d / ".gitkeep"
        if not any(d.iterdir()) and not gitkeep.exists():
            gitkeep.touch()
    return created
