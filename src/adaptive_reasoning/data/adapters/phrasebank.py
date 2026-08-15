"""Financial PhraseBank adapter.

Format is one line per example, ``<sentence>@<label>``, encoded in Latin-1 (not UTF-8
- decoding as UTF-8 raises). Labels are ``positive`` / ``neutral`` / ``negative``.

This is the project's **Easy tier**, and it doubles as a sanity check on the whole
method: a sentiment judgement needs almost no reasoning, so a correctly trained
stopping policy must learn to halt within a handful of tokens here. If it does not,
something is wrong upstream.

Class balance at 75% annotator agreement is skewed - 2,146 neutral, 887 positive,
420 negative - so the builder caps the majority class rather than letting "always
neutral" become a viable shortcut.
"""

from __future__ import annotations

import random
from collections import Counter

from ... import paths
from ...config import Config
from ...logging_utils import get_logger
from ...schema import AnswerType, Domain, QARecord
from ..text_utils import clean

log = get_logger("data.phrasebank")

LABELS = ["positive", "neutral", "negative"]

_QUESTION_TEMPLATES = [
    "What is the sentiment of this financial statement for the company involved?",
    "From an investor's point of view, is the sentiment of this news positive, "
    "neutral, or negative?",
    "Classify the financial sentiment expressed in the sentence above.",
]


def load(cfg: Config) -> list[QARecord]:
    path = paths.RAW_SOURCES["phrasebank"] / f"Sentences_{_config_suffix(cfg)}.txt"
    if not path.exists():
        log.warning("PhraseBank file missing: %s (run the download step)", path.name)
        return []

    # The distributed files are Latin-1; UTF-8 decoding fails on them.
    lines = [ln for ln in path.read_bytes().decode("latin-1").splitlines() if ln.strip()]

    rng = random.Random(cfg.project.seed)
    records: list[QARecord] = []
    counts: Counter[str] = Counter()

    for i, line in enumerate(lines):
        sentence, _, label = line.rpartition("@")
        label = label.strip().lower()
        sentence = clean(sentence)
        if not sentence or label not in LABELS:
            continue
        counts[label] += 1
        records.append(
            QARecord(
                id=f"phrasebank::{i}",
                source="phrasebank",
                domain=Domain.SENTIMENT,
                question=rng.choice(_QUESTION_TEMPLATES),
                context=sentence,
                gold_answer=label,
                answer_type=AnswerType.CATEGORICAL,
                answer_options=LABELS,
            )
        )

    log.info("phrasebank: %d records %s", len(records), dict(counts))
    return records


def _config_suffix(cfg: Config) -> str:
    """Map ``sentences_75agree`` to the distributed filename ``Sentences_75Agree``."""
    name = cfg.data.phrasebank_config.removeprefix("sentences_")
    return name.replace("agree", "Agree").replace("all", "All")
