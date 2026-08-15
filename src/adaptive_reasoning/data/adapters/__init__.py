"""One adapter per source. Each exposes ``load(cfg) -> list[QARecord]``."""

from . import convfinqa, finqa, german_credit, paysim, phrasebank, tatqa

ADAPTERS = {
    "finqa": finqa,
    "tatqa": tatqa,
    "convfinqa": convfinqa,
    "phrasebank": phrasebank,
    "paysim": paysim,
    "german_credit": german_credit,
}

__all__ = ["ADAPTERS", "convfinqa", "finqa", "german_credit", "paysim", "phrasebank", "tatqa"]
