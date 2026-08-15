"""Create the project directory tree and drop READMEs into the raw data folders.

Run once after cloning:  python scripts/bootstrap.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from adaptive_reasoning import paths  # noqa: E402

# What the user must place in each manual-download folder.
_RAW_NOTES = {
    "finqa": "Downloaded automatically in Phase 1 from HuggingFace (ibm-research/finqa).",
    "tatqa": "Downloaded automatically in Phase 1 from HuggingFace (next-tat/TAT-QA).",
    "phrasebank": (
        "Downloaded automatically in Phase 1 from HuggingFace "
        "(takala/financial_phrasebank, config sentences_75agree)."
    ),
    "convfinqa": (
        "MANUAL. Clone https://github.com/czyssrs/ConvFinQA and unzip data.zip here.\n"
        "Expected files: train.json, dev.json, train_turn.json, dev_turn.json"
    ),
    "paysim": (
        "MANUAL. Download from https://www.kaggle.com/datasets/ealaxi/paysim1 "
        "(free Kaggle account required).\n"
        "Expected file: PS_20174392719_1491204439457_log.csv"
    ),
    "german_credit": (
        "Downloaded automatically in Phase 1 from the UCI archive "
        "(german.data + german.doc)."
    ),
}


def main() -> int:
    created = paths.ensure_dirs()

    for name, note in _RAW_NOTES.items():
        readme = paths.RAW_SOURCES[name] / "README.md"
        if not readme.exists():
            readme.write_text(f"# {name}\n\n{note}\n", encoding="utf-8")

    print(f"Project root: {paths.ROOT}")
    if created:
        print(f"Created {len(created)} directories:")
        for d in created:
            print(f"  + {d.relative_to(paths.ROOT)}")
    else:
        print("All directories already present.")
    print("\nRaw data folders are annotated with download instructions.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
