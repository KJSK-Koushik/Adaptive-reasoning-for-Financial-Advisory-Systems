"""Bundle the code and dataset into one zip for upload to Kaggle.

    python scripts/package_for_kaggle.py

Produces ``artifacts/kaggle/adaptive-reasoning.zip`` (~8 MB). Upload that as a private
Kaggle Dataset and attach it to the notebook - see docs/KAGGLE_SETUP.md.

The zip carries the *same* source tree that runs locally. The notebook is a thin
wrapper around ``scripts/run_pilot.py``, ``run_phase2.py`` and ``run_phase3.py`` rather
than a reimplementation, so remote results stay reproducible on this machine.
"""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from adaptive_reasoning import paths  # noqa: E402

OUT_DIR = paths.ARTIFACTS / "kaggle"
OUT_ZIP = OUT_DIR / "adaptive-reasoning.zip"

# Everything under these directories, minus the exclusions below.
CODE_DIRS = ["src", "configs", "scripts"]
EXTRA_FILES = ["requirements-gpu.txt", "README.md"]

EXCLUDE_PARTS = {"__pycache__", ".pytest_cache", ".ruff_cache", ".ipynb_checkpoints"}
EXCLUDE_SUFFIXES = {".pyc", ".pyo", ".log"}


def _keep(path: Path) -> bool:
    if any(part in EXCLUDE_PARTS for part in path.parts):
        return False
    return path.suffix not in EXCLUDE_SUFFIXES


def main() -> int:
    if not paths.UNIFIED_DATASET.exists():
        print(f"missing {paths.UNIFIED_DATASET}\nRun: python scripts/run_phase1.py")
        return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    n_files = 0

    with zipfile.ZipFile(OUT_ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
        for name in CODE_DIRS:
            root = paths.ROOT / name
            for path in sorted(root.rglob("*")):
                if path.is_file() and _keep(path):
                    zf.write(path, Path("project") / path.relative_to(paths.ROOT))
                    n_files += 1

        for name in EXTRA_FILES:
            path = paths.ROOT / name
            if path.exists():
                zf.write(path, Path("project") / name)
                n_files += 1

        # The dataset itself - the only large item.
        zf.write(
            paths.UNIFIED_DATASET,
            Path("project") / paths.UNIFIED_DATASET.relative_to(paths.ROOT),
        )
        n_files += 1

        # Difficulty labels, if Phase 2 has already produced any locally.
        if paths.DIFFICULTY_LABELS.exists():
            zf.write(
                paths.DIFFICULTY_LABELS,
                Path("project") / paths.DIFFICULTY_LABELS.relative_to(paths.ROOT),
            )
            n_files += 1

    size_mb = OUT_ZIP.stat().st_size / 1e6
    print("=" * 68)
    print("  KAGGLE PACKAGE READY")
    print("=" * 68)
    print(f"  file    {OUT_ZIP}")
    print(f"  size    {size_mb:.1f} MB")
    print(f"  files   {n_files}")
    print()
    print("  Next: upload this as a private Kaggle Dataset named")
    print("        'adaptive-reasoning-fas', then run notebooks/phase3_kaggle.ipynb.")
    print("        Full instructions: docs/KAGGLE_SETUP.md")
    print("=" * 68)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
