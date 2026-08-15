"""Phase 1 - build the unified dataset.

    python scripts/run_phase1.py                # download + build
    python scripts/run_phase1.py --no-download  # build from what is already on disk
    python scripts/run_phase1.py --experiment smoke
    python scripts/run_phase1.py --inspect convfinqa
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from adaptive_reasoning import paths  # noqa: E402
from adaptive_reasoning.config import load_config  # noqa: E402
from adaptive_reasoning.data import build, download  # noqa: E402
from adaptive_reasoning.logging_utils import setup_logging  # noqa: E402
from adaptive_reasoning.seeding import set_seed  # noqa: E402


def inspect_source(name: str) -> int:
    """Print the raw structure of a downloaded source - useful for manual datasets."""
    folder = paths.RAW_SOURCES.get(name)
    if folder is None:
        print(f"unknown source {name!r}; expected one of {sorted(paths.RAW_SOURCES)}")
        return 1

    files = sorted(p for p in folder.iterdir() if p.is_file() and p.suffix in {".json", ".csv", ".data", ".txt"})
    if not files:
        print(f"no data files in {folder}")
        return 1

    for path in files:
        print("=" * 70)
        print(f"{path.name}  ({path.stat().st_size / 1e6:.1f} MB)")
        print("=" * 70)
        if path.suffix == ".json":
            data = json.loads(path.read_text(encoding="utf-8"))
            print(f"  type: {type(data).__name__}, length: {len(data)}")
            probe = data[0] if isinstance(data, list) and data else data
            if isinstance(probe, dict):
                print(f"  top-level keys: {sorted(probe)}")
                for key in ("qa", "annotation"):
                    if isinstance(probe.get(key), dict):
                        print(f"  {key} keys: {sorted(probe[key])}")
        else:
            for i, line in enumerate(path.read_text(encoding="latin-1").splitlines()[:3]):
                print(f"  {i}: {line[:200]}")
        print()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the unified Phase 1 dataset")
    parser.add_argument("--experiment", help="config in configs/experiment/")
    parser.add_argument("--no-download", action="store_true", help="skip network fetches")
    parser.add_argument("--force-download", action="store_true", help="re-fetch cached files")
    parser.add_argument("--dry-run", action="store_true", help="build but do not write")
    parser.add_argument("--inspect", metavar="SOURCE", help="print raw structure and exit")
    args = parser.parse_args()

    if args.inspect:
        return inspect_source(args.inspect)

    cfg = load_config(args.experiment)
    setup_logging(cfg.logging.level, cfg.logging.to_file, cfg.logging.rich_console, "phase1")
    set_seed(cfg.project.seed)
    paths.ensure_dirs()

    if not args.no_download:
        download.download_all(force=args.force_download)

    missing = download.missing_manual()

    records, summary = build.build(cfg, write=not args.dry_run)

    print()
    print("=" * 68)
    print("  PHASE 1 SUMMARY")
    print("=" * 68)
    print(f"  total records      {summary['total']:,}")
    print(f"  duplicates removed {summary['duplicates_removed']:,}")
    print(f"  mean question len  {summary['mean_question_chars']} chars")
    print(f"  mean context len   {summary['mean_context_chars']} chars")
    print("\n  by source:")
    for source, count in sorted(summary["by_source"].items(), key=lambda kv: -kv[1]):
        print(f"    {source:<16} {count:>7,}")
    print("\n  by domain:")
    for domain, count in sorted(summary["by_domain"].items(), key=lambda kv: -kv[1]):
        print(f"    {domain:<16} {count:>7,}")
    print("\n  by answer type:")
    for kind, count in sorted(summary["by_answer_type"].items()):
        print(f"    {kind:<16} {count:>7,}")
    print("\n  by split:")
    for split, count in sorted(summary["by_split"].items()):
        print(f"    {split:<16} {count:>7,}")

    if missing:
        print("\n  MISSING MANUAL SOURCES (built without them):")
        for source, instruction in missing.items():
            print(f"    - {source}: {instruction}")

    if not args.dry_run:
        print(f"\n  wrote {paths.UNIFIED_DATASET}")
    print("=" * 68)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
