"""Dataset acquisition.

Deliberately does **not** use ``datasets.load_dataset``: as of ``datasets`` 3.x the
library refuses to run repository loading scripts, which breaks FinQA
(``ibm-research/finqa``) and Financial PhraseBank (``takala/financial_phrasebank``).
Both still publish their raw archives, so we fetch those directly and parse them
ourselves. That also pins us to the exact upstream files, which is better for
reproducibility than whatever a loader script decides to do.

Everything lands under ``data/raw/<source>/`` and is skipped if already present.
"""

from __future__ import annotations

import urllib.error
import urllib.request
import zipfile
from pathlib import Path

from .. import paths
from ..logging_utils import get_logger

log = get_logger("data.download")

_USER_AGENT = "adaptive-reasoning-fas/0.1 (academic research)"

FINQA_BASE = "https://raw.githubusercontent.com/czyssrs/FinQA/main/dataset/"
FINQA_FILES = ["train.json", "dev.json", "test.json"]

TATQA_REPO = "next-tat/TAT-QA"
TATQA_FILES = ["tatqa_dataset_train.json", "tatqa_dataset_dev.json"]

PHRASEBANK_REPO = "takala/financial_phrasebank"
PHRASEBANK_ARCHIVE = "data/FinancialPhraseBank-v1.0.zip"

# UCI serves these over plain HTTP with no authentication, so despite being listed as
# a "manual" source in the original plan it can simply be fetched.
GERMAN_CREDIT_BASE = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/"
)
GERMAN_CREDIT_FILES = ["german.data", "german.doc"]


def _download(url: str, dest: Path, force: bool = False) -> Path:
    """Fetch ``url`` to ``dest`` unless it already exists."""
    if dest.exists() and not force and dest.stat().st_size > 0:
        log.info("cached  %s (%.1f MB)", dest.name, dest.stat().st_size / 1e6)
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    log.info("fetching %s", url)
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        with urllib.request.urlopen(req, timeout=120) as resp, tmp.open("wb") as fh:
            while chunk := resp.read(1 << 20):
                fh.write(chunk)
    except urllib.error.URLError as exc:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"failed to download {url}: {exc}") from exc
    tmp.replace(dest)
    log.info("saved   %s (%.1f MB)", dest.name, dest.stat().st_size / 1e6)
    return dest


def download_finqa(force: bool = False) -> list[Path]:
    """FinQA, straight from the authors' GitHub repository (~100 MB total)."""
    out = paths.RAW_SOURCES["finqa"]
    return [_download(FINQA_BASE + f, out / f, force) for f in FINQA_FILES]


def download_tatqa(force: bool = False) -> list[Path]:
    """TAT-QA raw JSON from the HuggingFace repo.

    Only train and dev are fetched: the public test split ships without answers.
    """
    from huggingface_hub import hf_hub_download

    out = paths.RAW_SOURCES["tatqa"]
    out.mkdir(parents=True, exist_ok=True)
    results = []
    for name in TATQA_FILES:
        dest = out / name
        if dest.exists() and not force and dest.stat().st_size > 0:
            log.info("cached  %s", name)
            results.append(dest)
            continue
        log.info("fetching %s from %s", name, TATQA_REPO)
        src = hf_hub_download(TATQA_REPO, name, repo_type="dataset")
        dest.write_bytes(Path(src).read_bytes())
        results.append(dest)
    return results


def download_phrasebank(force: bool = False) -> Path:
    """Financial PhraseBank, extracted from the v1.0 zip on the Hub.

    Returns the directory holding the ``Sentences_*Agree.txt`` files.
    """
    from huggingface_hub import hf_hub_download

    out = paths.RAW_SOURCES["phrasebank"]
    out.mkdir(parents=True, exist_ok=True)

    marker = out / "Sentences_75Agree.txt"
    if marker.exists() and not force:
        log.info("cached  Financial PhraseBank")
        return out

    log.info("fetching %s from %s", PHRASEBANK_ARCHIVE, PHRASEBANK_REPO)
    archive = hf_hub_download(PHRASEBANK_REPO, PHRASEBANK_ARCHIVE, repo_type="dataset")

    with zipfile.ZipFile(archive) as zf:
        for member in zf.namelist():
            name = Path(member).name
            # Skip macOS AppleDouble sidecars (._Foo.txt) present in the archive.
            if member.endswith(".txt") and not name.startswith("._"):
                # Flatten: the zip nests everything under a version directory.
                target = out / name
                target.write_bytes(zf.read(member))
                log.info("extracted %s", target.name)
    return out


def download_german_credit(force: bool = False) -> list[Path]:
    """Statlog German Credit from the UCI archive. ``german.doc`` decodes the codes."""
    out = paths.RAW_SOURCES["german_credit"]
    return [
        _download(GERMAN_CREDIT_BASE + f, out / f, force) for f in GERMAN_CREDIT_FILES
    ]


# Sources the user must place by hand, with the instruction shown on failure.
MANUAL_SOURCES = {
    "convfinqa": (
        ["train.json", "dev.json"],
        "Clone https://github.com/czyssrs/ConvFinQA and unzip data.zip into "
        "data/raw/convfinqa/",
    ),
    "paysim": (
        ["PS_20174392719_1491204439457_log.csv"],
        "Download from https://www.kaggle.com/datasets/ealaxi/paysim1 and place the "
        "CSV in data/raw/paysim/",
    ),
}


def check_manual(source: str) -> Path | None:
    """Return the first present expected file for a manual source, else ``None``."""
    expected, _ = MANUAL_SOURCES[source]
    folder = paths.RAW_SOURCES[source]
    for name in expected:
        candidate = folder / name
        if candidate.exists():
            return candidate
    # Be forgiving about the PaySim filename - people rename it.
    if source == "paysim":
        csvs = sorted(folder.glob("*.csv"))
        if csvs:
            return csvs[0]
    return None


def missing_manual() -> dict[str, str]:
    """Map each absent manual source to its download instruction."""
    return {
        src: instruction
        for src, (_, instruction) in MANUAL_SOURCES.items()
        if check_manual(src) is None
    }


def download_all(force: bool = False) -> dict[str, str]:
    """Fetch every automatic source. Returns the still-missing manual sources."""
    download_finqa(force)
    download_tatqa(force)
    download_phrasebank(force)
    download_german_credit(force)
    return missing_manual()
