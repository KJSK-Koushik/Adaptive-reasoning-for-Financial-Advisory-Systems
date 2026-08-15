from __future__ import annotations

from adaptive_reasoning import paths


def test_root_is_the_repository_root():
    assert (paths.ROOT / "configs" / "default.yaml").exists()
    assert (paths.ROOT / "src" / "adaptive_reasoning").is_dir()


def test_raw_sources_cover_every_documented_dataset():
    expected = {"finqa", "tatqa", "convfinqa", "phrasebank", "paysim", "german_credit"}
    assert set(paths.RAW_SOURCES) == expected


def test_all_dirs_live_under_root():
    for d in paths.ALL_DIRS:
        assert paths.ROOT in d.parents or d == paths.ROOT


def test_ensure_dirs_is_idempotent():
    paths.ensure_dirs()
    assert paths.ensure_dirs() == []
