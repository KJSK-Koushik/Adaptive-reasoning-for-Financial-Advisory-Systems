from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from adaptive_reasoning.config import load_config  # noqa: E402


@pytest.fixture(scope="session")
def cfg():
    return load_config()
