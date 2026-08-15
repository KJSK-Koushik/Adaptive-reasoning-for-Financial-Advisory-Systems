"""Reproducibility helpers.

Results in the report must be regenerable, so every entry point seeds through here.
"""

from __future__ import annotations

import os
import random


def set_seed(seed: int, deterministic_torch: bool = False) -> None:
    """Seed python, numpy and torch (all devices)."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            # Slower, but makes DQN runs bit-for-bit repeatable.
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
