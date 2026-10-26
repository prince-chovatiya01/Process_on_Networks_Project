"""
Reproducible seed handling and random number generation utilities.
"""

import random
import numpy as np
from typing import Optional


def set_seed(seed: int = 42) -> None:
    """Set global random seeds for Python and NumPy."""
    random.seed(seed)
    np.random.seed(seed)


def get_rng(seed: Optional[int] = None) -> np.random.Generator:
    """Return a NumPy Generator initialized with the given seed."""
    if seed is None:
        seed = 42
    return np.random.default_rng(seed)
