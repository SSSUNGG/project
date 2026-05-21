"""Global seed utilities for reproducibility."""
from __future__ import annotations

import random

import numpy as np
import torch


def set_global_seed(seed: int) -> None:
    """Set all RNG seeds for reproducibility.

    Must be called at the start of every script.
    env.reset() must still pass seed explicitly — this does NOT set env seeds.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
