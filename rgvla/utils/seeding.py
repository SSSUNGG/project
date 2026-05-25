import numpy as np
from typing import List, Tuple


def make_seed_grid(n_seeds: int, n_episodes_per_seed: int) -> List[Tuple[int, int]]:
    return [(s, ep) for s in range(n_seeds) for ep in range(n_episodes_per_seed)]


def episode_seed(seed: int, episode: int) -> int:
    """(seed, episode) → unique integer seed for env.reset()."""
    return seed * 100_000 + episode
