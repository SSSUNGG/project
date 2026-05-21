"""Bootstrap utilities."""
from __future__ import annotations

import numpy as np


def bootstrap_mean_ci(
    data: np.ndarray,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Compute bootstrap confidence interval for the mean.

    Args:
        data:             1-D array of observations.
        n_bootstrap:      Number of bootstrap resamples.
        confidence_level: E.g., 0.95 for 95% CI.
        seed:             RNG seed for reproducibility.

    Returns:
        (mean, ci_low, ci_high)
    """
    data = np.asarray(data, dtype=np.float64)
    observed_mean = float(data.mean())

    rng = np.random.default_rng(seed)
    boot_means = np.array([
        rng.choice(data, size=len(data), replace=True).mean()
        for _ in range(n_bootstrap)
    ])

    alpha = (1 - confidence_level) / 2
    ci_low = float(np.percentile(boot_means, 100 * alpha))
    ci_high = float(np.percentile(boot_means, 100 * (1 - alpha)))

    return observed_mean, ci_low, ci_high
