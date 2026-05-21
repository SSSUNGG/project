from __future__ import annotations

import numpy as np


def horizon_label(
    success: bool,
    fail_step: int,
    episode_length: int,
    H: int,
) -> np.ndarray:
    """Compute horizon-based binary labels for one episode.

    Args:
        success:        Whether the episode succeeded.
        fail_step:      Index of the terminal failure step (0-indexed). -1 if success.
        episode_length: Total number of steps in the episode.
        H:              Horizon window size (from config, task-specific).

    Returns:
        labels: np.ndarray of shape (episode_length,), dtype int32.
                1 means "within H steps of failure", 0 otherwise.

    Labeling rules (from spec §5.4):
        - success == True:  all labels = 0
        - success == False: steps in [max(0, fail_step - H), fail_step] get label 1
    """
    labels = np.zeros(episode_length, dtype=np.int32)

    if success:
        return labels

    if fail_step < 0 or fail_step >= episode_length:
        return labels

    start = max(0, fail_step - H)
    end = fail_step  # inclusive
    labels[start : end + 1] = 1
    return labels
