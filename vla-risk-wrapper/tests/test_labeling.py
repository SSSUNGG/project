"""Tests for horizon-based labeling (spec §5.4)."""
from __future__ import annotations

import numpy as np
import pytest

from src.detector.labeling import horizon_label


def test_horizon_labeling_success_episode():
    """All labels are 0 for a successful episode."""
    labels = horizon_label(success=True, fail_step=-1, episode_length=50, H=10)
    assert labels.shape == (50,)
    assert labels.sum() == 0, "Success episode must have all labels = 0"


def test_horizon_labeling_failure_episode():
    """fail_step=70, H=20 → steps 50-70 get label 1."""
    labels = horizon_label(success=False, fail_step=70, episode_length=100, H=20)
    assert labels.shape == (100,)
    # Steps 50..70 inclusive should be 1
    assert labels[50:71].sum() == 21, f"Expected 21 ones, got {labels[50:71].sum()}"
    assert labels[:50].sum() == 0
    assert labels[71:].sum() == 0


def test_horizon_labeling_short_episode_fail_early():
    """fail_step=5, H=20 → steps 0-5 get label 1, no negative indexing."""
    labels = horizon_label(success=False, fail_step=5, episode_length=20, H=20)
    assert labels.shape == (20,)
    # max(0, 5-20) = 0, so steps 0..5 inclusive = 6 ones
    assert labels[0:6].sum() == 6, f"Expected 6 ones, got {labels[0:6].sum()}"
    assert labels[6:].sum() == 0
    # No negative indexing
    assert labels.min() >= 0


def test_horizon_labeling_boundary_at_zero():
    """fail_step=0, H=10 → only step 0 gets label 1."""
    labels = horizon_label(success=False, fail_step=0, episode_length=10, H=10)
    assert labels[0] == 1
    assert labels[1:].sum() == 0


def test_horizon_labeling_exact_H_window():
    """Labels exactly fill [fail_step-H, fail_step]."""
    H = 10
    fail = 50
    length = 100
    labels = horizon_label(success=False, fail_step=fail, episode_length=length, H=H)
    expected_ones = H + 1  # inclusive range
    assert labels.sum() == expected_ones
