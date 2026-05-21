"""Tests for src/eval/ module."""
from __future__ import annotations

import numpy as np
import pytest

from src.eval.metrics import (
    ece,
    lead_time_at_recall,
    paired_bootstrap_diff,
    pareto_frontier,
    step_auroc,
    success_rate_with_ci,
)
from src.eval.bootstrap import bootstrap_mean_ci


def test_paired_bootstrap_same_input_gives_zero_diff():
    """Paired diff of identical arrays is 0."""
    data = np.array([1, 0, 1, 1, 0, 0, 1, 0], dtype=float)
    mean_diff, ci_low, ci_high, p = paired_bootstrap_diff(data, data, n_bootstrap=500)
    assert abs(mean_diff) < 1e-10, f"Expected 0 diff, got {mean_diff}"


def test_pareto_frontier_simple_case():
    """Basic Pareto test: (10, 0.8) and (30, 0.5) — both on frontier."""
    # (latency, success) — minimize latency, maximize success
    points = [
        (10.0, 0.8),  # idx 0 — Pareto optimal (low latency + high success)
        (30.0, 0.5),  # idx 1 — dominated by idx 0 on both axes
        (5.0, 0.4),   # idx 2 — Pareto optimal (lowest latency)
        (50.0, 0.9),  # idx 3 — Pareto optimal (highest success)
    ]
    frontier = pareto_frontier(points)
    # idx 1 should be dominated (idx 0 has lower latency AND higher success)
    assert 0 in frontier
    assert 2 in frontier
    assert 3 in frontier
    assert 1 not in frontier


def test_lead_time_recovers_known_value():
    """Simple case: detector always fires at step 0 for fail_step=10."""
    probs_by_traj = [np.ones(15)]  # always high risk
    fail_steps = [10]

    tau, mean_lt = lead_time_at_recall(probs_by_traj, fail_steps, target_recall=0.9)
    assert mean_lt >= 0.0


def test_step_auroc_perfect_predictor():
    """Perfect predictor gives AUROC = 1.0."""
    probs = np.array([0.9, 0.1, 0.9, 0.1])
    labels = np.array([1, 0, 1, 0])
    auroc = step_auroc(probs, labels)
    assert auroc == 1.0


def test_step_auroc_single_class_returns_05():
    """AUROC is 0.5 when only one class present (undefined, return default)."""
    probs = np.array([0.7, 0.8, 0.6])
    labels = np.array([1, 1, 1])
    auroc = step_auroc(probs, labels)
    assert auroc == 0.5


def test_ece_perfect_calibration():
    """Perfectly calibrated predictor: prob 0.7 → 70% actual success."""
    n = 1000
    rng = np.random.default_rng(42)
    probs = rng.uniform(0, 1, n)
    labels = (rng.uniform(0, 1, n) < probs).astype(float)
    ece_val = ece(probs, labels, n_bins=10)
    # Well-calibrated should give ECE < 0.05
    assert ece_val < 0.1, f"ECE {ece_val} is too high for perfect calibration"


def test_success_rate_with_ci_all_success():
    """All successes → mean=1.0, CI=[1.0, 1.0]."""
    successes = np.ones(50)
    mean, ci_lo, ci_hi = success_rate_with_ci(successes, n_bootstrap=200)
    assert abs(mean - 1.0) < 1e-10
    assert abs(ci_lo - 1.0) < 1e-10
    assert abs(ci_hi - 1.0) < 1e-10


def test_bootstrap_mean_ci_range():
    """CI should contain the true mean for reasonable data."""
    rng = np.random.default_rng(0)
    data = rng.normal(0.5, 0.1, 200)
    mean, lo, hi = bootstrap_mean_ci(data, n_bootstrap=1000, confidence_level=0.95)
    assert lo <= mean <= hi
    assert abs(mean - 0.5) < 0.1
