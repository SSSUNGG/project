"""Evaluation metrics (spec §8)."""
from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.metrics import roc_auc_score


def step_auroc(probs: np.ndarray, labels: np.ndarray) -> float:
    """Area under the ROC curve at step level.

    Args:
        probs:  Predicted risk probabilities, shape (N,).
        labels: Binary ground-truth labels, shape (N,).

    Returns:
        AUROC in [0, 1]. Returns 0.5 if only one class present.
    """
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int32)

    if len(np.unique(labels)) < 2:
        return 0.5

    return float(roc_auc_score(labels, probs))


def lead_time_at_recall(
    probs_by_traj: list[np.ndarray],
    fail_steps: list[int],
    target_recall: float = 0.9,
) -> tuple[float, float]:
    """Compute the threshold at target recall and mean lead time to failure.

    Args:
        probs_by_traj: List of (T_i,) arrays of risk scores per trajectory.
                       Only failed trajectories (fail_steps[i] >= 0).
        fail_steps:    Failure step index for each trajectory (-1 if success).
        target_recall: Desired recall level.

    Returns:
        (tau_at_target_recall, mean_lead_time_steps)
        tau_at_target_recall: the risk threshold achieving target_recall.
        mean_lead_time_steps: mean number of steps before failure at which
                              the detector first fires (first step with prob >= tau).
    """
    # Collect all (prob, label) pairs for threshold search
    all_probs = []
    all_labels = []
    for probs, fs in zip(probs_by_traj, fail_steps):
        if fs < 0:
            continue  # skip success episodes
        n = len(probs)
        # Label: 1 in [max(0, fs-H), fs], but here we use a simple flag:
        # any step in failure window gets label 1
        for t, p in enumerate(probs):
            all_probs.append(float(p))
            all_labels.append(1 if t == fs else 0)

    if not all_probs:
        return 0.5, 0.0

    all_probs_arr = np.array(all_probs)
    all_labels_arr = np.array(all_labels)

    # Find threshold tau such that recall >= target_recall
    thresholds = np.unique(all_probs_arr)
    best_tau = float(thresholds[-1])
    for tau in sorted(thresholds):
        preds = (all_probs_arr >= tau).astype(int)
        if all_labels_arr.sum() == 0:
            break
        recall = float((preds & all_labels_arr).sum()) / float(all_labels_arr.sum())
        if recall >= target_recall:
            best_tau = tau
            break

    # Compute mean lead time at best_tau
    lead_times = []
    for probs, fs in zip(probs_by_traj, fail_steps):
        if fs < 0:
            continue
        for t, p in enumerate(probs):
            if p >= best_tau:
                lead_times.append(float(fs - t))
                break

    mean_lead = float(np.mean(lead_times)) if lead_times else 0.0
    return float(best_tau), mean_lead


def ece(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Expected Calibration Error.

    Args:
        probs:  Predicted probabilities, shape (N,).
        labels: Binary labels, shape (N,).
        n_bins: Number of equal-width bins in [0, 1].

    Returns:
        ECE in [0, 1].
    """
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece_val = 0.0

    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (probs >= lo) & (probs < hi)
        if mask.sum() == 0:
            continue
        bin_probs = probs[mask]
        bin_labels = labels[mask]
        bin_frac = mask.sum() / len(probs)
        ece_val += bin_frac * abs(bin_probs.mean() - bin_labels.mean())

    return float(ece_val)


def success_rate_with_ci(
    successes: np.ndarray,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
) -> tuple[float, float, float]:
    """Bootstrap confidence interval for success rate.

    Args:
        successes:     Binary array of episode outcomes, shape (N,).
        n_bootstrap:   Number of bootstrap resamples.
        confidence_level: CI width (e.g., 0.95 for 95% CI).

    Returns:
        (mean, ci_low, ci_high)
    """
    from src.eval.bootstrap import bootstrap_mean_ci

    return bootstrap_mean_ci(
        np.asarray(successes, dtype=np.float64),
        n_bootstrap=n_bootstrap,
        confidence_level=confidence_level,
    )


def paired_bootstrap_diff(
    cond_a: np.ndarray,
    cond_b: np.ndarray,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
) -> tuple[float, float, float, float]:
    """Paired bootstrap hypothesis test for difference in means.

    IMPORTANT: cond_a and cond_b must be indexed by the same (seed, ep) grid.

    Args:
        cond_a, cond_b: Binary arrays of episode outcomes (same length).
        n_bootstrap:    Number of resamples.
        confidence_level: CI width.

    Returns:
        (mean_diff, ci_low, ci_high, p_value)
        p_value is two-tailed (fraction of bootstrapped diffs with opposite sign).
    """
    cond_a = np.asarray(cond_a, dtype=np.float64)
    cond_b = np.asarray(cond_b, dtype=np.float64)
    assert len(cond_a) == len(cond_b), "cond_a and cond_b must have same length"

    diff = cond_a - cond_b
    observed_diff = float(diff.mean())

    rng = np.random.default_rng(42)
    boot_diffs = np.array([
        rng.choice(diff, size=len(diff), replace=True).mean()
        for _ in range(n_bootstrap)
    ])

    alpha = (1 - confidence_level) / 2
    ci_low = float(np.percentile(boot_diffs, 100 * alpha))
    ci_high = float(np.percentile(boot_diffs, 100 * (1 - alpha)))

    p_value = float(
        min(
            2 * np.mean(boot_diffs <= 0) if observed_diff > 0 else 2 * np.mean(boot_diffs >= 0),
            1.0,
        )
    )

    return observed_diff, ci_low, ci_high, p_value


def pareto_frontier(points: list[tuple[float, float]]) -> list[int]:
    """Return indices of Pareto-optimal points (minimize latency, maximize success).

    Args:
        points: List of (latency_ms, success_rate) tuples.

    Returns:
        Sorted list of indices on the Pareto frontier.
    """
    if not points:
        return []

    pts = np.array(points)  # (N, 2): col0=latency, col1=success
    n = len(pts)
    on_frontier = []

    for i in range(n):
        dominated = False
        for j in range(n):
            if i == j:
                continue
            # j dominates i if j has lower or equal latency AND higher or equal success
            if pts[j, 0] <= pts[i, 0] and pts[j, 1] >= pts[i, 1]:
                if pts[j, 0] < pts[i, 0] or pts[j, 1] > pts[i, 1]:
                    dominated = True
                    break
        if not dominated:
            on_frontier.append(i)

    return sorted(on_frontier)
