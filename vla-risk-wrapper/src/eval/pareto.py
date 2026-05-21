"""Pareto frontier utilities for success–latency trade-off analysis."""
from __future__ import annotations

import numpy as np

from src.eval.metrics import pareto_frontier


def aggregate_for_pareto(
    results_by_condition: dict[str, dict],
) -> dict[str, tuple[float, float, float, float]]:
    """Extract (mean_latency_ms, mean_success_rate, ci_low, ci_high) per condition.

    Args:
        results_by_condition: dict of condition_id → EvalResult dict containing
            'successes' (list/array of bool) and 'latencies_ms' (list/array of float).

    Returns:
        dict of condition_id → (mean_latency_ms, mean_success_rate, ci_low, ci_high).
    """
    from src.eval.bootstrap import bootstrap_mean_ci

    aggregated = {}
    for cond_id, result in results_by_condition.items():
        successes = np.asarray(result.get("successes", []), dtype=np.float64)
        latencies = np.asarray(result.get("latencies_ms", []), dtype=np.float64)

        mean_lat = float(latencies.mean()) if len(latencies) > 0 else 0.0
        mean_sr, ci_low, ci_high = bootstrap_mean_ci(successes) if len(successes) > 0 else (0.0, 0.0, 0.0)

        aggregated[cond_id] = (mean_lat, mean_sr, ci_low, ci_high)

    return aggregated


def mark_frontier_points(
    aggregated: dict[str, tuple[float, float, float, float]],
) -> dict[str, bool]:
    """Mark which conditions are on the Pareto frontier.

    Returns dict of condition_id → is_on_frontier.
    """
    cond_ids = list(aggregated.keys())
    points = [(v[0], v[1]) for v in aggregated.values()]  # (latency, success)

    frontier_indices = pareto_frontier(points)
    frontier_set = set(frontier_indices)

    return {cid: (i in frontier_set) for i, cid in enumerate(cond_ids)}
