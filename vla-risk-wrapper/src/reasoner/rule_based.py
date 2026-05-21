from __future__ import annotations

import time

import numpy as np
from omegaconf import DictConfig

from src.reasoner.base import BaseReasoner, ReasonerOutput, VALID_PRIMITIVES


class RuleBasedReasoner(BaseReasoner):
    """Deterministic rule-based reasoner for wrapper validation.

    Used in PR7–PR8 to verify wrapper end-to-end behavior before
    integrating Qwen2.5-VL in PR9. All thresholds are config-driven.

    Decision logic:
      1. risk_score > request_help_threshold  → request_help
      2. ee_z very low (possible stuck) AND not grasped → re_grasp
      3. recent_actions show near-zero progress → re_approach
      4. otherwise                              → continue
    """

    def __init__(self, cfg: DictConfig):
        self._request_help_thr = float(cfg.get("request_help_risk_threshold", 0.85))
        self._z_threshold = float(cfg.get("re_grasp_z_threshold", 0.05))
        self._progress_thr = float(cfg.get("re_approach_progress_threshold", 0.01))

    def diagnose(
        self,
        rgb: np.ndarray,
        instruction: str,
        recent_actions: np.ndarray,
        risk_score: float,
    ) -> ReasonerOutput:
        t0 = time.perf_counter()
        primitive_id = self._decide(rgb, recent_actions, risk_score)
        latency_ms = (time.perf_counter() - t0) * 1000

        assert primitive_id in VALID_PRIMITIVES, f"Invalid primitive: {primitive_id}"
        return ReasonerOutput(
            primitive_id=primitive_id,
            rationale=f"rule-based: selected {primitive_id} (risk={risk_score:.2f})",
            latency_ms=latency_ms,
        )

    def _decide(self, rgb, recent_actions: np.ndarray, risk_score: float) -> str:
        if risk_score >= self._request_help_thr:
            return "request_help"

        # Estimate progress as mean absolute delta in xyz
        if recent_actions is not None and len(recent_actions) >= 2:
            xyz_deltas = np.abs(recent_actions[:, :3])  # (K, 3)
            mean_progress = float(xyz_deltas.mean())
            if mean_progress < self._progress_thr:
                return "re_approach"

        # Heuristic: if risk is moderately high, try re-grasp
        if risk_score >= 0.6:
            return "re_grasp"

        return "continue"
