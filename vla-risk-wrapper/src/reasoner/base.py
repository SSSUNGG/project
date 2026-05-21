from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

import numpy as np


PrimitiveID = Literal[
    "re_grasp", "re_approach", "align_then_insert", "request_help", "continue"
]

VALID_PRIMITIVES: set[str] = {
    "re_grasp", "re_approach", "align_then_insert", "request_help", "continue"
}


@dataclass
class ReasonerOutput:
    primitive_id: str   # one of VALID_PRIMITIVES
    rationale: str
    latency_ms: float


class BaseReasoner(ABC):
    """Abstract base class for failure reasoners."""

    @abstractmethod
    def diagnose(
        self,
        rgb: np.ndarray,            # (H, W, 3) uint8
        instruction: str,
        recent_actions: np.ndarray, # (K, 7)
        risk_score: float,
    ) -> ReasonerOutput:
        """Diagnose the current state and select a recovery primitive.

        Args:
            rgb:            Current RGB observation.
            instruction:    Language task instruction.
            recent_actions: Last K actions taken by the policy.
            risk_score:     Risk probability from the detector.

        Returns:
            ReasonerOutput with primitive_id, rationale, and latency_ms.
        """
        ...
