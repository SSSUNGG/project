from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class PolicyOutput:
    action: np.ndarray          # (7,) float32 — 7-DoF EE delta pose
    hidden_state: np.ndarray    # (D,) float32 — last-layer hidden, mean-pooled
    raw_logits: Optional[np.ndarray] = None
    latency_ms: float = 0.0


class BasePolicy(ABC):
    """Abstract base class for VLA policies."""

    @abstractmethod
    def reset(self, instruction: str) -> None:
        """Reset internal state for a new episode."""
        ...

    @abstractmethod
    def predict(self, obs: dict) -> PolicyOutput:
        """Run one inference step.

        Args:
            obs: Normalized observation dict from ManiSkill3.

        Returns:
            PolicyOutput with action, hidden_state, and latency_ms.
        """
        ...
