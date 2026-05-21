from __future__ import annotations

import numpy as np

from src.primitives.base import BasePrimitive, PrimitiveState


class Continue(BasePrimitive):
    """No-op primitive: terminates immediately, letting the VLA resume in 1 step."""

    @property
    def max_steps(self) -> int:
        return 1

    def step(self, obs: dict, state: PrimitiveState):
        action = np.zeros(7, dtype=np.float32)
        return action, state, True
