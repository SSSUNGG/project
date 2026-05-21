from __future__ import annotations

import numpy as np

from src.primitives.base import BasePrimitive, PrimitiveState


class ReApproach(BasePrimitive):
    """Re-approach primitive: retreat 10 cm → start re-entry.

    Phase sequence:
      0: open gripper (1 step)
      1: retreat 10 cm backward (5 steps along -Y, configurable)
      2: re-entry approach forward (3 steps along +Y)
    Total: 9 steps (within 8-10 spec range)
    """

    RETREAT_CM = 0.10
    STEPS_RETREAT = 5
    STEPS_APPROACH = 3

    @property
    def max_steps(self) -> int:
        return 12

    def step(self, obs: dict, state: PrimitiveState):
        action = np.zeros(7, dtype=np.float32)
        state = PrimitiveState(
            phase=state.phase,
            step_in_phase=state.step_in_phase,
            abort_episode=state.abort_episode,
            extra=dict(state.extra),
        )

        if state.phase == 0:
            # Safety: open gripper
            action[6] = 1.0
            state.phase = 1
            state.step_in_phase = 0
            return action, state, False

        elif state.phase == 1:
            # Retreat along -Y (away from object/slot)
            action[1] = -(self.RETREAT_CM / self.STEPS_RETREAT)
            state.step_in_phase += 1
            if state.step_in_phase >= self.STEPS_RETREAT:
                state.phase = 2
                state.step_in_phase = 0
            return action, state, False

        elif state.phase == 2:
            # Re-approach: move back toward insertion point
            action[1] = (self.RETREAT_CM / 2.0) / self.STEPS_APPROACH
            state.step_in_phase += 1
            if state.step_in_phase >= self.STEPS_APPROACH:
                # Hand off back to VLA for fine approach
                return action, state, True
            return action, state, False

        return action, state, True
