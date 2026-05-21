from __future__ import annotations

import numpy as np

from src.primitives.base import BasePrimitive, PrimitiveState


class AlignThenInsert(BasePrimitive):
    """Align yaw ±5° then slowly insert.

    Phase sequence:
      0: yaw correction (3 steps, ±5° distributed)
      1: slow insertion (5 steps, small +Y or +Z increments)
    Total: 8 steps (within 6-10 spec range)
    """

    YAW_DEG = 5.0
    STEPS_YAW = 3
    STEPS_INSERT = 5
    INSERT_STEP_M = 0.005  # 5 mm per step

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
            # Yaw correction: rotate around Z-axis
            yaw_rad = np.deg2rad(self.YAW_DEG) / self.STEPS_YAW
            action[5] = yaw_rad
            state.step_in_phase += 1
            if state.step_in_phase >= self.STEPS_YAW:
                state.phase = 1
                state.step_in_phase = 0
            return action, state, False

        elif state.phase == 1:
            # Slow insertion along +Y
            action[1] = self.INSERT_STEP_M
            state.step_in_phase += 1
            if state.step_in_phase >= self.STEPS_INSERT:
                return action, state, True
            return action, state, False

        return action, state, True
