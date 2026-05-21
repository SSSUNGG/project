from __future__ import annotations

import numpy as np

from src.primitives.base import BasePrimitive, PrimitiveState


class RequestHelp(BasePrimitive):
    """Request human help: open gripper → move to safe home → set abort flag.

    Phase sequence:
      0: open gripper (1 step)
      1: lift to safety height (2 steps)
      2: set abort flag and terminate (1 step)
    Total: 4 steps (within 3-5 spec range)
    """

    SAFE_LIFT_M = 0.10
    STEPS_LIFT = 2

    @property
    def max_steps(self) -> int:
        return 6

    def step(self, obs: dict, state: PrimitiveState):
        action = np.zeros(7, dtype=np.float32)
        state = PrimitiveState(
            phase=state.phase,
            step_in_phase=state.step_in_phase,
            abort_episode=state.abort_episode,
            extra=dict(state.extra),
        )

        if state.phase == 0:
            action[6] = 1.0  # open gripper
            state.phase = 1
            state.step_in_phase = 0
            return action, state, False

        elif state.phase == 1:
            action[2] = self.SAFE_LIFT_M / self.STEPS_LIFT
            state.step_in_phase += 1
            if state.step_in_phase >= self.STEPS_LIFT:
                state.phase = 2
                state.step_in_phase = 0
            return action, state, False

        else:
            # Set abort flag
            state.abort_episode = True
            return action, state, True
