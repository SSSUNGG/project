from __future__ import annotations

import numpy as np

from src.primitives.base import BasePrimitive, PrimitiveState

# EE delta pose action layout (pd_ee_delta_pose):
# [dx, dy, dz, drx, dry, drz, gripper]
# gripper: +1 = open, -1 = close


class ReGrasp(BasePrimitive):
    """Re-grasp primitive: open gripper → lift 5 cm → re-descend → close.

    Phase sequence:
      0: open gripper (1 step)
      1: lift up 5 cm (3 steps)
      2: descend back 5 cm (3 steps)
      3: close gripper (1 step)
    Total: 8 steps (within 6-8 spec range)
    """

    LIFT_CM = 0.05          # 5 cm
    STEPS_LIFT = 3
    STEPS_DESCEND = 3

    @property
    def max_steps(self) -> int:
        return 10

    def step(self, obs: dict, state: PrimitiveState):
        action = np.zeros(7, dtype=np.float32)
        state = PrimitiveState(
            phase=state.phase,
            step_in_phase=state.step_in_phase,
            abort_episode=state.abort_episode,
            extra=dict(state.extra),
        )

        if state.phase == 0:
            # Open gripper
            action[6] = 1.0
            state.phase = 1
            state.step_in_phase = 0
            return action, state, False

        elif state.phase == 1:
            # Lift up
            action[2] = self.LIFT_CM / self.STEPS_LIFT
            state.step_in_phase += 1
            if state.step_in_phase >= self.STEPS_LIFT:
                state.phase = 2
                state.step_in_phase = 0
            return action, state, False

        elif state.phase == 2:
            # Descend
            action[2] = -(self.LIFT_CM / self.STEPS_DESCEND)
            state.step_in_phase += 1
            if state.step_in_phase >= self.STEPS_DESCEND:
                state.phase = 3
                state.step_in_phase = 0
            return action, state, False

        else:
            # Close gripper
            action[6] = -1.0
            return action, state, True
