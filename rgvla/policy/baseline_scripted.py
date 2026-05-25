from __future__ import annotations
import numpy as np
from typing import Optional


class ScriptedBaseline:
    """
    간단한 FSM 베이스라인 (frozen — 수정/재학습 금지).

    Phase: approach_above → descend → grasp → lift → move_to_goal → place → release → done

    이 정책은 완벽하지 않으며 randomization 수준에 따라
    PickCube 40~70%, PickSingleYCB 30~60% 성공률이 목표다.
    Wrapper의 효과를 측정하려면 baseline이 적당히 실패해야 한다.
    """

    # phase별 최대 step 수 (안전망)
    _MAX_STEPS = {
        "approach_above": 50,
        "descend":        35,
        "grasp":          15,
        "lift":           35,
        "move_to_goal":   50,
        "place":          25,
        "release":        10,
        "done":           999,
    }
    _PHASES = list(_MAX_STEPS.keys())

    def __init__(self, task: str, cfg: dict | None = None):
        self.task = task
        self.cfg = cfg or {}
        self._phase = "approach_above"
        self._phase_step = 0
        self._lift_start_z: float = 0.0
        self._gripper_open_val = 1.0   # 열기
        self._gripper_close_val = -1.0  # 닫기

    def reset(self, obs) -> None:
        self._phase = "approach_above"
        self._phase_step = 0
        self._lift_start_z = 0.0

    def act(self, obs) -> np.ndarray:
        """obs dict → action (7,)."""
        tcp_pos = self._get_tcp_pos(obs)
        obj_pos = self._get_obj_pos(obs)
        goal_pos = self._get_goal_pos(obs)

        action = np.zeros(7, dtype=np.float32)
        action[6] = self._gripper_open_val  # 기본: 열기

        if self._phase == "approach_above":
            target = obj_pos + np.array([0.0, 0.0, 0.15])
            delta = target - tcp_pos
            action[:3] = np.clip(delta * 4.0, -0.08, 0.08)
            action[6] = self._gripper_open_val
            if np.linalg.norm(delta) < 0.025 or self._phase_step >= self._MAX_STEPS["approach_above"]:
                self._next_phase()

        elif self._phase == "descend":
            # object 위 약 1.5cm 지점으로 내려감
            grasp_offset = self._grasp_height_offset()
            target = obj_pos + np.array([0.0, 0.0, grasp_offset])
            delta = target - tcp_pos
            action[:3] = np.clip(delta * 4.0, -0.05, 0.05)
            action[6] = self._gripper_open_val
            if np.linalg.norm(delta) < 0.020 or self._phase_step >= self._MAX_STEPS["descend"]:
                self._next_phase()

        elif self._phase == "grasp":
            action[:3] = np.zeros(3)
            action[6] = self._gripper_close_val
            if self._phase_step >= self._MAX_STEPS["grasp"]:
                self._lift_start_z = tcp_pos[2]
                self._next_phase()

        elif self._phase == "lift":
            target_z = self._lift_start_z + 0.18
            target = np.array([tcp_pos[0], tcp_pos[1], target_z])
            delta = target - tcp_pos
            action[:3] = np.clip(delta * 4.0, -0.05, 0.05)
            action[6] = self._gripper_close_val
            if tcp_pos[2] >= target_z - 0.02 or self._phase_step >= self._MAX_STEPS["lift"]:
                self._next_phase()

        elif self._phase == "move_to_goal":
            target = goal_pos + np.array([0.0, 0.0, 0.08])
            delta = target - tcp_pos
            action[:3] = np.clip(delta * 4.0, -0.08, 0.08)
            action[6] = self._gripper_close_val
            if np.linalg.norm(delta) < 0.025 or self._phase_step >= self._MAX_STEPS["move_to_goal"]:
                self._next_phase()

        elif self._phase == "place":
            target = goal_pos + np.array([0.0, 0.0, 0.01])
            delta = target - tcp_pos
            action[:3] = np.clip(delta * 4.0, -0.04, 0.04)
            action[6] = self._gripper_close_val
            if np.linalg.norm(delta) < 0.015 or self._phase_step >= self._MAX_STEPS["place"]:
                self._next_phase()

        elif self._phase == "release":
            action[:3] = np.zeros(3)
            action[6] = self._gripper_open_val
            if self._phase_step >= self._MAX_STEPS["release"]:
                self._next_phase()

        elif self._phase == "done":
            action[:3] = np.zeros(3)
            action[6] = self._gripper_open_val

        self._phase_step += 1
        return action

    def last_hidden(self) -> Optional[np.ndarray]:
        return None  # Medium-A

    # ─── 내부 유틸 ────────────────────────────────────────────
    def _next_phase(self):
        idx = self._PHASES.index(self._phase)
        if idx + 1 < len(self._PHASES):
            self._phase = self._PHASES[idx + 1]
        self._phase_step = 0

    def _grasp_height_offset(self) -> float:
        """task에 따른 파지 높이 조정."""
        if "YCB" in self.task:
            return 0.02   # YCB 물체는 크기 다양
        return 0.015      # 기본 (cube)

    @staticmethod
    def _to_np(x) -> np.ndarray:
        if hasattr(x, "numpy"):
            x = x.numpy()
        return np.asarray(x, dtype=np.float32).reshape(-1)

    def _get_tcp_pos(self, obs) -> np.ndarray:
        return self._to_np(obs["extra"]["tcp_pose"])[:3]

    def _get_obj_pos(self, obs) -> np.ndarray:
        extra = obs.get("extra", {})
        for key in ("obj_pose", "cube_pose", "object_pose"):
            if key in extra:
                return self._to_np(extra[key])[:3]
        raise KeyError(f"obj_pose 키를 찾지 못함. extra keys: {list(extra.keys())}")

    def _get_goal_pos(self, obs) -> np.ndarray:
        return self._to_np(obs["extra"]["goal_pos"])[:3]
