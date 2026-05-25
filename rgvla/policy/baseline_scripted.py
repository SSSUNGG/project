from __future__ import annotations
import numpy as np
from typing import Optional

# pd_ee_delta_pose 스케일: action=1.0 → ~10cm 이동
# (ManiSkill3 Panda pd_ee_delta_pose 컨트롤러 기본값)
_POS_SCALE = 0.1   # 1.0 action = 10cm

# 호버 / 파지 높이 상수
_HOVER    = 0.15   # 물체 위 15cm에서 접근
_GRASP_Z  = 0.015  # 물체 중심 위 1.5cm에서 닫기 (cube 반지름 ≈ 2cm)
_LIFT_H   = 0.20   # 파지 후 20cm 들어올림
_ABOVE_GOAL = 0.08 # 목표 위 8cm → 내려놓기


class ScriptedBaseline:
    """
    FSM 베이스라인 (frozen — 수정/재학습 금지).

    Phase: approach_above → descend → grasp → lift → move_to_goal → place → release → done

    obs['extra']['tcp_to_obj_pos'] = obj_pos - tcp_pos  (relative, 직접 사용)
    obs['extra']['obj_to_goal_pos'] = goal_pos - obj_pos (relative, 직접 사용)

    action[:3] = np.clip(delta / _POS_SCALE, -1, 1)
      → delta < 10cm 면 1 step에 도달, 더 멀면 최고속도로 접근
    """

    # phase별 최대 step 수 (안전망: 이 안에 목표 못 도달하면 강제 전환)
    _MAX = {
        "approach_above": 15,
        "descend":        12,
        "grasp":          12,
        "lift":           12,
        "move_to_goal":   20,
        "place":          10,
        "release":         8,
        "done":          999,
    }
    _PHASES = list(_MAX.keys())

    def __init__(self, task: str, cfg: dict | None = None):
        self.task = task
        self._phase = "approach_above"
        self._phase_step = 0
        self._lift_start_z = 0.0

    def reset(self, obs) -> None:
        self._phase = "approach_above"
        self._phase_step = 0
        self._lift_start_z = 0.0

    def act(self, obs) -> np.ndarray:
        extra = obs["extra"]

        # obs 구조 확인 결과 사용:
        #   tcp_to_obj_pos = obj_pos - tcp_pos
        #   obj_to_goal_pos = goal_pos - obj_pos
        tcp_to_obj  = self._np(extra["tcp_to_obj_pos"])   # (3,)
        obj_to_goal = self._np(extra["obj_to_goal_pos"])  # (3,)
        tcp_pos     = self._np(extra["tcp_pose"])[:3]     # (3,)
        is_grasped  = bool(self._np(extra.get("is_grasped", [0]))[0])

        action = np.zeros(7, dtype=np.float32)
        action[6] = 1.0  # 기본: gripper open

        grasp_z = _GRASP_Z + (0.01 if "YCB" in self.task else 0.0)

        if self._phase == "approach_above":
            # TCP → (obj_x, obj_y, obj_z + HOVER)
            # delta_to_target = tcp_to_obj + [0, 0, HOVER]
            delta = tcp_to_obj + np.array([0., 0., _HOVER])
            action[:3] = np.clip(delta / _POS_SCALE, -1., 1.)
            action[6] = 1.0
            if np.linalg.norm(delta) < 0.025 or self._phase_step >= self._MAX["approach_above"]:
                self._next()

        elif self._phase == "descend":
            # TCP → (obj_x, obj_y, obj_z + grasp_z)
            delta = tcp_to_obj + np.array([0., 0., grasp_z])
            action[:3] = np.clip(delta / _POS_SCALE, -1., 1.)
            action[6] = 1.0
            if np.linalg.norm(delta) < 0.018 or self._phase_step >= self._MAX["descend"]:
                self._lift_start_z = tcp_pos[2]
                self._next()

        elif self._phase == "grasp":
            action[:3] = np.zeros(3)
            action[6] = -1.0  # close
            if is_grasped or self._phase_step >= self._MAX["grasp"]:
                self._lift_start_z = tcp_pos[2]
                self._next()

        elif self._phase == "lift":
            # 수직으로 들어올리기
            target_z = self._lift_start_z + _LIFT_H
            dz = target_z - tcp_pos[2]
            action[:3] = np.clip(np.array([0., 0., dz]) / _POS_SCALE, -1., 1.)
            action[6] = -1.0
            if dz < 0.025 or self._phase_step >= self._MAX["lift"]:
                self._next()

        elif self._phase == "move_to_goal":
            # TCP → (goal_x, goal_y, goal_z + ABOVE_GOAL)
            # tcp_to_obj + obj_to_goal = tcp_to_goal
            tcp_to_goal = tcp_to_obj + obj_to_goal
            delta = tcp_to_goal + np.array([0., 0., _ABOVE_GOAL])
            action[:3] = np.clip(delta / _POS_SCALE, -1., 1.)
            action[6] = -1.0
            if np.linalg.norm(delta) < 0.030 or self._phase_step >= self._MAX["move_to_goal"]:
                self._next()

        elif self._phase == "place":
            # 천천히 내려놓기
            tcp_to_goal = tcp_to_obj + obj_to_goal
            delta = tcp_to_goal + np.array([0., 0., 0.01])
            action[:3] = np.clip(delta / _POS_SCALE, -1., 1.)
            action[6] = -1.0
            if np.linalg.norm(delta) < 0.015 or self._phase_step >= self._MAX["place"]:
                self._next()

        elif self._phase == "release":
            action[:3] = np.zeros(3)
            action[6] = 1.0  # open
            if self._phase_step >= self._MAX["release"]:
                self._next()

        # done: 아무것도 안 함

        self._phase_step += 1
        return action

    def last_hidden(self) -> Optional[np.ndarray]:
        return None  # Medium-A

    def _next(self):
        idx = self._PHASES.index(self._phase)
        if idx + 1 < len(self._PHASES):
            self._phase = self._PHASES[idx + 1]
        self._phase_step = 0

    @staticmethod
    def _np(x) -> np.ndarray:
        if hasattr(x, "numpy"):
            x = x.numpy()
        return np.asarray(x, dtype=np.float32).reshape(-1)
