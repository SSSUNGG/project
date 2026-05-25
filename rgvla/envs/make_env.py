from __future__ import annotations
import numpy as np
import gymnasium as gym

try:
    import mani_skill.envs  # noqa  환경 등록
    _MANISKILL_AVAILABLE = True
except ImportError:
    _MANISKILL_AVAILABLE = False

SUPPORTED_TASKS = {
    "PickCube-v1",
    "StackCube-v1",
    "PickSingleYCB-v1",
    "PegInsertionSide-v1",
}

# Phase별 randomization 강도
# eval: 본실험 (낮음), collect: 데이터 수집 (실패율 30~50%)
RANDOMIZATION_PRESETS = {
    "eval":    {"obj_xy_std": 0.05, "obj_yaw_std": 0.5},
    "collect": {"obj_xy_std": 0.10, "obj_yaw_std": 1.0},
}


MAX_EPISODE_STEPS = {
    "PickCube-v1":        200,
    "StackCube-v1":       300,
    "PickSingleYCB-v1":   250,
    "PegInsertionSide-v1": 400,
}


def make_env(
    task: str,
    seed: int = 0,
    randomization: dict | None = None,
    obs_mode: str = "state",
    num_envs: int = 1,
) -> "RGEnvWrapper":
    assert _MANISKILL_AVAILABLE, "mani_skill 미설치. pip install mani_skill"
    assert task in SUPPORTED_TASKS, f"미지원 task: {task}"

    randomization = randomization or RANDOMIZATION_PRESETS["eval"]

    # max_episode_steps 명시: ManiSkill3 기본값(50)은 너무 짧음
    env = gym.make(
        task,
        obs_mode=obs_mode,
        control_mode="pd_ee_delta_pose",
        num_envs=num_envs,
        max_episode_steps=MAX_EPISODE_STEPS.get(task, 200),
    )
    return RGEnvWrapper(env, randomization)


class RGEnvWrapper(gym.Wrapper):
    """결정적 리셋 + 관측 추출 헬퍼."""

    def __init__(self, env: gym.Env, randomization: dict):
        super().__init__(env)
        self.randomization = randomization

    def reset_with(self, seed: int, episode: int):
        """(seed, episode) → bit-identical 초기 obs 보장."""
        eseed = seed * 100_000 + episode
        obs, info = self.env.reset(seed=eseed)
        return obs, info

    # ─── 관측 추출 헬퍼 ──────────────────────────────────
    @staticmethod
    def _to_np(x) -> np.ndarray:
        """tensor/배치 제거 → 1D float32."""
        if hasattr(x, "numpy"):
            x = x.numpy()
        return np.asarray(x, dtype=np.float32).reshape(-1)

    def get_tcp_pos(self, obs) -> np.ndarray:
        return self._to_np(obs["extra"]["tcp_pose"])[:3]

    def get_obj_pos(self, obs) -> np.ndarray:
        extra = obs.get("extra", {})
        for key in ("obj_pose", "cube_pose", "object_pose"):
            if key in extra:
                return self._to_np(extra[key])[:3]
        raise KeyError(f"obj_pose 키 없음. available: {list(extra.keys())}")

    def get_goal_pos(self, obs) -> np.ndarray:
        return self._to_np(obs["extra"]["goal_pos"])[:3]

    def get_proprio(self, obs) -> np.ndarray:
        parts = []
        agent = obs.get("agent", {})
        for key in ("qpos", "qvel"):
            if key in agent:
                parts.append(self._to_np(agent[key]))
        return np.concatenate(parts) if parts else np.zeros(18, np.float32)

    def is_success(self, info) -> bool:
        val = info.get("success", info.get("is_success", False))
        if hasattr(val, "item"):
            return bool(val.item())
        if hasattr(val, "__iter__"):
            import numpy as np
            return bool(np.any(val))
        return bool(val)

    def print_obs_structure(self, obs, prefix="obs"):
        """디버깅용."""
        if isinstance(obs, dict):
            for k, v in obs.items():
                self.print_obs_structure(v, f"{prefix}['{k}']")
        elif hasattr(obs, "shape"):
            print(f"  {prefix}: shape={tuple(obs.shape)}, dtype={obs.dtype}")
        else:
            print(f"  {prefix}: {type(obs)}")
