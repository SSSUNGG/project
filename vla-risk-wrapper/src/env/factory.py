from __future__ import annotations

import gymnasium as gym
from omegaconf import DictConfig

from src.env.randomization import apply_randomization


_TASK_ID_MAP = {
    "peg_insertion": "PegInsertionSide-v1",
    "assembling_kits": "AssemblingKits-v1",
    "pick_single_ycb": "PickSingleYCB-v1",
    "pick_cube": "PickCube-v1",
    "stack_cube": "StackCube-v1",
}


def make_env(cfg: DictConfig, seed: int) -> gym.Env:
    """Create a ManiSkill3 environment from Hydra config.

    Returns an unwrapped env object; caller must call env.reset(seed=...) explicitly.
    """
    env_id = cfg.env.id
    obs_mode = cfg.env.get("obs_mode", "rgb")
    control_mode = cfg.env.get("control_mode", "pd_ee_delta_pose")
    robot_uids = cfg.env.get("robot", "panda")
    max_episode_steps = cfg.env.get("max_episode_steps", 100)
    render_mode = cfg.env.get("render_mode", None)

    env = gym.make(
        env_id,
        obs_mode=obs_mode,
        control_mode=control_mode,
        robot_uids=robot_uids,
        max_episode_steps=max_episode_steps,
        render_mode=render_mode,
    )

    if cfg.env.get("randomization", None) is not None:
        apply_randomization(env, cfg.env.randomization)

    return env


def resolve_task_id(task_name: str) -> str:
    """Convert short task name to ManiSkill env ID."""
    if task_name in _TASK_ID_MAP:
        return _TASK_ID_MAP[task_name]
    return task_name
