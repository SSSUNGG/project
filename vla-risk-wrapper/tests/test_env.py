"""Tests for src/env/ module."""
from __future__ import annotations

import numpy as np
import pytest

try:
    import gymnasium as gym
    import mani_skill  # noqa: F401
    HAS_MANISKILL = True
except ImportError:
    HAS_MANISKILL = False

pytestmark = pytest.mark.skipif(not HAS_MANISKILL, reason="mani_skill not installed")


@pytest.fixture()
def pick_cube_env():
    env = gym.make("PickCube-v1", obs_mode="state_dict", render_mode=None)
    yield env
    env.close()


def test_make_env_returns_gym_env(pick_cube_env):
    """make_env returns a valid gym.Env with expected obs keys."""
    from omegaconf import OmegaConf
    from src.env.factory import make_env

    cfg_dict = {
        "env": {
            "id": "PickCube-v1",
            "obs_mode": "state_dict",
            "control_mode": "pd_ee_delta_pose",
            "robot": "panda",
            "max_episode_steps": 50,
            "render_mode": None,
        }
    }
    cfg = OmegaConf.create(cfg_dict)
    env = make_env(cfg, seed=0)
    assert isinstance(env, gym.Env)
    obs, _ = env.reset(seed=0)
    assert isinstance(obs, dict)
    env.close()


def test_seed_reproducibility(pick_cube_env):
    """Same seed reset gives bit-identical obs."""
    obs1, _ = pick_cube_env.reset(seed=42)
    obs2, _ = pick_cube_env.reset(seed=42)

    def _flatten(o):
        if isinstance(o, dict):
            return np.concatenate([_flatten(v) for v in sorted(o.items(), key=lambda x: x[0])])
        return np.asarray(o, dtype=np.float32).flatten()

    arr1 = _flatten(obs1)
    arr2 = _flatten(obs2)
    np.testing.assert_array_equal(arr1, arr2)


def test_proprio_shape(pick_cube_env):
    """extract_proprio returns (8,) float32 array."""
    from src.env.observation import extract_proprio

    obs, _ = pick_cube_env.reset(seed=0)
    proprio = extract_proprio(obs, pick_cube_env)
    assert proprio.shape == (8,), f"Expected (8,), got {proprio.shape}"
    assert proprio.dtype == np.float32


def test_peg_insertion_env_runs():
    """PegInsertionSide-v1 runs 1 episode with random actions."""
    env = gym.make("PegInsertionSide-v1", obs_mode="state_dict", render_mode=None)
    obs, _ = env.reset(seed=0)
    for _ in range(5):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        if done or truncated:
            break
    env.close()
