from __future__ import annotations

import numpy as np
import gymnasium as gym


def normalize_obs(obs: dict, env: gym.Env) -> dict:
    """Normalize a ManiSkill3 obs dict into a consistent format.

    RGB images are kept as uint8 (H, W, 3).
    Depth maps are kept as float32.
    Vector observations are cast to float32.
    """
    normalized = {}
    for key, val in obs.items():
        if isinstance(val, dict):
            normalized[key] = normalize_obs(val, env)
        elif isinstance(val, np.ndarray):
            if val.dtype == np.uint8:
                normalized[key] = val
            else:
                normalized[key] = val.astype(np.float32)
        else:
            normalized[key] = val
    return normalized


def extract_proprio(obs: dict, env: gym.Env) -> np.ndarray:
    """Extract proprioceptive state from obs.

    Returns shape (8,) float32:
      [0]   gripper_open  (1 scalar, 0=closed, 1=open)
      [1:7] ee_pose       (position xyz + quaternion xyzw, 7-dim → use pos xyz + euler or quat)
      [7]   contact_proxy (magnitude of contact force or binary grasp signal)

    Actual fields depend on ManiSkill3 obs structure. Falls back gracefully
    if expected keys are missing.
    """
    proprio = np.zeros(8, dtype=np.float32)

    # Try to extract from agent/extra sub-dicts (ManiSkill3 structure)
    agent_obs = obs.get("agent", obs)

    # Gripper openness
    if "qpos" in agent_obs:
        qpos = np.asarray(agent_obs["qpos"], dtype=np.float32).flatten()
        # Panda: last 2 joints are finger joints; mean > 0.02 means open
        if len(qpos) >= 2:
            finger_val = float(qpos[-1])
            proprio[0] = np.clip(finger_val / 0.04, 0.0, 1.0)

    # EE pose (6-dim: position xyz + euler rpy or 3+4 quaternion)
    extra = obs.get("extra", {})
    if "tcp_pose" in extra:
        tcp = np.asarray(extra["tcp_pose"], dtype=np.float32).flatten()
        # tcp_pose is typically (7,): pos(3) + quat(4)
        proprio[1:7] = tcp[:6]
    elif "ee_pos" in extra:
        ee_pos = np.asarray(extra["ee_pos"], dtype=np.float32).flatten()
        proprio[1:4] = ee_pos[:3]

    # Contact proxy: use is_grasped flag or contact force norm
    if "is_grasped" in extra:
        proprio[7] = float(extra["is_grasped"])
    elif "contact_forces" in extra:
        forces = np.asarray(extra["contact_forces"], dtype=np.float32).flatten()
        proprio[7] = float(np.linalg.norm(forces))

    return proprio
