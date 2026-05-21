from __future__ import annotations

import numpy as np
from omegaconf import DictConfig


_STRENGTH_PRESETS = {
    "easy": dict(
        object_pose_xy_std=0.02,
        object_pose_yaw_std=0.1,
        lighting_jitter=0.1,
        camera_pose_jitter=0.01,
    ),
    "medium": dict(
        object_pose_xy_std=0.04,
        object_pose_yaw_std=0.3,
        lighting_jitter=0.3,
        camera_pose_jitter=0.02,
    ),
    "hard": dict(
        object_pose_xy_std=0.08,
        object_pose_yaw_std=0.6,
        lighting_jitter=0.5,
        camera_pose_jitter=0.04,
    ),
}


def apply_randomization(env, cfg_random: DictConfig) -> None:
    """Apply domain randomization to a ManiSkill3 environment.

    Object pose and lighting randomization require ManiSkill3 reconfigure hooks.
    Parts that are not supported by the current API are left as no-ops with TODO markers.
    """
    strength = cfg_random.get("strength", "medium")
    if strength in _STRENGTH_PRESETS:
        preset = _STRENGTH_PRESETS[strength]
        xy_std = cfg_random.get("object_pose_xy_std", preset["object_pose_xy_std"])
        yaw_std = cfg_random.get("object_pose_yaw_std", preset["object_pose_yaw_std"])
        lighting_jitter = cfg_random.get("lighting_jitter", preset["lighting_jitter"])
        cam_jitter = cfg_random.get("camera_pose_jitter", preset["camera_pose_jitter"])
    else:
        xy_std = cfg_random.get("object_pose_xy_std", 0.04)
        yaw_std = cfg_random.get("object_pose_yaw_std", 0.3)
        lighting_jitter = cfg_random.get("lighting_jitter", 0.3)
        cam_jitter = cfg_random.get("camera_pose_jitter", 0.02)

    # TODO: ManiSkill3 exposes randomization via env.unwrapped._setup_lighting() and
    # actor pose initialization. Hooking into these requires sub-classing the task or
    # using the reconfigure callback API. The parameters are stored here for when that
    # integration is done.
    _params = dict(
        object_pose_xy_std=xy_std,
        object_pose_yaw_std=yaw_std,
        lighting_jitter=lighting_jitter,
        camera_pose_jitter=cam_jitter,
        distractor_count=cfg_random.get("distractor_count", 0),
    )
    # No-op: ManiSkill3 domain randomization is configured at env construction time
    # via task-specific kwargs. See ManiSkill docs for the reconfigure API.
    _ = _params  # suppress unused warning until integration is complete
