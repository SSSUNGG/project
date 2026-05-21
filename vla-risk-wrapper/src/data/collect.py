from __future__ import annotations

import time
import subprocess
from collections import deque
from typing import Optional

import numpy as np
from omegaconf import DictConfig

from src.data.schema import EpisodeMeta, StepRecord, new_episode_id
from src.env.observation import extract_proprio, normalize_obs
from src.policy.base import BasePolicy


def collect_episode(
    env,
    policy: BasePolicy,
    instruction: str,
    seed: int,
    cfg: DictConfig,
    episode_seed: Optional[int] = None,
) -> tuple[list[StepRecord], EpisodeMeta]:
    """Run one episode and collect per-step records.

    Horizon-based labeling is NOT done here — it's applied at detector training time.
    fail_step is computed as the terminal step index for failed episodes.

    Args:
        env:          ManiSkill3 gym.Env (already constructed).
        policy:       Loaded VLA policy.
        instruction:  Language instruction string for the task.
        seed:         Episode seed (used for env.reset).
        cfg:          Hydra config (expects cfg.env.max_episode_steps etc.).
        episode_seed: Override seed for env.reset (if None, uses seed).

    Returns:
        (list[StepRecord], EpisodeMeta)
    """
    K = cfg.detector.input.get("action_history_K", 8) if hasattr(cfg, "detector") else 8
    action_dim = 7
    episode_id = new_episode_id()
    task_id = cfg.env.get("id", "unknown").replace("-v1", "").lower()
    randomization_strength = cfg.env.randomization.get("strength", "medium")

    reset_seed = episode_seed if episode_seed is not None else seed
    obs, _ = env.reset(seed=reset_seed)
    obs = normalize_obs(obs, env)

    policy.reset(instruction)

    action_history = deque(
        [np.zeros(action_dim, dtype=np.float32)] * K, maxlen=K
    )

    step_records: list[StepRecord] = []
    total_latency_ms = 0.0
    wall_start = time.perf_counter()

    success = False
    fail_step = -1
    step_idx = 0

    max_steps = cfg.env.get("max_episode_steps", 100)

    for step_idx in range(max_steps):
        policy_out = policy.predict(obs)
        total_latency_ms += policy_out.latency_ms

        action = policy_out.action.astype(np.float32)
        hidden = policy_out.hidden_state.astype(np.float16)
        proprio = extract_proprio(obs, env)

        hist_arr = np.stack(list(action_history), axis=0)  # (K, 7)

        obs_next, reward, done, truncated, info = env.step(action)
        obs_next = normalize_obs(obs_next, env)

        is_success = bool(info.get("success", False))
        is_grasped = bool(info.get("is_grasped", False))

        record = StepRecord(
            episode_id=episode_id,
            task_id=task_id,
            seed=seed,
            step=step_idx,
            vla_hidden=hidden,
            action=action,
            action_history=hist_arr,
            proprio=proprio,
            rgb_thumb=None,
            info_is_grasped=is_grasped,
            info_is_success=is_success,
            reward=float(reward),
            done=bool(done),
            truncated=bool(truncated),
        )
        step_records.append(record)

        action_history.append(action)
        obs = obs_next

        if is_success:
            success = True
            break
        if done or truncated:
            break

    if not success:
        fail_step = step_idx

    wall_elapsed = time.perf_counter() - wall_start
    n_steps = len(step_records)
    mean_latency = total_latency_ms / max(n_steps, 1)

    commit_sha = _get_commit_sha()

    policy_id = type(policy).__name__.lower()

    meta = EpisodeMeta(
        episode_id=episode_id,
        task_id=task_id,
        seed=seed,
        policy_id=policy_id,
        randomization_strength=randomization_strength,
        success=success,
        fail_step=fail_step,
        episode_length=n_steps,
        wall_clock_seconds=float(wall_elapsed),
        mean_latency_ms_per_step=float(mean_latency),
        commit_sha=commit_sha,
    )

    return step_records, meta


def _get_commit_sha() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=2
        )
        return result.stdout.strip()
    except Exception:  # noqa: BLE001
        return ""
