from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

import numpy as np


@dataclass
class StepRecord:
    """Per-step record stored in parquet.

    vla_hidden is stored as float16 to save space (~8 KB per step).
    action_history pads with zeros for the first K steps.
    """
    episode_id: str
    task_id: str
    seed: int
    step: int
    vla_hidden: np.ndarray          # float16, (4096,)
    action: np.ndarray              # float32, (7,)
    action_history: np.ndarray      # float32, (K, 7)
    proprio: np.ndarray             # float32, (8,)
    rgb_thumb: Optional[np.ndarray] # uint8, (64, 64, 3) — optional
    info_is_grasped: bool
    info_is_success: bool
    reward: float
    done: bool
    truncated: bool

    def validate(self) -> None:
        assert self.vla_hidden.ndim == 1, "vla_hidden must be 1-D"
        assert self.action.shape == (7,), f"action shape {self.action.shape}"
        assert self.proprio.shape == (8,), f"proprio shape {self.proprio.shape}"
        assert self.action_history.ndim == 2 and self.action_history.shape[1] == 7

    def to_dict(self) -> dict:
        d = {
            "episode_id": self.episode_id,
            "task_id": self.task_id,
            "seed": np.int32(self.seed),
            "step": np.int32(self.step),
            "vla_hidden": self.vla_hidden.astype(np.float16),
            "action": self.action.astype(np.float32),
            "action_history": self.action_history.astype(np.float32),
            "proprio": self.proprio.astype(np.float32),
            "info_is_grasped": bool(self.info_is_grasped),
            "info_is_success": bool(self.info_is_success),
            "reward": np.float32(self.reward),
            "done": bool(self.done),
            "truncated": bool(self.truncated),
        }
        if self.rgb_thumb is not None:
            d["rgb_thumb"] = self.rgb_thumb.astype(np.uint8)
        return d


@dataclass
class EpisodeMeta:
    """Per-episode metadata stored in parquet."""
    episode_id: str
    task_id: str
    seed: int
    policy_id: str
    randomization_strength: str
    success: bool
    fail_step: int          # -1 if success
    episode_length: int
    wall_clock_seconds: float
    mean_latency_ms_per_step: float
    condition: str = "C1_baseline"
    tau: float = -1.0
    reasoner_calls: int = 0
    primitive_history: List[str] = field(default_factory=list)
    commit_sha: str = ""
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> dict:
        return {
            "episode_id": self.episode_id,
            "task_id": self.task_id,
            "seed": np.int32(self.seed),
            "policy_id": self.policy_id,
            "randomization_strength": self.randomization_strength,
            "success": bool(self.success),
            "fail_step": np.int32(self.fail_step),
            "episode_length": np.int32(self.episode_length),
            "wall_clock_seconds": np.float32(self.wall_clock_seconds),
            "mean_latency_ms_per_step": np.float32(self.mean_latency_ms_per_step),
            "condition": self.condition,
            "tau": np.float32(self.tau),
            "reasoner_calls": np.int32(self.reasoner_calls),
            "primitive_history": self.primitive_history,
            "commit_sha": self.commit_sha,
            "created_at": self.created_at,
        }


def new_episode_id() -> str:
    return str(uuid.uuid4())
