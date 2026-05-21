from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from omegaconf import DictConfig


@dataclass
class DetectorConfig:
    use_vla_hidden: bool = True
    vla_hidden_dim: int = 4096
    vla_proj_dim: int = 256

    use_action_history: bool = True
    action_history_K: int = 8
    action_dim: int = 7

    use_proprio: bool = True
    proprio_dim: int = 8

    @property
    def in_dim(self) -> int:
        d = 0
        if self.use_vla_hidden:
            d += self.vla_proj_dim
        if self.use_action_history:
            d += self.action_history_K * self.action_dim
        if self.use_proprio:
            d += self.proprio_dim
        return d

    @classmethod
    def from_omegaconf(cls, cfg: DictConfig) -> "DetectorConfig":
        inp = cfg.input
        return cls(
            use_vla_hidden=inp.get("use_vla_hidden", True),
            vla_hidden_dim=inp.get("vla_hidden_dim", 4096),
            vla_proj_dim=inp.get("vla_proj_dim", 256),
            use_action_history=inp.get("use_action_history", True),
            action_history_K=inp.get("action_history_K", 8),
            action_dim=inp.get("action_dim", 7),
            use_proprio=inp.get("use_proprio", True),
            proprio_dim=inp.get("proprio_dim", 8),
        )


def build_feature_vector(
    vla_hidden: Optional[np.ndarray],
    action_history: Optional[np.ndarray],
    proprio: Optional[np.ndarray],
    cfg: DetectorConfig,
) -> np.ndarray:
    """Concatenate enabled feature streams into a single 1-D float32 vector.

    The projection of vla_hidden is NOT done here (that's the nn.Linear in RiskDetector).
    This function only assembles the raw inputs into the correct layout for the dataset.

    Note: vla_hidden is the raw (D,) float16 array from the step record.
          The model's vla_proj Linear will reduce it to proj_dim at train time.

    Args:
        vla_hidden:     (vla_hidden_dim,) float16 or float32, or None.
        action_history: (K, 7) float32, or None.
        proprio:        (8,) float32, or None.
        cfg:            DetectorConfig specifying which streams to include.

    Returns:
        Concatenated feature vector as float32 numpy array.
    """
    parts = []

    if cfg.use_vla_hidden:
        if vla_hidden is None:
            parts.append(np.zeros(cfg.vla_hidden_dim, dtype=np.float32))
        else:
            parts.append(np.asarray(vla_hidden, dtype=np.float32))

    if cfg.use_action_history:
        if action_history is None:
            parts.append(np.zeros(cfg.action_history_K * cfg.action_dim, dtype=np.float32))
        else:
            parts.append(np.asarray(action_history, dtype=np.float32).flatten())

    if cfg.use_proprio:
        if proprio is None:
            parts.append(np.zeros(cfg.proprio_dim, dtype=np.float32))
        else:
            parts.append(np.asarray(proprio, dtype=np.float32))

    if not parts:
        return np.zeros(1, dtype=np.float32)

    return np.concatenate(parts, axis=0).astype(np.float32)


def raw_feature_dim(cfg: DetectorConfig) -> int:
    """Total raw feature dimension (before vla_proj)."""
    d = 0
    if cfg.use_vla_hidden:
        d += cfg.vla_hidden_dim
    if cfg.use_action_history:
        d += cfg.action_history_K * cfg.action_dim
    if cfg.use_proprio:
        d += cfg.proprio_dim
    return d
