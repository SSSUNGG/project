"""Checkpoint save/load helpers."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch


def save_checkpoint(state: dict, path: str, is_best: bool = False) -> None:
    """Save a checkpoint dict. Optionally copy to best.pt."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)
    if is_best:
        best_path = path.parent / f"best_{path.name}"
        torch.save(state, best_path)


def load_checkpoint(path: str, device: str = "cpu") -> dict:
    """Load a checkpoint from disk."""
    return torch.load(path, map_location=device)


def load_detector(ckpt_path: str, device: str = "cpu"):
    """Load a trained RiskDetector from a checkpoint file."""
    from src.detector.features import DetectorConfig
    from src.detector.model import RiskDetector

    ckpt = load_checkpoint(ckpt_path, device=device)
    cfg = ckpt["cfg"]
    det_cfg = DetectorConfig.from_omegaconf(cfg)
    model_cfg = cfg.model

    model = RiskDetector(det_cfg, model_cfg)
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device)
    model.eval()
    return model
