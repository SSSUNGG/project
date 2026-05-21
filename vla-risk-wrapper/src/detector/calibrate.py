from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.detector.model import RiskDetector
from src.detector.train import _split_features


def temperature_scale(
    model: RiskDetector,
    val_loader: DataLoader,
    device: str = "cuda",
    max_iter: int = 50,
    lr: float = 0.01,
) -> float:
    """Fit temperature parameter T on a calibration set via NLL minimization.

    Args:
        model:      Trained RiskDetector (weights frozen). The temperature
                    buffer will be updated in-place.
        val_loader: DataLoader for the calibration subset.
        device:     'cuda' or 'cpu'.
        max_iter:   Maximum L-BFGS iterations.
        lr:         Learning rate for L-BFGS.

    Returns:
        The fitted temperature value (float).
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # Collect all logits and labels
    logits_list = []
    labels_list = []

    with torch.no_grad():
        for feats, labels in val_loader:
            feats = feats.to(device)
            labels = labels.to(device)
            vla_h, ah, prop = _split_features(feats, model.cfg)
            logits = model.forward(vla_h, ah, prop)
            logits_list.append(logits.detach())
            labels_list.append(labels)

    if not logits_list:
        return 1.0

    all_logits = torch.cat(logits_list)
    all_labels = torch.cat(labels_list)

    # Optimize temperature
    temperature = nn.Parameter(torch.ones(1, device=device))
    optimizer = torch.optim.LBFGS([temperature], lr=lr, max_iter=max_iter)
    criterion = nn.BCEWithLogitsLoss()

    def eval_step():
        optimizer.zero_grad()
        scaled_logits = all_logits / temperature.clamp(min=1e-3)
        loss = criterion(scaled_logits, all_labels)
        loss.backward()
        return loss

    optimizer.step(eval_step)

    T = float(temperature.clamp(min=0.1, max=10.0).item())

    # Update model temperature buffer
    with torch.no_grad():
        model.temperature.fill_(T)

    return T


def calibrate_and_save(
    ckpt_path: str,
    val_loader: DataLoader,
    device: str = "cuda",
) -> Path:
    """Load checkpoint, calibrate, and re-save with updated temperature.

    Args:
        ckpt_path:  Path to checkpoint saved by train_detector().
        val_loader: Calibration DataLoader (typically 50% of val set).
        device:     CUDA or CPU.

    Returns:
        Same ckpt_path after update.
    """
    from src.detector.features import DetectorConfig

    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt["cfg"]
    det_cfg = DetectorConfig.from_omegaconf(cfg)

    model_cfg = cfg.model
    model = RiskDetector(det_cfg, model_cfg)
    model.load_state_dict(ckpt["model_state"])

    T = temperature_scale(model, val_loader, device=device)
    print(f"Calibrated temperature: {T:.4f}")

    ckpt["temperature"] = T
    ckpt["model_state"] = model.state_dict()
    torch.save(ckpt, ckpt_path)

    return Path(ckpt_path)
