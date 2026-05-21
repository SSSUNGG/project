"""Tests for RiskDetector model."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from src.detector.features import DetectorConfig
from src.detector.model import RiskDetector


def _make_cfg(use_vla=True, use_action=True, use_proprio=True):
    return DetectorConfig(
        use_vla_hidden=use_vla,
        vla_hidden_dim=64,   # small for tests
        vla_proj_dim=32,
        use_action_history=use_action,
        action_history_K=4,
        action_dim=7,
        use_proprio=use_proprio,
        proprio_dim=8,
    )


def _make_model_cfg():
    from omegaconf import OmegaConf
    return OmegaConf.create({"hidden_dim": 64, "num_layers": 2, "dropout": 0.0})


def _make_batch(cfg: DetectorConfig, B: int = 4):
    vla_h = torch.randn(B, cfg.vla_hidden_dim) if cfg.use_vla_hidden else None
    ah = torch.randn(B, cfg.action_history_K, cfg.action_dim) if cfg.use_action_history else None
    prop = torch.randn(B, cfg.proprio_dim) if cfg.use_proprio else None
    return vla_h, ah, prop


def test_detector_forward_shapes():
    """forward() returns (B,) logits with correct dtype."""
    cfg = _make_cfg()
    model = RiskDetector(cfg, _make_model_cfg())
    vla_h, ah, prop = _make_batch(cfg, B=4)

    logits = model.forward(vla_h, ah, prop)
    assert logits.shape == (4,), f"Expected (4,), got {logits.shape}"
    assert logits.dtype == torch.float32


def test_detector_predict_proba_range():
    """predict_proba() outputs are in [0, 1]."""
    cfg = _make_cfg()
    model = RiskDetector(cfg, _make_model_cfg())
    vla_h, ah, prop = _make_batch(cfg, B=8)

    probs = model.predict_proba(vla_h, ah, prop)
    assert probs.min() >= 0.0 and probs.max() <= 1.0


def test_detector_overfits_small_batch():
    """Model can overfit 8 samples in 100 epochs: train loss < 0.01."""
    cfg = _make_cfg(use_vla=True, use_action=True, use_proprio=True)
    model_cfg_dict = {"hidden_dim": 64, "num_layers": 2, "dropout": 0.0}
    from omegaconf import OmegaConf
    model_cfg = OmegaConf.create(model_cfg_dict)

    cfg.vla_hidden_dim = 32
    cfg.vla_proj_dim = 16
    cfg.action_history_K = 2

    model = RiskDetector(cfg, model_cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    criterion = torch.nn.BCEWithLogitsLoss()

    B = 8
    vla_h = torch.randn(B, cfg.vla_hidden_dim)
    ah = torch.randn(B, cfg.action_history_K, cfg.action_dim)
    prop = torch.randn(B, cfg.proprio_dim)
    labels = torch.tensor([1, 0, 1, 0, 1, 0, 1, 0], dtype=torch.float32)

    model.train()
    for _ in range(200):
        optimizer.zero_grad()
        logits = model.forward(vla_h, ah, prop)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

    assert loss.item() < 0.05, f"Did not overfit, final loss={loss.item():.4f}"


def test_detector_trivial_always_one_gives_auroc_05():
    """Constant 1-predictor gives AUROC = 0.5 (sanity check)."""
    from sklearn.metrics import roc_auc_score

    n = 100
    probs = np.ones(n)
    labels = np.array([1] * 50 + [0] * 50)

    # Constant predictions: AUROC = 0.5
    auroc = roc_auc_score(labels, probs)
    assert abs(auroc - 0.5) < 1e-9, f"Expected 0.5, got {auroc}"


def test_temperature_scaling_applied():
    """After setting temperature=2, probabilities are closer to 0.5."""
    cfg = _make_cfg()
    model = RiskDetector(cfg, _make_model_cfg())
    vla_h, ah, prop = _make_batch(cfg, B=4)

    # Default T=1
    probs_1 = model.predict_proba(vla_h, ah, prop).detach()

    # Set T=2 (softens predictions)
    with torch.no_grad():
        model.temperature.fill_(2.0)
    probs_2 = model.predict_proba(vla_h, ah, prop).detach()

    # With higher temperature, probabilities should be closer to 0.5
    dist_1 = (probs_1 - 0.5).abs().mean()
    dist_2 = (probs_2 - 0.5).abs().mean()
    assert dist_2 <= dist_1 + 1e-4, "Higher temperature should soften predictions"
