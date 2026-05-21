"""Tests for src/policy/ module."""
from __future__ import annotations

import numpy as np
import pytest

from src.policy.base import PolicyOutput


def _make_dummy_obs():
    return {
        "image": np.zeros((128, 128, 3), dtype=np.uint8),
        "agent": {"qpos": np.zeros(9, dtype=np.float32)},
        "extra": {"tcp_pose": np.zeros(7, dtype=np.float32)},
    }


def test_openvla_raises_not_implemented():
    """OpenVLAPolicy must raise NotImplementedError until Phase 2 criteria pass."""
    from omegaconf import OmegaConf
    from src.policy.openvla_policy import OpenVLAPolicy

    cfg = OmegaConf.create({"hf_id": "openvla/openvla-7b"})
    with pytest.raises(NotImplementedError):
        OpenVLAPolicy(cfg, device="cpu")


@pytest.mark.slow
def test_octo_loads():
    """OctoPolicy loads without OOM (requires octo package)."""
    from omegaconf import OmegaConf
    from src.policy.octo_policy import OctoPolicy

    cfg = OmegaConf.create({"hf_id": "rail-berkeley/octo-base"})
    policy = OctoPolicy(cfg, device="cpu")
    assert policy is not None


def test_octo_predict_shapes_dummy():
    """OctoPolicy in dummy mode returns correct shapes."""
    from omegaconf import OmegaConf
    from src.policy.octo_policy import OctoPolicy

    cfg = OmegaConf.create({"hf_id": "rail-berkeley/octo-base"})
    policy = OctoPolicy.__new__(OctoPolicy)
    policy.cfg = cfg
    policy.device = "cpu"
    policy._use_gpu = False
    policy._loaded = False
    policy._model = None
    policy._instruction = "pick up the cube"

    obs = _make_dummy_obs()
    out = policy.predict(obs)

    assert isinstance(out, PolicyOutput)
    assert out.action.shape == (7,), f"action shape {out.action.shape}"
    assert out.action.dtype == np.float32
    assert out.hidden_state.ndim == 1
    assert out.hidden_state.dtype == np.float32
    assert out.latency_ms >= 0.0


def test_octo_hidden_state_not_all_zeros_after_real_run():
    """Hidden state should be non-zero when model is actually loaded."""
    pytest.skip("Requires octo package and model weights — run manually")
