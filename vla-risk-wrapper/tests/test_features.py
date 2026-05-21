"""Tests for src/detector/features.py (spec §2 required)."""
from __future__ import annotations

import numpy as np
import pytest

from src.detector.features import DetectorConfig, build_feature_vector, raw_feature_dim


def _cfg(use_vla=True, use_action=True, use_proprio=True):
    return DetectorConfig(
        use_vla_hidden=use_vla,
        vla_hidden_dim=64,
        vla_proj_dim=32,
        use_action_history=use_action,
        action_history_K=4,
        action_dim=7,
        use_proprio=use_proprio,
        proprio_dim=8,
    )


def test_full_feature_dim():
    """Concatenated feature vector has correct dimension."""
    cfg = _cfg()
    expected = cfg.vla_hidden_dim + cfg.action_history_K * cfg.action_dim + cfg.proprio_dim
    assert raw_feature_dim(cfg) == expected


def test_build_feature_vector_shape():
    """build_feature_vector returns (D,) float32."""
    cfg = _cfg()
    vla_h = np.zeros(cfg.vla_hidden_dim, dtype=np.float16)
    ah = np.zeros((cfg.action_history_K, cfg.action_dim), dtype=np.float32)
    prop = np.zeros(cfg.proprio_dim, dtype=np.float32)

    feat = build_feature_vector(vla_h, ah, prop, cfg)
    assert feat.ndim == 1
    assert feat.dtype == np.float32
    assert feat.shape[0] == raw_feature_dim(cfg)


def test_disabled_streams_excluded():
    """Disabled streams reduce feature dimension correctly."""
    cfg_full = _cfg(use_vla=True, use_action=True, use_proprio=True)
    cfg_no_vla = _cfg(use_vla=False, use_action=True, use_proprio=True)
    cfg_action_only = _cfg(use_vla=False, use_action=True, use_proprio=False)

    assert raw_feature_dim(cfg_no_vla) == cfg_full.action_history_K * cfg_full.action_dim + cfg_full.proprio_dim
    assert raw_feature_dim(cfg_action_only) == cfg_full.action_history_K * cfg_full.action_dim


def test_none_inputs_give_zeros():
    """None inputs for disabled streams produce zero-padded features."""
    cfg = _cfg(use_vla=True, use_action=False, use_proprio=False)
    feat = build_feature_vector(None, None, None, cfg)
    assert feat.shape[0] == cfg.vla_hidden_dim
    np.testing.assert_array_equal(feat, np.zeros(cfg.vla_hidden_dim, dtype=np.float32))


def test_float16_vla_hidden_converted_to_float32():
    """float16 vla_hidden is upcast to float32 in output."""
    cfg = _cfg()
    vla_h = np.ones(cfg.vla_hidden_dim, dtype=np.float16)
    ah = np.zeros((cfg.action_history_K, cfg.action_dim), dtype=np.float32)
    prop = np.zeros(cfg.proprio_dim, dtype=np.float32)

    feat = build_feature_vector(vla_h, ah, prop, cfg)
    assert feat.dtype == np.float32
