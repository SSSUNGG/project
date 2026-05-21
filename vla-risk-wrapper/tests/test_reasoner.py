"""Tests for src/reasoner/ module."""
from __future__ import annotations

import numpy as np
import pytest
from omegaconf import OmegaConf
from unittest.mock import MagicMock, patch

from src.reasoner.base import VALID_PRIMITIVES, ReasonerOutput
from src.reasoner.rule_based import RuleBasedReasoner


def _make_rule_cfg(**kwargs):
    defaults = {
        "request_help_risk_threshold": 0.85,
        "re_grasp_z_threshold": 0.05,
        "re_approach_progress_threshold": 0.01,
    }
    defaults.update(kwargs)
    return OmegaConf.create(defaults)


def _dummy_rgb():
    return np.zeros((64, 64, 3), dtype=np.uint8)


def _dummy_actions(K=8, near_zero=False):
    if near_zero:
        return np.zeros((K, 7), dtype=np.float32)
    return np.random.randn(K, 7).astype(np.float32) * 0.05


def test_rule_based_returns_valid_primitive_id():
    """RuleBasedReasoner always returns a valid primitive_id."""
    reasoner = RuleBasedReasoner(_make_rule_cfg())
    for risk in [0.1, 0.5, 0.7, 0.9]:
        out = reasoner.diagnose(_dummy_rgb(), "pick cube", _dummy_actions(), risk)
        assert out.primitive_id in VALID_PRIMITIVES, f"Invalid: {out.primitive_id}"
        assert out.latency_ms >= 0.0


def test_rule_based_high_risk_returns_request_help():
    """Risk above threshold → request_help."""
    reasoner = RuleBasedReasoner(_make_rule_cfg(request_help_risk_threshold=0.85))
    out = reasoner.diagnose(_dummy_rgb(), "insert peg", _dummy_actions(), risk_score=0.95)
    assert out.primitive_id == "request_help"


def test_rule_based_near_zero_actions_returns_re_approach():
    """Near-zero action history → re_approach."""
    reasoner = RuleBasedReasoner(_make_rule_cfg(
        request_help_risk_threshold=0.85,
        re_approach_progress_threshold=0.01,
    ))
    out = reasoner.diagnose(
        _dummy_rgb(), "insert peg",
        _dummy_actions(near_zero=True),
        risk_score=0.3,
    )
    assert out.primitive_id == "re_approach"


def test_rule_based_output_is_reasoner_output():
    """Output is a ReasonerOutput dataclass."""
    reasoner = RuleBasedReasoner(_make_rule_cfg())
    out = reasoner.diagnose(_dummy_rgb(), "task", _dummy_actions(), 0.5)
    assert isinstance(out, ReasonerOutput)


def test_vlm_parse_fail_falls_back_to_continue():
    """VLMReasoner falls back to 'continue' on JSON parse failure."""
    from src.reasoner.vlm_reasoner import VLMReasoner

    cfg = OmegaConf.create({"hf_id": "mock/model", "max_new_tokens": 64})

    # Create without loading real model
    with patch.object(VLMReasoner, "_load_model", lambda self: None):
        reasoner = VLMReasoner(cfg, device="cpu")
        reasoner._loaded = True
        reasoner._model = MagicMock()
        reasoner._processor = MagicMock()

        # Simulate VLM returning invalid JSON
        reasoner._call_vlm = lambda *a, **kw: ("continue", "parse_fail")

        out = reasoner.diagnose(_dummy_rgb(), "task", _dummy_actions(), 0.7)
        assert out.primitive_id == "continue"


def test_vlm_output_in_allowed_set():
    """VLMReasoner._parse_output extracts primitive_id from valid JSON."""
    from src.reasoner.vlm_reasoner import VLMReasoner

    cfg = OmegaConf.create({"hf_id": "mock/model", "max_new_tokens": 64})

    with patch.object(VLMReasoner, "_load_model", lambda self: None):
        reasoner = VLMReasoner(cfg, device="cpu")
        reasoner._loaded = False

    valid_json = '{"primitive": "re_grasp", "rationale": "gripper slipped"}'
    pid, rationale = reasoner._parse_output(valid_json)
    assert pid in VALID_PRIMITIVES
    assert pid == "re_grasp"

    invalid_json = "I think re_grasp is best"
    pid2, _ = reasoner._parse_output(invalid_json)
    assert pid2 == "continue"
