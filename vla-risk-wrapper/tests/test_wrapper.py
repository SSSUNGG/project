"""Tests for RiskGatedWrapper (spec §12)."""
from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from src.policy.base import BasePolicy, PolicyOutput
from src.reasoner.base import BaseReasoner, ReasonerOutput
from src.primitives.base import BasePrimitive, PrimitiveState
from src.wrapper.risk_gated import RiskGatedWrapper, WrapperStepInfo


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_policy(hidden_dim=16):
    class _Policy(BasePolicy):
        def reset(self, instruction): pass
        def predict(self, obs):
            return PolicyOutput(
                action=np.zeros(7, dtype=np.float32),
                hidden_state=np.zeros(hidden_dim, dtype=np.float32),
                latency_ms=1.0,
            )
    return _Policy()


def _make_reasoner(primitive_id="continue"):
    class _Reasoner(BaseReasoner):
        def diagnose(self, rgb, instruction, recent_actions, risk_score):
            return ReasonerOutput(primitive_id=primitive_id, rationale="test", latency_ms=0.5)
    return _Reasoner()


def _make_continue_primitive():
    from src.primitives.continue_ import Continue
    return Continue()


def _make_detector(risk_value=0.0):
    """Mock detector that returns a fixed risk value."""
    det = MagicMock()
    det.predict_proba_numpy.return_value = float(risk_value)
    return det


def _make_wrapper(mode="none", tau=0.5, reasoner_pid="continue", risk_value=0.0):
    policy = _make_policy()
    detector = _make_detector(risk_value)
    reasoner = _make_reasoner(reasoner_pid)
    primitives = {
        "re_grasp": _make_continue_primitive(),
        "re_approach": _make_continue_primitive(),
        "align_then_insert": _make_continue_primitive(),
        "request_help": _make_continue_primitive(),
        "continue": _make_continue_primitive(),
    }
    wrapper = RiskGatedWrapper(
        policy=policy,
        detector=detector,
        reasoner=reasoner,
        primitives=primitives,
        tau=tau,
        mode=mode,
        action_history_K=8,
        device="cpu",
    )
    return wrapper


def _dummy_obs():
    return {
        "image": np.zeros((64, 64, 3), dtype=np.uint8),
        "_cached_proprio": np.zeros(8, dtype=np.float32),
    }


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_wrapper_none_mode_never_calls_reasoner():
    """mode='none' must never call the reasoner, regardless of risk."""
    reasoner = _make_reasoner("re_grasp")
    call_log = []

    class _TrackingReasoner(BaseReasoner):
        def diagnose(self, *a, **kw):
            call_log.append(1)
            return ReasonerOutput("continue", "tracking", 0.0)

    wrapper = RiskGatedWrapper(
        policy=_make_policy(),
        detector=_make_detector(risk_value=0.99),  # very high risk
        reasoner=_TrackingReasoner(),
        primitives={"continue": _make_continue_primitive()},
        tau=0.5,
        mode="none",
        device="cpu",
    )
    wrapper.reset("task")
    obs = _dummy_obs()
    for _ in range(5):
        info = wrapper.step(obs)
        assert not info.reasoner_called

    assert len(call_log) == 0, f"Reasoner was called {len(call_log)} times in mode='none'"


def test_wrapper_always_mode_calls_every_step_except_primitive():
    """mode='always' calls reasoner each step when no primitive is active."""
    call_log = []

    class _TrackingReasoner(BaseReasoner):
        def diagnose(self, *a, **kw):
            call_log.append(1)
            return ReasonerOutput("continue", "tracking", 0.0)

    wrapper = RiskGatedWrapper(
        policy=_make_policy(),
        detector=None,
        reasoner=_TrackingReasoner(),
        primitives={"continue": _make_continue_primitive()},
        tau=0.5,
        mode="always",
        device="cpu",
    )
    wrapper.reset("task")
    obs = _dummy_obs()
    N = 5
    for _ in range(N):
        info = wrapper.step(obs)

    assert len(call_log) == N, f"Expected {N} calls, got {len(call_log)}"


def test_wrapper_risk_gated_respects_tau():
    """mode='risk_gated': reasoner called only when risk >= tau."""
    call_log = []

    class _TrackingReasoner(BaseReasoner):
        def diagnose(self, *a, **kw):
            call_log.append(1)
            return ReasonerOutput("continue", "x", 0.0)

    # Risk below tau
    wrapper_low = RiskGatedWrapper(
        policy=_make_policy(),
        detector=_make_detector(risk_value=0.3),
        reasoner=_TrackingReasoner(),
        primitives={"continue": _make_continue_primitive()},
        tau=0.5,
        mode="risk_gated",
        device="cpu",
    )
    wrapper_low.reset("task")
    for _ in range(5):
        wrapper_low.step(_dummy_obs())
    calls_low = len(call_log)

    call_log.clear()

    # Risk above tau
    wrapper_high = RiskGatedWrapper(
        policy=_make_policy(),
        detector=_make_detector(risk_value=0.7),
        reasoner=_TrackingReasoner(),
        primitives={"continue": _make_continue_primitive()},
        tau=0.5,
        mode="risk_gated",
        device="cpu",
    )
    wrapper_high.reset("task")
    for _ in range(5):
        wrapper_high.step(_dummy_obs())
    calls_high = len(call_log)

    assert calls_low == 0, f"Expected 0 calls below tau, got {calls_low}"
    assert calls_high == 5, f"Expected 5 calls above tau, got {calls_high}"


def test_primitive_lock_blocks_new_reasoner_call():
    """When a multi-step primitive is running, no new reasoner calls are made."""
    from src.primitives.re_grasp import ReGrasp

    call_log = []

    class _TrackingReasoner(BaseReasoner):
        def diagnose(self, *a, **kw):
            call_log.append(1)
            return ReasonerOutput("continue", "x", 0.0)

    # Use a real multi-step primitive (ReGrasp takes ~8 steps)
    wrapper = RiskGatedWrapper(
        policy=_make_policy(),
        detector=_make_detector(risk_value=0.99),
        reasoner=_TrackingReasoner(),
        primitives={
            "re_grasp": ReGrasp(),
            "continue": _make_continue_primitive(),
        },
        tau=0.5,
        mode="risk_gated",
        device="cpu",
    )
    wrapper.reset("task")

    # First step: risk >= tau → reasoner called → re_grasp activated
    obs = _dummy_obs()
    wrapper._make_reasoner_return_regrasp = True

    class _ReGraspReasoner(BaseReasoner):
        def diagnose(self, *a, **kw):
            call_log.append(1)
            return ReasonerOutput("re_grasp", "x", 0.0)

    wrapper.reasoner = _ReGraspReasoner()

    info = wrapper.step(obs)
    n_calls_after_first = len(call_log)

    # Subsequent steps while primitive is active should NOT call reasoner
    call_log.clear()
    for _ in range(5):
        if wrapper._active_primitive is not None:
            wrapper.step(obs)

    assert len(call_log) == 0, (
        f"Reasoner was called {len(call_log)} times while primitive was active"
    )


def test_latency_breakdown_has_all_four_keys():
    """WrapperStepInfo.latency_breakdown_ms contains all four required keys."""
    wrapper = _make_wrapper(mode="risk_gated", tau=0.5, risk_value=0.0)
    wrapper.reset("task")
    info = wrapper.step(_dummy_obs())

    required_keys = {"policy", "detector", "reasoner", "primitive"}
    assert required_keys.issubset(info.latency_breakdown_ms.keys()), (
        f"Missing keys: {required_keys - set(info.latency_breakdown_ms)}"
    )
    for k, v in info.latency_breakdown_ms.items():
        assert v >= 0.0, f"Negative latency for {k}: {v}"


def test_wrapper_info_has_correct_types():
    """WrapperStepInfo fields have correct types."""
    wrapper = _make_wrapper(mode="none")
    wrapper.reset("pick cube")
    info = wrapper.step(_dummy_obs())

    assert isinstance(info, WrapperStepInfo)
    assert info.action.shape == (7,)
    assert info.action.dtype == np.float32
    assert isinstance(info.risk_score, float)
    assert isinstance(info.reasoner_called, bool)
