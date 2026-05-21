from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy as np
import torch

from src.policy.base import BasePolicy
from src.reasoner.base import BaseReasoner
from src.primitives.base import BasePrimitive, PrimitiveState


@dataclass
class WrapperStepInfo:
    action: np.ndarray
    risk_score: float
    reasoner_called: bool
    primitive_active: Optional[str]
    latency_breakdown_ms: dict[str, float]
    abort_episode: bool = False


class RiskGatedWrapper:
    """Wrapper that selectively calls a reasoner based on a learned risk detector.

    Modes:
        "none":        bypass detector and reasoner; execute VLA action directly.
        "always":      call reasoner every step (except when primitive is active).
        "risk_gated":  call reasoner only when detector risk score >= tau.

    Primitive lock:
        While a primitive is executing, new reasoner calls are suppressed.
        The primitive runs to completion before the wrapper returns control to the VLA.

    Latency:
        All GPU operations are wrapped with torch.cuda.synchronize() before
        time.perf_counter() to ensure accurate wall-clock measurements.
    """

    def __init__(
        self,
        policy: BasePolicy,
        detector,           # Optional[RiskDetector]
        reasoner: BaseReasoner,
        primitives: dict[str, BasePrimitive],
        tau: float,
        mode: Literal["none", "always", "risk_gated"],
        action_history_K: int = 8,
        device: str = "cuda",
    ):
        self.policy = policy
        self.detector = detector
        self.reasoner = reasoner
        self.primitives = primitives
        self.tau = tau
        self.mode = mode
        self.K = action_history_K
        self.device = device
        self._use_gpu = device == "cuda" and torch.cuda.is_available()

        self._instruction: str = ""
        self._action_history: deque = deque(
            [np.zeros(7, dtype=np.float32)] * self.K, maxlen=self.K
        )
        self._active_primitive: Optional[BasePrimitive] = None
        self._primitive_state: Optional[PrimitiveState] = None
        self._active_primitive_name: Optional[str] = None

        self.reasoner_call_count: int = 0
        self.primitive_history: list[str] = []

    def reset(self, instruction: str) -> None:
        self._instruction = instruction
        self._action_history = deque(
            [np.zeros(7, dtype=np.float32)] * self.K, maxlen=self.K
        )
        self._active_primitive = None
        self._primitive_state = None
        self._active_primitive_name = None
        self.reasoner_call_count = 0
        self.primitive_history = []
        self.policy.reset(instruction)

    def step(self, obs: dict) -> WrapperStepInfo:
        latency = {"policy": 0.0, "detector": 0.0, "reasoner": 0.0, "primitive": 0.0}

        # ── Phase 1: VLA policy forward ──────────────────────────────────────────
        self._gpu_sync()
        t0 = time.perf_counter()
        policy_out = self.policy.predict(obs)
        self._gpu_sync()
        t1 = time.perf_counter()
        latency["policy"] = (t1 - t0) * 1000

        vla_action = policy_out.action.astype(np.float32)
        vla_hidden = policy_out.hidden_state

        action_hist_arr = np.stack(list(self._action_history), axis=0)  # (K, 7)

        # ── Phase 2: Risk detection ───────────────────────────────────────────────
        risk_score = 0.0
        if self.mode != "none" and self.detector is not None:
            self._gpu_sync()
            t2s = time.perf_counter()
            risk_score = self._compute_risk(vla_hidden, action_hist_arr, obs)
            self._gpu_sync()
            t2e = time.perf_counter()
            latency["detector"] = (t2e - t2s) * 1000

        # ── Phase 3: Primitive lock ───────────────────────────────────────────────
        if self._active_primitive is not None:
            self._gpu_sync()
            t3s = time.perf_counter()
            action, self._primitive_state, done = self._active_primitive.step(
                obs, self._primitive_state
            )
            self._gpu_sync()
            t3e = time.perf_counter()
            latency["primitive"] = (t3e - t3s) * 1000

            abort = bool(self._primitive_state.abort_episode)
            if done:
                self._active_primitive = None
                self._primitive_state = None
                self._active_primitive_name = None

            self._action_history.append(action)
            return WrapperStepInfo(
                action=action,
                risk_score=risk_score,
                reasoner_called=False,
                primitive_active=self._active_primitive_name,
                latency_breakdown_ms=latency,
                abort_episode=abort,
            )

        # ── Phase 4: Reasoner decision ────────────────────────────────────────────
        reasoner_called = False
        reasoner_out = None
        t3s = time.perf_counter()

        should_call = False
        if self.mode == "always":
            should_call = True
        elif self.mode == "risk_gated" and risk_score >= self.tau:
            should_call = True

        if should_call:
            rgb = self._extract_rgb(obs)
            reasoner_out = self.reasoner.diagnose(
                rgb, self._instruction, action_hist_arr, risk_score
            )
            reasoner_called = True
            self.reasoner_call_count += 1

        self._gpu_sync()
        t3e = time.perf_counter()
        latency["reasoner"] = (t3e - t3s) * 1000

        # ── Phase 5: Primitive execution or VLA ─────────────────────────────────
        t4s = time.perf_counter()
        abort = False

        if reasoner_out is not None and reasoner_out.primitive_id != "continue":
            pid = reasoner_out.primitive_id
            primitive = self.primitives.get(pid)
            if primitive is not None:
                self._active_primitive = primitive
                self._primitive_state = primitive.initial_state()
                self._active_primitive_name = pid
                self.primitive_history.append(pid)

                # Execute first step of the new primitive immediately
                action, self._primitive_state, done = primitive.step(obs, self._primitive_state)
                if done:
                    abort = bool(self._primitive_state.abort_episode)
                    self._active_primitive = None
                    self._primitive_state = None
                    self._active_primitive_name = None
            else:
                action = vla_action
        else:
            action = vla_action

        t4e = time.perf_counter()
        latency["primitive"] += (t4e - t4s) * 1000

        self._action_history.append(action)

        return WrapperStepInfo(
            action=action,
            risk_score=risk_score,
            reasoner_called=reasoner_called,
            primitive_active=self._active_primitive_name,
            latency_breakdown_ms=latency,
            abort_episode=abort,
        )

    def _compute_risk(self, vla_hidden, action_hist, obs) -> float:
        """Run the detector and return a scalar risk probability."""
        from src.env.observation import extract_proprio

        try:
            proprio = obs.get("_cached_proprio", None)
            if proprio is None:
                # We need the env reference here — but wrapper doesn't hold it.
                # Callers should cache proprio in obs under '_cached_proprio'.
                proprio = np.zeros(8, dtype=np.float32)

            risk = self.detector.predict_proba_numpy(
                vla_hidden.astype(np.float32),
                action_hist.astype(np.float32),
                proprio.astype(np.float32),
                device=self.device,
            )
            return float(risk)
        except Exception:  # noqa: BLE001
            return 0.0

    def _gpu_sync(self):
        if self._use_gpu:
            torch.cuda.synchronize()

    @staticmethod
    def _extract_rgb(obs: dict) -> np.ndarray:
        for key in ("image", "rgb", "image_primary", "sensor_data"):
            if key in obs:
                val = obs[key]
                if isinstance(val, dict):
                    for sub in ("rgb", "image", "Color"):
                        if sub in val:
                            return np.asarray(val[sub], dtype=np.uint8)
                return np.asarray(val, dtype=np.uint8)
        return np.zeros((64, 64, 3), dtype=np.uint8)
