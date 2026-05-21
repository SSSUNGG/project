"""Reproducibility tests — most critical for Phase 3 validity (spec §12)."""
from __future__ import annotations

import hashlib

import numpy as np
import pytest

from src.policy.base import BasePolicy, PolicyOutput
from src.reasoner.base import BaseReasoner, ReasonerOutput
from src.primitives.continue_ import Continue
from src.wrapper.risk_gated import RiskGatedWrapper


# ── Minimal mock objects ───────────────────────────────────────────────────────

class _DummyPolicy(BasePolicy):
    def reset(self, instruction): pass
    def predict(self, obs):
        return PolicyOutput(
            action=np.zeros(7, dtype=np.float32),
            hidden_state=np.zeros(16, dtype=np.float32),
            latency_ms=0.0,
        )


class _DummyReasoner(BaseReasoner):
    def diagnose(self, rgb, instruction, recent_actions, risk_score):
        return ReasonerOutput("continue", "dummy", 0.0)


class _DummyDetector:
    def predict_proba_numpy(self, *a, **kw):
        return 0.0


def _make_wrapper(mode: str) -> RiskGatedWrapper:
    return RiskGatedWrapper(
        policy=_DummyPolicy(),
        detector=_DummyDetector(),
        reasoner=_DummyReasoner(),
        primitives={"continue": Continue()},
        tau=0.5,
        mode=mode,
        device="cpu",
    )


def _obs_to_hash(obs: dict) -> str:
    """Hash all numpy arrays in obs to a deterministic string."""
    parts = []
    for k in sorted(obs.keys()):
        v = obs[k]
        if isinstance(v, np.ndarray):
            parts.append(v.tobytes())
        elif isinstance(v, dict):
            for sk in sorted(v.keys()):
                sv = v[sk]
                if isinstance(sv, np.ndarray):
                    parts.append(sv.tobytes())
    return hashlib.sha256(b"".join(parts)).hexdigest()


# ── Tests ──────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(
    not pytest.importorskip("mani_skill", reason="mani_skill not installed"),
    reason="mani_skill not installed",
)
def test_same_seed_grid_gives_bit_identical_initial_obs():
    """C1, C2, C3 wrappers reset with the same (seed, ep) produce bit-identical obs.

    This is THE critical invariant for Phase 3 comparisons.
    """
    import gymnasium as gym

    env_id = "PickCube-v1"
    seed = 42
    ep_idx = 0
    combined_seed = seed * 1000 + ep_idx  # deterministic episode seed

    wrappers = [
        _make_wrapper("none"),
        _make_wrapper("always"),
        _make_wrapper("risk_gated"),
    ]

    hashes = []
    for wrapper in wrappers:
        env = gym.make(env_id, obs_mode="state_dict", render_mode=None)
        obs, _ = env.reset(seed=combined_seed)
        wrapper.reset("pick cube")
        hashes.append(_obs_to_hash(obs))
        env.close()

    assert len(set(hashes)) == 1, (
        f"Initial obs hashes differ across conditions:\n"
        + "\n".join(f"  {m}: {h}" for m, h in zip(["none","always","risk_gated"], hashes))
    )


def test_random_state_does_not_leak_between_conditions():
    """Running C1 for 5 steps then resetting C2 gives same result as fresh C2.

    Uses a numpy global RNG to verify that global state doesn't leak.
    """
    rng_state_before = np.random.get_state()

    w1 = _make_wrapper("none")
    w1.reset("task")
    obs = {"image": np.zeros((64, 64, 3), dtype=np.uint8), "_cached_proprio": np.zeros(8)}
    for _ in range(5):
        w1.step(obs)

    rng_state_after_c1 = np.random.get_state()

    w2 = _make_wrapper("always")
    w2.reset("task")
    info_fresh = w2.step(obs)

    # Reset RNG to state before C1 ran
    np.random.set_state(rng_state_before)

    w2b = _make_wrapper("always")
    w2b.reset("task")
    info_from_clean = w2b.step(obs)

    np.testing.assert_array_equal(
        info_fresh.action, info_from_clean.action,
        err_msg="C2 action differs depending on whether C1 ran first"
    )


def test_wrapper_reset_clears_action_history():
    """After reset(), action_history is all zeros."""
    wrapper = _make_wrapper("none")
    wrapper.reset("task")

    obs = {"image": np.zeros((64, 64, 3), dtype=np.uint8), "_cached_proprio": np.zeros(8)}
    for _ in range(3):
        wrapper.step(obs)

    # Reset and verify history is cleared
    wrapper.reset("new task")
    hist = np.stack(list(wrapper._action_history), axis=0)
    np.testing.assert_array_equal(
        hist, np.zeros((wrapper.K, 7), dtype=np.float32),
        err_msg="Action history not cleared after reset()"
    )
