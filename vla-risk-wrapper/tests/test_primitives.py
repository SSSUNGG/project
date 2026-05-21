"""Tests for src/primitives/ module."""
from __future__ import annotations

import numpy as np
import pytest

from src.primitives.base import PrimitiveState


def _dummy_obs():
    return {}


def _run_primitive(primitive, max_steps_override=None):
    """Run a primitive to completion, return all actions and final state."""
    state = primitive.initial_state()
    actions = []
    max_s = max_steps_override or primitive.max_steps
    for _ in range(max_s):
        action, state, done = primitive.step(_dummy_obs(), state)
        actions.append(action)
        if done:
            return actions, state, True
    return actions, state, False


def test_continue_terminates_in_one_step():
    from src.primitives.continue_ import Continue

    p = Continue()
    actions, state, terminated = _run_primitive(p)
    assert terminated, "Continue must finish in 1 step"
    assert len(actions) == 1


def test_continue_action_shape():
    from src.primitives.continue_ import Continue

    p = Continue()
    state = p.initial_state()
    action, _, _ = p.step(_dummy_obs(), state)
    assert action.shape == (7,)
    assert action.dtype == np.float32


def test_re_grasp_terminates_within_max_steps():
    from src.primitives.re_grasp import ReGrasp

    p = ReGrasp()
    actions, state, terminated = _run_primitive(p)
    assert terminated, "ReGrasp must terminate within max_steps"
    assert len(actions) <= p.max_steps


def test_re_grasp_action_shape():
    from src.primitives.re_grasp import ReGrasp

    p = ReGrasp()
    state = p.initial_state()
    for _ in range(p.max_steps):
        action, state, done = p.step(_dummy_obs(), state)
        assert action.shape == (7,), f"Bad shape: {action.shape}"
        assert action.dtype == np.float32
        if done:
            break


def test_re_approach_terminates_within_max_steps():
    from src.primitives.re_approach import ReApproach

    p = ReApproach()
    _, _, terminated = _run_primitive(p)
    assert terminated


def test_re_approach_action_shape():
    from src.primitives.re_approach import ReApproach

    p = ReApproach()
    state = p.initial_state()
    for _ in range(p.max_steps):
        action, state, done = p.step(_dummy_obs(), state)
        assert action.shape == (7,)
        assert action.dtype == np.float32
        if done:
            break


def test_align_then_insert_terminates_within_max_steps():
    from src.primitives.align_then_insert import AlignThenInsert

    p = AlignThenInsert()
    _, _, terminated = _run_primitive(p)
    assert terminated


def test_request_help_terminates_within_max_steps():
    from src.primitives.request_help import RequestHelp

    p = RequestHelp()
    _, _, terminated = _run_primitive(p)
    assert terminated


def test_request_help_sets_abort_in_state():
    from src.primitives.request_help import RequestHelp

    p = RequestHelp()
    _, final_state, _ = _run_primitive(p)
    assert final_state.abort_episode is True, "RequestHelp must set abort_episode=True"


def test_all_primitives_within_15_steps():
    """All primitives finish within 15 steps (global guard)."""
    from src.primitives.re_grasp import ReGrasp
    from src.primitives.re_approach import ReApproach
    from src.primitives.align_then_insert import AlignThenInsert
    from src.primitives.request_help import RequestHelp
    from src.primitives.continue_ import Continue

    for cls in [ReGrasp, ReApproach, AlignThenInsert, RequestHelp, Continue]:
        p = cls()
        _, _, terminated = _run_primitive(p, max_steps_override=15)
        assert terminated, f"{cls.__name__} did not finish in 15 steps"
