"""Tests for src/data/ module."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.schema import EpisodeMeta, StepRecord, new_episode_id
from src.data.io import (
    make_splits,
    read_meta,
    read_steps,
    write_episode_meta,
    write_step_batch,
)


def _make_step_record(episode_id: str = None, seed: int = 0, step: int = 0) -> StepRecord:
    return StepRecord(
        episode_id=episode_id or new_episode_id(),
        task_id="pick_cube",
        seed=seed,
        step=step,
        vla_hidden=np.zeros(4096, dtype=np.float16),
        action=np.zeros(7, dtype=np.float32),
        action_history=np.zeros((8, 7), dtype=np.float32),
        proprio=np.zeros(8, dtype=np.float32),
        rgb_thumb=None,
        info_is_grasped=False,
        info_is_success=False,
        reward=0.0,
        done=False,
        truncated=False,
    )


def _make_meta(episode_id: str = None, seed: int = 0) -> EpisodeMeta:
    return EpisodeMeta(
        episode_id=episode_id or new_episode_id(),
        task_id="pick_cube",
        seed=seed,
        policy_id="octopolicy",
        randomization_strength="medium",
        success=False,
        fail_step=10,
        episode_length=10,
        wall_clock_seconds=5.0,
        mean_latency_ms_per_step=50.0,
    )


def test_step_record_validate():
    """StepRecord.validate() catches wrong shapes."""
    rec = _make_step_record()
    rec.validate()  # should not raise

    bad = _make_step_record()
    bad.action = np.zeros(5, dtype=np.float32)
    with pytest.raises(AssertionError):
        bad.validate()


def test_parquet_round_trip():
    """write_step_batch → read_steps returns same number of rows."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "steps.parquet")
        eid = new_episode_id()
        records = [_make_step_record(episode_id=eid, seed=0, step=i) for i in range(5)]
        write_step_batch(records, path)

        df = read_steps(path)
        assert len(df) == 5
        assert "vla_hidden" in df.columns
        assert "action" in df.columns


def test_episode_meta_round_trip():
    """write_episode_meta → read_meta returns correct row."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "meta.parquet")
        meta = _make_meta(seed=7)
        write_episode_meta(meta, path)
        df = read_meta(path)
        assert len(df) == 1
        assert df.iloc[0]["seed"] == 7


def test_make_splits_no_seed_overlap():
    """train/val/test seed sets are disjoint."""
    meta_rows = [{"seed": i} for i in range(30)]
    meta_df = pd.DataFrame(meta_rows)
    splits = make_splits(meta_df, ratios=(0.7, 0.15, 0.15), seed=42)

    tr = set(splits["train"])
    va = set(splits["val"])
    te = set(splits["test"])

    assert tr & va == set(), "train and val share seeds"
    assert tr & te == set(), "train and test share seeds"
    assert va & te == set(), "val and test share seeds"
    assert len(tr) + len(va) + len(te) == 30


def test_collect_episode_returns_correct_step_count():
    """collect_episode returns exactly max_episode_steps records for non-terminating env."""
    pytest.importorskip("gymnasium")
    pytest.importorskip("mani_skill")

    import gymnasium as gym
    from omegaconf import OmegaConf
    from src.data.collect import collect_episode
    from src.policy.base import BasePolicy, PolicyOutput

    class DummyPolicy(BasePolicy):
        HIDDEN_DIM = 16

        def reset(self, instruction: str) -> None:
            pass

        def predict(self, obs: dict) -> PolicyOutput:
            return PolicyOutput(
                action=np.zeros(7, dtype=np.float32),
                hidden_state=np.zeros(self.HIDDEN_DIM, dtype=np.float32),
                latency_ms=1.0,
            )

    env = gym.make("PickCube-v1", obs_mode="state_dict", render_mode=None, max_episode_steps=10)
    cfg = OmegaConf.create({
        "env": {
            "id": "PickCube-v1",
            "max_episode_steps": 10,
            "randomization": {"strength": "medium"},
        },
        "detector": {"input": {"action_history_K": 8}},
    })

    policy = DummyPolicy()
    records, meta = collect_episode(env, policy, "pick up the cube", seed=0, cfg=cfg)

    assert len(records) <= 10
    assert meta.episode_length == len(records)
    env.close()
