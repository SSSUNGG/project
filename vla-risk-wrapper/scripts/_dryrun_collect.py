"""Dry-run: collect 3 episodes from PickCube-v1 and inspect the parquet.

Run: python scripts/_dryrun_collect.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import tempfile
import numpy as np

from omegaconf import OmegaConf

def main():
    import gymnasium as gym
    from src.env.factory import make_env
    from src.env.observation import normalize_obs
    from src.policy.octo_policy import OctoPolicy
    from src.data.collect import collect_episode
    from src.data.io import write_step_batch, write_episode_meta, read_steps, read_meta

    cfg = OmegaConf.create({
        "env": {
            "id": "PickCube-v1",
            "obs_mode": "state_dict",
            "control_mode": "pd_ee_delta_pose",
            "robot": "panda",
            "max_episode_steps": 10,
            "render_mode": None,
            "randomization": {"strength": "medium"},
        },
        "policy": {"hf_id": "rail-berkeley/octo-base"},
        "detector": {"input": {"action_history_K": 8}},
        "device": "cpu",
    })

    print("Making env...")
    env = make_env(cfg, seed=0)

    print("Loading policy (dummy mode if octo not installed)...")
    policy = OctoPolicy(cfg.policy, device="cpu")

    with tempfile.TemporaryDirectory() as tmpdir:
        step_path = str(Path(tmpdir) / "dryrun_steps.parquet")
        meta_path = str(Path(tmpdir) / "dryrun_meta.parquet")

        n_episodes = 3
        all_records = []
        all_metas = []

        for i in range(n_episodes):
            print(f"Collecting episode {i+1}/{n_episodes}...")
            records, meta = collect_episode(
                env, policy, "pick up the cube",
                seed=i, cfg=cfg, episode_seed=i * 1000
            )
            all_records.extend(records)
            all_metas.append(meta)

        write_step_batch(all_records, step_path)
        for meta in all_metas:
            write_episode_meta(meta, meta_path)

        # Verify
        steps_df = read_steps(step_path)
        meta_df = read_meta(meta_path)

        print(f"\n=== Dry-run results ===")
        print(f"Steps: {len(steps_df)} rows")
        print(f"Episodes: {len(meta_df)} rows")
        print(f"Step columns: {list(steps_df.columns)}")

        if "vla_hidden" in steps_df.columns:
            sample = steps_df.iloc[0]["vla_hidden"]
            print(f"vla_hidden sample shape: {np.array(sample).shape}")
            print(f"vla_hidden dtype: {np.array(sample).dtype}")

        print(f"\nSuccess rates: {meta_df['success'].tolist()}")
        print("\nDry-run passed!")

    env.close()


if __name__ == "__main__":
    main()
