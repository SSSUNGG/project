"""Phase 2: Collect step-level data for detector training.

Usage:
    python scripts/phase2_collect_data.py
    python scripts/phase2_collect_data.py --resume
    python scripts/phase2_collect_data.py experiment=phase2_collect_data
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import hydra
from omegaconf import DictConfig

from src.utils.seeding import set_global_seed
from src.utils.colab import get_drive_root, ensure_drive_dirs


@hydra.main(config_path="../configs", config_name="base", version_base="1.3")
def main(cfg: DictConfig) -> None:
    set_global_seed(cfg.seed)
    ensure_drive_dirs()

    drive_root = get_drive_root()
    step_dir = Path(drive_root) / "data" / "steps"
    meta_dir = Path(drive_root) / "data" / "meta"
    step_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    from src.env.factory import make_env
    from src.policy.octo_policy import OctoPolicy
    from src.data.collect import collect_episode
    from src.data.io import write_episode_meta, write_step_batch, read_meta

    policy = OctoPolicy(cfg.policy, device=cfg.device)

    tasks = cfg.experiment.get("tasks", ["peg_insertion", "assembling_kits"])
    n_episodes = cfg.experiment.collect.get("n_episodes_per_task", 1000)
    sync_every = cfg.experiment.collect.get("drive_sync_every_n_episodes", 5)
    resume = "--resume" in sys.argv

    for task_id in tasks:
        task_cfg = cfg.copy()
        env = make_env(task_cfg, seed=0)

        step_path = str(step_dir / f"{task_id}_steps.parquet")
        meta_path = str(meta_dir / f"{task_id}_meta.parquet")

        completed_ids = set()
        if resume and Path(meta_path).exists():
            try:
                meta_df = read_meta(meta_path)
                completed_ids = set(meta_df["episode_id"].tolist())
                print(f"Resuming {task_id}: {len(completed_ids)} episodes already done.")
            except Exception:  # noqa: BLE001
                pass

        instruction = _task_instruction(task_id)
        step_buf = []

        for ep_idx in range(n_episodes):
            seed = ep_idx % 100
            ep_seed = seed * 1000 + ep_idx

            records, meta = collect_episode(
                env, policy, instruction, seed=seed, cfg=cfg, episode_seed=ep_seed
            )

            if meta.episode_id in completed_ids:
                continue

            step_buf.extend(records)
            write_episode_meta(meta, meta_path)

            if (ep_idx + 1) % sync_every == 0:
                if step_buf:
                    write_step_batch(step_buf, step_path)
                    step_buf.clear()
                print(f"[{task_id}] {ep_idx+1}/{n_episodes} episodes collected.")

        if step_buf:
            write_step_batch(step_buf, step_path)
        env.close()
        print(f"Done collecting {task_id}.")


def _task_instruction(task_id: str) -> str:
    return {
        "peg_insertion": "Insert the peg into the slot.",
        "assembling_kits": "Assemble the kit by inserting all pieces.",
        "pick_single_ycb": "Pick up the object.",
        "pick_cube": "Pick up the cube.",
        "stack_cube": "Stack the cube on top of the other cube.",
    }.get(task_id, "Complete the manipulation task.")


if __name__ == "__main__":
    main()
