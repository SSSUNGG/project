"""Phase 1: Run baseline VLA evaluation across tasks and seeds.

Usage:
    python scripts/phase1_run_baseline.py
    python scripts/phase1_run_baseline.py env=peg_insertion
"""
from __future__ import annotations

import os
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

    drive_root = get_drive_root()
    ensure_drive_dirs()
    results_dir = Path(drive_root) / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Import here to avoid heavy imports at module level
    from src.env.factory import make_env
    from src.policy.octo_policy import OctoPolicy
    from src.data.collect import collect_episode
    from src.data.io import write_episode_meta, write_step_batch
    from src.wrapper.risk_gated import RiskGatedWrapper
    from src.reasoner.rule_based import RuleBasedReasoner
    from src.primitives.continue_ import Continue

    policy = OctoPolicy(cfg.policy, device=cfg.device)
    reasoner = RuleBasedReasoner(cfg.reasoner)
    primitives = {"continue": Continue()}

    wrapper = RiskGatedWrapper(
        policy=policy,
        detector=None,
        reasoner=reasoner,
        primitives=primitives,
        tau=1.0,    # tau=1.0 means never fires in risk_gated mode
        mode="none",
        device=cfg.device,
    )

    n_seeds = cfg.experiment.eval.get("n_seeds", 5)
    n_eps = cfg.experiment.eval.get("n_episodes_per_seed", 100)
    task_id = cfg.env.id.split("-")[0].lower()
    instruction = _task_instruction(task_id)

    step_buf = []
    meta_path = str(results_dir / f"phase1_meta_{task_id}.parquet")
    step_path = str(Path(drive_root) / "data" / "steps" / f"phase1_{task_id}.parquet")

    env = make_env(cfg, seed=0)

    total_successes = 0
    total_episodes = 0

    for seed in range(n_seeds):
        for ep_idx in range(n_eps):
            ep_seed = seed * 1000 + ep_idx
            records, meta = collect_episode(
                env, policy, instruction, seed=seed, cfg=cfg, episode_seed=ep_seed
            )
            meta.condition = "C1_baseline"
            step_buf.extend(records)
            write_episode_meta(meta, meta_path)

            if len(step_buf) >= 5 * n_eps:
                write_step_batch(step_buf, step_path)
                step_buf.clear()

            total_successes += int(meta.success)
            total_episodes += 1

            print(
                f"[{task_id}] seed={seed} ep={ep_idx} success={meta.success} "
                f"({total_successes}/{total_episodes})"
            )

    if step_buf:
        write_step_batch(step_buf, step_path)

    env.close()
    print(f"\nPhase 1 done. Success rate: {total_successes/total_episodes:.3f}")


def _task_instruction(task_id: str) -> str:
    instructions = {
        "peginsertionside": "Insert the peg into the slot.",
        "assemblingkits": "Assemble the kit.",
        "picksingleycb": "Pick up the object.",
        "pickcube": "Pick up the cube.",
        "stackcube": "Stack the cube.",
    }
    return instructions.get(task_id.lower().replace("-", ""), "Complete the task.")


if __name__ == "__main__":
    main()
