"""Phase 3: Run three-condition wrapper evaluation (C1/C2/C3 × tau sweep).

Usage:
    python scripts/phase3_run_wrapper.py
    python scripts/phase3_run_wrapper.py --resume
    python scripts/phase3_run_wrapper.py experiment=phase3_wrapper_eval
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
    results_dir = Path(drive_root) / "results"
    ckpt_dir = Path(drive_root) / "checkpoints"
    results_dir.mkdir(parents=True, exist_ok=True)

    from src.policy.octo_policy import OctoPolicy
    from src.reasoner.rule_based import RuleBasedReasoner
    from src.primitives.re_grasp import ReGrasp
    from src.primitives.re_approach import ReApproach
    from src.primitives.align_then_insert import AlignThenInsert
    from src.primitives.request_help import RequestHelp
    from src.primitives.continue_ import Continue
    from src.wrapper.risk_gated import RiskGatedWrapper
    from src.utils.checkpoint import load_detector
    from src.eval.runner import run_condition

    resume = "--resume" in sys.argv

    policy = OctoPolicy(cfg.policy, device=cfg.device)
    reasoner = RuleBasedReasoner(cfg.reasoner)
    primitives = {
        "re_grasp": ReGrasp(),
        "re_approach": ReApproach(),
        "align_then_insert": AlignThenInsert(),
        "request_help": RequestHelp(),
        "continue": Continue(),
    }

    n_seeds = cfg.experiment.eval.get("n_seeds", 5)
    n_eps = cfg.experiment.eval.get("n_episodes_per_seed", 100)
    seed_grid = [(s, e) for s in range(n_seeds) for e in range(n_eps)]

    for task_id in cfg.experiment.get("tasks", ["peg_insertion", "assembling_kits"]):
        # Load detector for this task (if available)
        detector = None
        ckpt_path = ckpt_dir / f"detector_{task_id}_{cfg.detector.name}.pt"
        if ckpt_path.exists():
            try:
                detector = load_detector(str(ckpt_path), device=cfg.device)
                print(f"Loaded detector for {task_id} from {ckpt_path}")
            except Exception as e:
                print(f"WARNING: Could not load detector for {task_id}: {e}")

        for condition in cfg.experiment.conditions:
            cond_id = condition["id"]
            wrapper_mode = condition.get("wrapper", "none")
            tau_values = condition.get("tau_sweep", [None]) if wrapper_mode == "risk_gated" else [None]

            for tau in tau_values:
                run_id = cond_id if tau is None else f"{cond_id}_tau{tau}"
                meta_path = str(results_dir / f"phase3_meta_{run_id}_{task_id}.parquet")

                print(f"\n=== {run_id} | {task_id} | tau={tau} ===")

                wrapper = RiskGatedWrapper(
                    policy=policy,
                    detector=detector,
                    reasoner=reasoner,
                    primitives=primitives,
                    tau=float(tau) if tau is not None else 1.0,
                    mode=wrapper_mode,
                    device=cfg.device,
                )

                result = run_condition(
                    condition_id=run_id,
                    task_id=task_id,
                    seed_grid=seed_grid,
                    wrapper=wrapper,
                    cfg=cfg,
                    tau=tau,
                    meta_parquet_path=meta_path,
                )

                import numpy as np
                sr = float(np.mean(result.successes)) if result.successes else 0.0
                mean_lat = float(np.mean(result.latencies_ms)) if result.latencies_ms else 0.0
                print(f"  success_rate={sr:.3f} mean_latency={mean_lat:.1f}ms")

    print("\nPhase 3 complete.")


if __name__ == "__main__":
    main()
