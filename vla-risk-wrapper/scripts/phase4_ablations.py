"""Phase 4: Ablation studies — detector features, horizon H, tau sweep.

Usage:
    python scripts/phase4_ablations.py
    python scripts/phase4_ablations.py experiment=phase4_ablation
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import hydra
from omegaconf import DictConfig

from src.utils.seeding import set_global_seed
from src.utils.colab import get_drive_root, ensure_drive_dirs


@hydra.main(config_path="../configs", config_name="base", version_base="1.3")
def main(cfg: DictConfig) -> None:
    set_global_seed(cfg.seed)
    ensure_drive_dirs()

    drive_root = get_drive_root()
    results_dir = Path(drive_root) / "results" / "phase4_ablations"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Detector feature ablations are run by re-training with different detector configs.
    # Horizon H ablations are run by re-labeling data with different H values.
    # This script logs which ablations have been run.

    ablation_cfg = cfg.get("experiment", {})
    tasks = ablation_cfg.get("tasks", ["peg_insertion", "assembling_kits"])
    horizon_sweep = ablation_cfg.get("horizon_sweep", [10, 20, 30])
    tau_sweep = ablation_cfg.get("tau_fine_sweep", [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])

    print("Phase 4 ablation plan:")
    print(f"  Tasks: {tasks}")
    print(f"  Horizon sweep: {horizon_sweep}")
    print(f"  Tau fine sweep: {tau_sweep}")
    print(f"  Detector ablations: {ablation_cfg.get('detector_ablations', [])}")
    print(f"  Output dir: {results_dir}")
    print("\nRun detector ablations with:")
    print("  python scripts/phase2_train_detector.py detector=mlp_hidden_only")
    print("  python scripts/phase2_train_detector.py detector=mlp_action_only")
    print("  python scripts/phase2_train_detector.py detector=mlp_proprio_only")
    print("  python scripts/phase2_train_detector.py detector=mlp_hybrid")
    print("\nThen re-run phase3 for each ablation.")


if __name__ == "__main__":
    main()
