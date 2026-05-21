"""Phase 2: Train the risk detector on collected step data.

Usage:
    python scripts/phase2_train_detector.py
    python scripts/phase2_train_detector.py detector=mlp_hidden_only
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import hydra
from omegaconf import DictConfig

from src.utils.seeding import set_global_seed
from src.utils.colab import get_drive_root


@hydra.main(config_path="../configs", config_name="base", version_base="1.3")
def main(cfg: DictConfig) -> None:
    set_global_seed(cfg.seed)

    drive_root = get_drive_root()
    step_dir = Path(drive_root) / "data" / "steps"
    meta_dir = Path(drive_root) / "data" / "meta"
    ckpt_dir = Path(drive_root) / "checkpoints"
    results_dir = Path(drive_root) / "results"
    metrics_path = str(results_dir / "phase2_metrics.json")

    from src.data.io import read_meta, make_splits
    from src.detector.dataset import make_dataloader
    from src.detector.features import DetectorConfig
    from src.detector.train import train_detector
    from src.detector.calibrate import calibrate_and_save
    from src.eval.metrics import step_auroc, lead_time_at_recall, ece

    tasks = cfg.experiment.get("tasks", ["peg_insertion", "assembling_kits"])
    det_cfg = DetectorConfig.from_omegaconf(cfg.detector)
    H = cfg.env.get("horizon_H", 10)

    all_results = {}

    for task_id in tasks:
        step_path = str(step_dir / f"{task_id}_steps.parquet")
        meta_path = str(meta_dir / f"{task_id}_meta.parquet")

        if not Path(meta_path).exists():
            print(f"WARNING: No data for {task_id} at {meta_path}. Skipping.")
            continue

        meta_df = read_meta(meta_path)
        splits = make_splits(meta_df, ratios=(0.7, 0.15, 0.15), seed=cfg.seed)

        print(f"[{task_id}] seeds — train:{len(splits['train'])} val:{len(splits['val'])} test:{len(splits['test'])}")

        train_loader = make_dataloader(
            step_path, meta_path, splits["train"], det_cfg, H=H,
            batch_size=cfg.detector.train.batch_size, shuffle=True
        )
        val_loader = make_dataloader(
            step_path, meta_path, splits["val"], det_cfg, H=H,
            batch_size=cfg.detector.train.batch_size, shuffle=False
        )

        ckpt_path = train_detector(
            cfg=cfg.detector,
            train_loader=train_loader,
            val_loader=val_loader,
            task_id=task_id,
            output_dir=str(ckpt_dir),
            metrics_path=metrics_path,
            device=cfg.device,
        )

        # Calibrate on second half of val set
        val_calib_loader = make_dataloader(
            step_path, meta_path, splits["val"][:len(splits["val"])//2], det_cfg, H=H,
            batch_size=cfg.detector.train.batch_size, shuffle=False
        )
        calibrate_and_save(str(ckpt_path), val_calib_loader, device=cfg.device)

        print(f"[{task_id}] Checkpoint saved: {ckpt_path}")

    print(f"\nPhase 2 complete. Metrics: {metrics_path}")


if __name__ == "__main__":
    main()
