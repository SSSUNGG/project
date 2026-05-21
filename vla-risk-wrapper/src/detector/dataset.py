from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from src.data.io import read_meta, read_steps
from src.detector.features import DetectorConfig, build_feature_vector
from src.detector.labeling import horizon_label


@dataclass
class StepSample:
    features: np.ndarray   # (D,) concatenated inputs (raw, before projection)
    label: int             # 0 or 1
    episode_id: str
    step: int
    task_id: str
    seed: int


class StepDataset(Dataset):
    """Dataset that joins step parquet with episode meta, applies horizon labeling."""

    def __init__(
        self,
        step_parquet_path: str,
        meta_parquet_path: str,
        split_seeds: list[int],
        cfg: DetectorConfig,
        H: int,
    ):
        self.cfg = cfg
        self.H = H
        self.samples: list[StepSample] = []
        self._build(step_parquet_path, meta_parquet_path, split_seeds)

    def _build(
        self,
        step_path: str,
        meta_path: str,
        split_seeds: list[int],
    ) -> None:
        meta_df = read_meta(meta_path)
        meta_df = meta_df[meta_df["seed"].isin(split_seeds)].reset_index(drop=True)

        if len(meta_df) == 0:
            return

        steps_df = read_steps(step_path)
        steps_df = steps_df[steps_df["seed"].isin(split_seeds)].reset_index(drop=True)

        meta_lookup = {
            row["episode_id"]: row
            for _, row in meta_df.iterrows()
        }

        for episode_id, ep_steps in steps_df.groupby("episode_id"):
            if episode_id not in meta_lookup:
                continue
            meta_row = meta_lookup[episode_id]
            ep_steps = ep_steps.sort_values("step").reset_index(drop=True)

            success = bool(meta_row["success"])
            fail_step = int(meta_row["fail_step"])
            ep_length = len(ep_steps)
            labels = horizon_label(success, fail_step, ep_length, self.H)

            for i, (_, row) in enumerate(ep_steps.iterrows()):
                vla_h = np.asarray(row["vla_hidden"], dtype=np.float32)
                ah = np.asarray(row["action_history"], dtype=np.float32)
                if ah.ndim == 1:
                    ah = ah.reshape(-1, 7)
                prop = np.asarray(row["proprio"], dtype=np.float32)

                feat = build_feature_vector(vla_h, ah, prop, self.cfg)

                self.samples.append(
                    StepSample(
                        features=feat,
                        label=int(labels[i]),
                        episode_id=str(episode_id),
                        step=int(row["step"]),
                        task_id=str(row["task_id"]),
                        seed=int(row["seed"]),
                    )
                )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        return (
            torch.tensor(s.features, dtype=torch.float32),
            torch.tensor(s.label, dtype=torch.float32),
        )


def make_dataloader(
    step_path: str,
    meta_path: str,
    split_seeds: list[int],
    cfg: DetectorConfig,
    H: int,
    batch_size: int = 256,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    dataset = StepDataset(step_path, meta_path, split_seeds, cfg, H)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
