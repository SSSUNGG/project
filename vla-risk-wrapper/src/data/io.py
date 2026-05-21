from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.data.schema import EpisodeMeta, StepRecord


def write_step_batch(records: list[StepRecord], path: str) -> None:
    """Append a batch of StepRecords to a parquet file.

    Uses pyarrow for efficient columnar storage. Arrays (vla_hidden, action, etc.)
    are stored as list columns. Creates the file if it doesn't exist.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    if not records:
        return

    rows = [r.to_dict() for r in records]
    df = pd.DataFrame(rows)

    # Convert ndarray columns to list of lists for parquet compatibility
    for col in ["vla_hidden", "action", "action_history", "proprio"]:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
    if "rgb_thumb" in df.columns:
        df["rgb_thumb"] = df["rgb_thumb"].apply(
            lambda x: x.tolist() if isinstance(x, np.ndarray) else x
        )

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    table = pa.Table.from_pandas(df, preserve_index=False)

    if path.exists():
        existing = pq.read_table(str(path))
        combined = pa.concat_tables([existing, table])
        pq.write_table(combined, str(path))
    else:
        pq.write_table(table, str(path))


def write_episode_meta(meta: EpisodeMeta, path: str) -> None:
    """Append one EpisodeMeta row to a parquet file."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    row = meta.to_dict()
    row["primitive_history"] = str(row["primitive_history"])  # serialize list as string

    df = pd.DataFrame([row])
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    table = pa.Table.from_pandas(df, preserve_index=False)

    if path.exists():
        existing = pq.read_table(str(path))
        combined = pa.concat_tables([existing, table])
        pq.write_table(combined, str(path))
    else:
        pq.write_table(table, str(path))


def read_steps(path: str, filters: Optional[dict] = None) -> pd.DataFrame:
    """Read step parquet. Optionally filter by dict of {column: value}."""
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    df = table.to_pandas()

    if filters:
        mask = pd.Series([True] * len(df))
        for col, val in filters.items():
            if col in df.columns:
                mask &= df[col] == val
        df = df[mask].reset_index(drop=True)

    # Reconstruct numpy arrays from list columns
    for col in ["vla_hidden", "action", "proprio"]:
        if col in df.columns:
            df[col] = df[col].apply(
                lambda x: np.array(x, dtype=np.float32) if not isinstance(x, np.ndarray) else x
            )
    if "action_history" in df.columns:
        df["action_history"] = df["action_history"].apply(
            lambda x: np.array(x, dtype=np.float32).reshape(-1, 7)
            if not isinstance(x, np.ndarray) else x
        )

    return df


def read_meta(path: str) -> pd.DataFrame:
    """Read episode meta parquet."""
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    return table.to_pandas()


def make_splits(
    meta_df: pd.DataFrame,
    ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 42,
) -> dict[str, list[int]]:
    """Split by seed (not by step or episode) to avoid data leakage.

    Returns dict with keys 'train', 'val', 'test', each containing a list of seed values.
    """
    rng = np.random.default_rng(seed)
    seeds = sorted(meta_df["seed"].unique().tolist())
    seeds_arr = np.array(seeds)
    rng.shuffle(seeds_arr)
    n = len(seeds_arr)
    n_tr = int(n * ratios[0])
    n_va = int(n * ratios[1])
    return {
        "train": seeds_arr[:n_tr].tolist(),
        "val": seeds_arr[n_tr : n_tr + n_va].tolist(),
        "test": seeds_arr[n_tr + n_va :].tolist(),
    }
