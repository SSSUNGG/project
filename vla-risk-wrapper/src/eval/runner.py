"""Evaluation runner for Phase 3 — three-condition seed-grid experiments."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from omegaconf import DictConfig

from src.data.schema import EpisodeMeta, new_episode_id
from src.data.io import write_episode_meta
from src.env.observation import normalize_obs, extract_proprio
from src.wrapper.risk_gated import RiskGatedWrapper


@dataclass
class EvalResult:
    condition_id: str
    task_id: str
    successes: list[bool] = field(default_factory=list)
    latencies_ms: list[float] = field(default_factory=list)
    reasoner_calls_per_ep: list[int] = field(default_factory=list)
    episode_metas: list[dict] = field(default_factory=list)


def run_condition(
    condition_id: str,
    task_id: str,
    seed_grid: list[tuple[int, int]],
    wrapper: RiskGatedWrapper,
    cfg: DictConfig,
    tau: Optional[float] = None,
    meta_parquet_path: Optional[str] = None,
    completed_keys: Optional[set] = None,
) -> EvalResult:
    """Run one condition (C1/C2/C3) over the full seed grid.

    Args:
        condition_id:      'C1_baseline', 'C2_always_reasoning', or 'C3_risk_gated'.
        task_id:           Short task name (e.g., 'peg_insertion').
        seed_grid:         List of (seed, ep_idx) tuples.
        wrapper:           Configured RiskGatedWrapper.
        cfg:               Full Hydra config.
        tau:               Tau value for C3 (None for C1/C2).
        meta_parquet_path: If set, append episode meta to this file every sync interval.
        completed_keys:    Set of (seed, ep_idx) already done (for resume).

    Returns:
        EvalResult with per-episode metrics.
    """
    import gymnasium as gym
    from src.env.factory import make_env

    result = EvalResult(condition_id=condition_id, task_id=task_id)
    instruction = _task_instruction(task_id)
    sync_interval = cfg.logging.get("drive_sync_every_n_episodes", 5)
    pending_metas: list[EpisodeMeta] = []

    env = make_env(cfg, seed=0)

    for seed, ep_idx in seed_grid:
        key = (seed, ep_idx)
        if completed_keys and key in completed_keys:
            continue

        episode_seed = seed * 1000 + ep_idx
        obs, _ = env.reset(seed=episode_seed)
        obs = normalize_obs(obs, env)
        obs["_cached_proprio"] = extract_proprio(obs, env)

        wrapper.reset(instruction)

        success = False
        fail_step = -1
        total_latency_ms = 0.0
        wall_start = time.perf_counter()
        max_steps = cfg.env.get("max_episode_steps", 100)
        n_steps = 0

        for step_idx in range(max_steps):
            step_info = wrapper.step(obs)
            total_latency_ms += sum(step_info.latency_breakdown_ms.values())
            n_steps += 1

            action = step_info.action
            obs_next, reward, done, truncated, info = env.step(action)
            obs_next = normalize_obs(obs_next, env)
            obs_next["_cached_proprio"] = extract_proprio(obs_next, env)

            if bool(info.get("success", False)):
                success = True
                break
            if step_info.abort_episode:
                break
            if done or truncated:
                break

            obs = obs_next

        if not success:
            fail_step = step_idx

        wall_elapsed = time.perf_counter() - wall_start
        mean_lat = total_latency_ms / max(n_steps, 1)

        result.successes.append(success)
        result.latencies_ms.append(mean_lat)
        result.reasoner_calls_per_ep.append(wrapper.reasoner_call_count)

        meta = EpisodeMeta(
            episode_id=new_episode_id(),
            task_id=task_id,
            seed=seed,
            policy_id=type(wrapper.policy).__name__.lower(),
            randomization_strength=cfg.env.randomization.get("strength", "medium"),
            success=success,
            fail_step=fail_step,
            episode_length=n_steps,
            wall_clock_seconds=float(wall_elapsed),
            mean_latency_ms_per_step=float(mean_lat),
            condition=condition_id,
            tau=float(tau) if tau is not None else -1.0,
            reasoner_calls=wrapper.reasoner_call_count,
            primitive_history=list(wrapper.primitive_history),
        )
        result.episode_metas.append(meta.to_dict())
        pending_metas.append(meta)

        if len(pending_metas) >= sync_interval and meta_parquet_path:
            for m in pending_metas:
                write_episode_meta(m, meta_parquet_path)
            pending_metas.clear()

    # Flush remaining
    if pending_metas and meta_parquet_path:
        for m in pending_metas:
            write_episode_meta(m, meta_parquet_path)

    env.close()
    return result


def run_phase3(
    cfg: DictConfig,
    wrapper_factory,          # callable(condition_id, tau) -> RiskGatedWrapper
    output_dir: Optional[str] = None,
    resume: bool = False,
) -> dict[str, EvalResult]:
    """Run all Phase 3 conditions across all tasks.

    Args:
        cfg:             Hydra phase3_wrapper_eval config.
        wrapper_factory: Callable that builds a wrapper given condition_id and tau.
        output_dir:      Directory for meta parquet files and summary CSV.
        resume:          If True, skip (seed, ep) pairs already in output_dir.

    Returns:
        dict mapping condition_id → EvalResult.
    """
    import pandas as pd
    from src.data.io import read_meta

    n_seeds = cfg.eval.get("n_seeds", 5)
    n_eps = cfg.eval.get("n_episodes_per_seed", 100)
    seed_grid = [(s, e) for s in range(n_seeds) for e in range(n_eps)]

    output_path = Path(output_dir) if output_dir else Path("results")
    output_path.mkdir(parents=True, exist_ok=True)

    all_results: dict[str, EvalResult] = {}

    for condition in cfg.conditions:
        cond_id = condition["id"]
        wrapper_mode = condition.get("wrapper", "none")
        tau_sweep = condition.get("tau_sweep", [None])

        for tau in (tau_sweep if wrapper_mode == "risk_gated" else [None]):
            run_id = cond_id if tau is None else f"{cond_id}_tau{tau}"

            completed = set()
            meta_path = str(output_path / f"phase3_meta_{run_id}.parquet")
            if resume and Path(meta_path).exists():
                try:
                    existing_meta = read_meta(meta_path)
                    for _, row in existing_meta.iterrows():
                        completed.add((int(row["seed"]), int(row.get("ep_idx", 0))))
                except Exception:  # noqa: BLE001
                    pass

            for task_id in cfg.tasks:
                wrapper = wrapper_factory(condition_id=cond_id, tau=tau, mode=wrapper_mode)
                result = run_condition(
                    condition_id=run_id,
                    task_id=task_id,
                    seed_grid=seed_grid,
                    wrapper=wrapper,
                    cfg=cfg,
                    tau=tau,
                    meta_parquet_path=meta_path,
                    completed_keys=completed,
                )
                all_results[f"{run_id}_{task_id}"] = result

    _save_summary_csv(all_results, output_path / "phase3_summary.csv")
    return all_results


def _task_instruction(task_id: str) -> str:
    instructions = {
        "peg_insertion": "Insert the peg into the slot.",
        "assembling_kits": "Assemble the kit by inserting all pieces.",
        "pick_single_ycb": "Pick up the object.",
        "pick_cube": "Pick up the cube.",
        "stack_cube": "Stack the cube on top of the other cube.",
    }
    return instructions.get(task_id, "Complete the manipulation task.")


def _save_summary_csv(results: dict[str, EvalResult], path: Path) -> None:
    import pandas as pd
    from src.eval.bootstrap import bootstrap_mean_ci

    rows = []
    for key, res in results.items():
        if not res.successes:
            continue
        sr, ci_lo, ci_hi = bootstrap_mean_ci(np.array(res.successes, dtype=float))
        mean_lat = float(np.mean(res.latencies_ms)) if res.latencies_ms else 0.0
        mean_calls = float(np.mean(res.reasoner_calls_per_ep)) if res.reasoner_calls_per_ep else 0.0
        rows.append({
            "condition": res.condition_id,
            "task": res.task_id,
            "success_mean": round(sr, 4),
            "success_ci_low": round(ci_lo, 4),
            "success_ci_high": round(ci_hi, 4),
            "latency_ms_per_step": round(mean_lat, 2),
            "reasoner_calls_per_ep": round(mean_calls, 2),
            "n_episodes": len(res.successes),
        })

    if rows:
        pd.DataFrame(rows).to_csv(path, index=False)
