"""
Phase 1: 베이스라인 실행 — 실패 확인

목표:
  1. ManiSkill3 환경 동작 확인
  2. ScriptedBaseline으로 N episodes 실행
  3. 성공/실패 로깅 + 성공률 확인
  4. Phase 목표 대역 체크 (PickCube/StackCube 40~70%, YCB 30~60%)

실행 (Colab / GPU 서버):
  python -m experiments.phase1_baseline --task PickCube-v1 --n_seeds 5 --eps 10 --debug_obs
  python -m experiments.phase1_baseline --task PickCube-v1 --n_seeds 5 --eps 10 --save
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from rgvla.envs.make_env import make_env, RANDOMIZATION_PRESETS
from rgvla.policy.baseline_scripted import ScriptedBaseline
from rgvla.utils.seeding import make_seed_grid


MAX_STEPS_PER_TASK = {
    "PickCube-v1":       200,
    "StackCube-v1":      300,
    "PickSingleYCB-v1":  250,
    "PegInsertionSide-v1": 400,
}

SUCCESS_TARGET = {
    "PickCube-v1":        (0.40, 0.70),
    "StackCube-v1":       (0.40, 0.70),
    "PickSingleYCB-v1":   (0.30, 0.60),
    "PegInsertionSide-v1":(0.20, 0.50),
}


def run_episode(env, policy, seed: int, episode: int, max_steps: int) -> dict:
    obs, info = env.reset_with(seed, episode)
    policy.reset(obs)

    success = False
    n_steps = 0
    t0 = time.perf_counter()

    for step in range(max_steps):
        action = policy.act(obs)
        # ManiSkill3 num_envs=1 → action에 배치 차원 추가
        action_in = action[np.newaxis, :] if action.ndim == 1 else action

        obs, reward, terminated, truncated, info = env.step(action_in)
        n_steps = step + 1
        success = env.is_success(info)

        # terminated/truncated 체크 (배치 차원 처리)
        term = bool(np.any(terminated)) if hasattr(terminated, "__iter__") else bool(terminated)
        trunc = bool(np.any(truncated)) if hasattr(truncated, "__iter__") else bool(truncated)

        if term or trunc or success:
            break

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return {
        "seed":         seed,
        "episode":      episode,
        "success":      success,
        "n_steps":      n_steps,
        "elapsed_ms":   elapsed_ms,
        "ms_per_step":  elapsed_ms / max(n_steps, 1),
        "final_phase":  policy._phase,
    }


def print_obs_structure(env, seed=0, ep=0):
    print("\n=== obs 구조 (obs_mode=state) ===")
    obs, _ = env.reset_with(seed, ep)
    env.print_obs_structure(obs)

    print("\n=== action space ===")
    print(f"  shape : {env.action_space.shape}")
    print(f"  low   : {env.action_space.low}")
    print(f"  high  : {env.action_space.high}")

    print("\n=== 샘플 위치 ===")
    print(f"  tcp_pos  : {env.get_tcp_pos(obs)}")
    try:
        print(f"  obj_pos  : {env.get_obj_pos(obs)}")
    except KeyError as e:
        print(f"  obj_pos  : [KeyError: {e}]")
    try:
        print(f"  goal_pos : {env.get_goal_pos(obs)}")
    except KeyError as e:
        print(f"  goal_pos : [KeyError: {e}]")
    print()


def main():
    parser = argparse.ArgumentParser(description="Phase 1: baseline 실행")
    parser.add_argument("--task",          default="PickCube-v1")
    parser.add_argument("--n_seeds",       type=int, default=5)
    parser.add_argument("--eps",           type=int, default=10,
                        help="seed 당 episode 수")
    parser.add_argument("--randomization", default="eval",
                        choices=["eval", "collect"],
                        help="eval=낮음, collect=실패율 30~50%")
    parser.add_argument("--debug_obs",     action="store_true",
                        help="첫 episode에서 obs 구조 출력")
    parser.add_argument("--save",          action="store_true",
                        help="results/ 에 JSON 저장")
    args = parser.parse_args()

    max_steps = MAX_STEPS_PER_TASK.get(args.task, 200)
    rand_cfg  = RANDOMIZATION_PRESETS[args.randomization]

    print(f"Task          : {args.task}")
    print(f"Randomization : {args.randomization}  {rand_cfg}")
    print(f"Grid          : {args.n_seeds} seeds × {args.eps} eps "
          f"= {args.n_seeds * args.eps} episodes")
    print(f"Max steps     : {max_steps}")

    env    = make_env(args.task, randomization=rand_cfg)
    policy = ScriptedBaseline(task=args.task)

    if args.debug_obs:
        print_obs_structure(env)

    seed_grid = make_seed_grid(args.n_seeds, args.eps)
    results   = []

    # ─── 헤더 ──────────────────────────────────────────
    print(f"\n{'seed':>4} {'ep':>3}  {'ok':>4}  {'steps':>5}  "
          f"{'phase':>20}  {'ms/step':>9}")
    print("─" * 60)

    for seed, ep in seed_grid:
        r = run_episode(env, policy, seed, ep, max_steps)
        results.append(r)
        ok_mark = "✓" if r["success"] else "✗"
        print(f"{r['seed']:>4} {r['episode']:>3}  {ok_mark:>4}  "
              f"{r['n_steps']:>5}  {r['final_phase']:>20}  "
              f"{r['ms_per_step']:>8.1f}ms")

    env.close()

    # ─── 집계 ──────────────────────────────────────────
    suc = [r["success"] for r in results]
    sr  = float(np.mean(suc))
    print("\n" + "═" * 60)
    print(f"Task            : {args.task}")
    print(f"Episodes        : {len(results)}")
    print(f"Success         : {sum(suc)}/{len(results)}  ({sr*100:.1f}%)")
    print(f"Mean steps      : {np.mean([r['n_steps'] for r in results]):.1f}")
    print(f"Mean latency    : {np.mean([r['ms_per_step'] for r in results]):.2f} ms/step")

    # Phase 1 목표 대역 판정
    lo, hi = SUCCESS_TARGET.get(args.task, (0.30, 0.70))
    print()
    if lo <= sr <= hi:
        print(f"✓ 성공률 {sr*100:.1f}% — 목표 대역 [{lo*100:.0f}%,{hi*100:.0f}%] 내. Phase 2로 진행 가능.")
    elif sr > hi:
        print(f"△ 성공률 {sr*100:.1f}% — 너무 높음. "
              f"--randomization collect 로 실패율을 높이거나 randomization 강도를 올리세요.")
    else:
        print(f"✗ 성공률 {sr*100:.1f}% — 너무 낮음. "
              f"RANDOMIZATION_PRESETS 의 obj_xy_std 를 낮춰 randomization 약화 필요.")

    if args.save:
        out_dir  = Path("results") / args.task
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "phase1_baseline.json"
        with open(out_path, "w") as f:
            json.dump({
                "task":           args.task,
                "randomization":  args.randomization,
                "n_episodes":     len(results),
                "success_rate":   sr,
                "n_success":      int(sum(suc)),
                "mean_steps":     float(np.mean([r["n_steps"] for r in results])),
                "mean_latency_ms": float(np.mean([r["ms_per_step"] for r in results])),
                "episodes":       results,
            }, f, indent=2)
        print(f"\n결과 저장: {out_path}")

    return sr


if __name__ == "__main__":
    main()
