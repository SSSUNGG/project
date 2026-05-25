"""
experiments/visualize.py

에피소드를 녹화하고 Colab에서 GIF/MP4로 표시한다.

Colab 사용법:
    from experiments.visualize import record_and_show
    record_and_show("PickCube-v1", n_episodes=3)
"""

from __future__ import annotations
import numpy as np
from pathlib import Path
import gymnasium as gym

import mani_skill.envs  # noqa

from rgvla.envs.make_env import MAX_EPISODE_STEPS
from rgvla.policy.baseline_scripted import ScriptedBaseline


# ─── 핵심: 에피소드 1개 녹화 ──────────────────────────────────

def record_episode(
    task: str,
    seed: int = 0,
    episode: int = 0,
    fps_skip: int = 2,   # 매 fps_skip 번째 frame만 저장 (GIF 용량 절약)
) -> dict:
    """
    1개 에피소드를 돌리며 RGB 프레임을 수집한다.

    Returns:
        frames   : list of (H, W, 3) uint8 arrays
        success  : bool
        n_steps  : int
        phase    : 마지막 phase 이름
    """
    max_steps = MAX_EPISODE_STEPS.get(task, 200)

    env = gym.make(
        task,
        obs_mode="state",
        control_mode="pd_ee_delta_pose",
        num_envs=1,
        max_episode_steps=max_steps,
        render_mode="rgb_array",   # ← 렌더링 ON
    )

    eseed = seed * 100_000 + episode
    obs, info = env.reset(seed=eseed)

    policy = ScriptedBaseline(task=task)
    policy.reset(obs)

    frames, success, n_steps = [], False, 0

    for step in range(max_steps):
        # 현재 상태 렌더링
        if step % fps_skip == 0:
            frame = _to_frame(env.render())
            if frame is not None:
                frames.append(frame)

        # 액션 적용
        action = policy.act(obs)
        obs, _, terminated, truncated, info = env.step(action[np.newaxis, :])
        n_steps = step + 1

        success = _to_bool(info.get("success", False))
        if _to_bool(terminated) or _to_bool(truncated) or success:
            frame = _to_frame(env.render())
            if frame is not None:
                frames.append(frame)
            break

    env.close()
    return {"frames": frames, "success": success,
            "n_steps": n_steps, "phase": policy._phase}


# ─── 여러 에피소드 녹화 + Colab 표시 ─────────────────────────

def record_and_show(
    task: str = "PickCube-v1",
    n_episodes: int = 3,
    seed: int = 0,
    fps: int = 15,
    out_dir: str = "videos",
    fps_skip: int = 2,
):
    """
    n_episodes 개를 녹화하고 Colab 인라인 표시.
    성공/실패 에피소드를 모두 포함해 비교할 수 있다.
    """
    Path(out_dir).mkdir(exist_ok=True)
    summary = []

    for ep in range(n_episodes):
        print(f"  recording episode {ep} ...", end=" ", flush=True)
        result = record_episode(task, seed=seed, episode=ep, fps_skip=fps_skip)
        ok = "✓ success" if result["success"] else f"✗ fail ({result['phase']})"
        print(f"{result['n_steps']} steps  {ok}")
        summary.append(result)

        if result["frames"]:
            gif_path = f"{out_dir}/{task}_ep{ep}_{'ok' if result['success'] else 'fail'}.gif"
            save_gif(result["frames"], gif_path, fps=fps)
            _display_gif(gif_path, label=f"ep={ep}  {ok}  ({result['n_steps']} steps)")

    # 전체 성공률 출력
    sr = np.mean([r["success"] for r in summary])
    print(f"\n성공률: {sum(r['success'] for r in summary)}/{n_episodes} ({sr*100:.0f}%)")
    return summary


# ─── 개별 frame 캡처 (단순 확인용) ───────────────────────────

def show_frames(task: str = "PickCube-v1", seed: int = 0, episode: int = 0,
                n_frames: int = 6):
    """
    에피소드에서 균등 간격으로 n_frames 장을 matplotlib으로 표시.
    """
    result = record_episode(task, seed=seed, episode=episode, fps_skip=1)
    frames = result["frames"]
    if not frames:
        print("프레임 없음 — GPU 렌더링 확인 필요")
        return

    import matplotlib.pyplot as plt
    idxs = np.linspace(0, len(frames) - 1, n_frames, dtype=int)
    fig, axes = plt.subplots(1, n_frames, figsize=(3 * n_frames, 3))
    for ax, i in zip(axes, idxs):
        ax.imshow(frames[i])
        ax.set_title(f"step {i * 2}")
        ax.axis("off")
    ok = "✓ success" if result["success"] else f"✗ fail ({result['phase']})"
    fig.suptitle(f"{task}  ep={episode}  {ok}  ({result['n_steps']} steps)")
    plt.tight_layout()
    plt.show()


# ─── 내부 유틸 ────────────────────────────────────────────────

def _to_frame(rendered) -> np.ndarray | None:
    """env.render() 반환값 → (H, W, 3) uint8."""
    if rendered is None:
        return None
    if hasattr(rendered, "numpy"):
        rendered = rendered.numpy()
    arr = np.asarray(rendered)
    if arr.ndim == 4:       # (1, H, W, C)
        arr = arr[0]
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def _to_bool(val) -> bool:
    if hasattr(val, "item"):
        return bool(val.item())
    if hasattr(val, "__iter__"):
        return bool(np.any(val))
    return bool(val)


def save_gif(frames: list, path: str, fps: int = 15):
    """frames → GIF 저장. imageio 우선, 없으면 Pillow로 폴백."""
    try:
        import imageio
        imageio.mimsave(path, frames, fps=fps, loop=0)
    except ImportError:
        from PIL import Image
        duration = int(1000 / fps)
        imgs = [Image.fromarray(f) for f in frames]
        imgs[0].save(path, save_all=True, append_images=imgs[1:],
                     loop=0, duration=duration)


def _display_gif(path: str, label: str = ""):
    """Colab에서 GIF 인라인 표시."""
    try:
        from IPython.display import HTML, display
        import base64
        with open(path, "rb") as f:
            data = base64.b64encode(f.read()).decode()
        display(HTML(
            f"<p><b>{label}</b></p>"
            f'<img src="data:image/gif;base64,{data}" '
            f'style="max-width:480px; border:1px solid #ccc"/>'
        ))
    except Exception:
        print(f"  saved: {path}")


# ─── CLI 실행 ─────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--task",        default="PickCube-v1")
    p.add_argument("--n_episodes",  type=int, default=3)
    p.add_argument("--seed",        type=int, default=0)
    p.add_argument("--fps",         type=int, default=15)
    p.add_argument("--frames_only", action="store_true",
                   help="GIF 대신 n_frames 장을 matplotlib으로 표시")
    args = p.parse_args()

    if args.frames_only:
        show_frames(args.task, seed=args.seed, episode=0)
    else:
        record_and_show(args.task, n_episodes=args.n_episodes,
                        seed=args.seed, fps=args.fps)
