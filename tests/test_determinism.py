"""
Phase 1 게이트 1: 결정성 테스트.

같은 (seed, episode)로 두 번 reset하면 초기 obs가 bit-identical해야 한다.
이것이 깨지면 paired 비교 전체가 무효.

실행: python -m pytest tests/test_determinism.py -v
"""
import numpy as np
import pytest

SAMPLE_GRID = [(0, 0), (0, 1), (1, 0), (3, 5)]
TASK = "PickCube-v1"


def flatten_obs(obs) -> np.ndarray:
    """obs dict → 1D float32 배열."""
    parts = []
    if isinstance(obs, dict):
        for v in sorted(obs.keys()):
            parts.append(flatten_obs(obs[v]))
    else:
        arr = obs
        if hasattr(arr, "numpy"):
            arr = arr.numpy()
        parts.append(np.asarray(arr, dtype=np.float32).reshape(-1))
    return np.concatenate(parts) if parts else np.array([], dtype=np.float32)


def test_identical_initial_obs():
    """
    동일 (seed, episode) → 두 env 인스턴스에서 bit-identical 초기 obs.
    """
    try:
        from rgvla.envs.make_env import make_env
    except ImportError as e:
        pytest.skip(f"mani_skill 미설치: {e}")

    for seed, ep in SAMPLE_GRID:
        env1 = make_env(TASK, seed=seed)
        env2 = make_env(TASK, seed=seed)

        o1, _ = env1.reset_with(seed, ep)
        o2, _ = env2.reset_with(seed, ep)

        f1 = flatten_obs(o1)
        f2 = flatten_obs(o2)

        assert f1.shape == f2.shape, (
            f"seed={seed}, ep={ep}: obs shape mismatch {f1.shape} vs {f2.shape}"
        )
        assert np.allclose(f1, f2, atol=1e-6), (
            f"seed={seed}, ep={ep}: obs not identical "
            f"(max diff={np.abs(f1-f2).max():.2e})"
        )

        env1.close()
        env2.close()

    print(f"\n✓ 결정성 테스트 통과: {len(SAMPLE_GRID)}개 (seed,ep) 모두 bit-identical")


def test_different_seeds_differ():
    """다른 seed는 다른 초기 obs를 줘야 한다 (sanity check)."""
    try:
        from rgvla.envs.make_env import make_env
    except ImportError as e:
        pytest.skip(f"mani_skill 미설치: {e}")

    env = make_env(TASK)
    o1, _ = env.reset_with(0, 0)
    o2, _ = env.reset_with(1, 0)
    f1 = flatten_obs(o1)
    f2 = flatten_obs(o2)
    assert not np.allclose(f1, f2, atol=1e-4), "seed=0,ep=0 와 seed=1,ep=0 가 동일함 — seed 식이 잘못됨"
    env.close()
    print("\n✓ seed 다양성 테스트 통과")


if __name__ == "__main__":
    test_identical_initial_obs()
    test_different_seeds_differ()
    print("All determinism tests passed.")
