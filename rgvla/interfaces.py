from __future__ import annotations
from dataclasses import dataclass, field
from typing import Protocol, Optional
import numpy as np

ACTION_DIM = 7  # pd_ee_delta_pose: (dx, dy, dz, drx, dry, drz, gripper)


@dataclass
class StepRecord:
    seed: int
    episode: int
    step: int
    task: str
    action: np.ndarray           # (7,)
    proprio: np.ndarray          # (D_p,)
    progress: float              # distance change toward target (negative = closer)
    num_candidates: int
    # --- label 생성 전용 — detector 입력 금지 ---
    gt_target_dist: float
    # --- VLA hidden state (Medium-B 전용) ---
    hidden_state: Optional[np.ndarray] = None
    # --- 사후 채움 ---
    label: Optional[int] = None
    risk_score: Optional[float] = None
    reasoning_called: bool = False
    primitive: Optional[str] = None


@dataclass
class EpisodeMeta:
    seed: int
    episode: int
    task: str
    success: bool
    fail_step: Optional[int]     # 실패 시 t_f, 성공 시 None
    n_steps: int
    condition: str               # "baseline" | "always" | "risk_gated"
    tau: Optional[float] = None
    mean_latency_ms: float = 0.0


class BasePolicy(Protocol):
    """Frozen — 절대 수정/재학습 금지."""
    def reset(self, obs) -> None: ...
    def act(self, obs) -> np.ndarray: ...            # returns (7,)
    def last_hidden(self) -> Optional[np.ndarray]: ...  # Medium-A: None


class FeatureExtractor(Protocol):
    def reset(self) -> None: ...
    def push(self, rec: StepRecord) -> None: ...
    def extract(self, rec: StepRecord) -> np.ndarray: ...  # (feature_dim,)
    @property
    def feature_dim(self) -> int: ...


class RiskDetector(Protocol):
    def score(self, feat: np.ndarray) -> float: ...  # ∈ [0, 1]
