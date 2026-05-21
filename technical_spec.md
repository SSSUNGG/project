# 기술 스펙 — VLA Risk-Aware Selective Recovery Wrapper

본 문서는 Claude Code로 구현을 시작하기 전 합의해 두는 *구현 명세*다. 제안서의 연구 설계를 코드 단위로 분해하고, 모듈 경계·데이터 스키마·함수 시그니처·설정 파일 포맷을 박아 둔다. 모든 항목은 Colab Free/Pro에서 동작하는 것을 전제로 한다.

---

## 0. 한 줄 요약

frozen OpenVLA(또는 Octo) 정책 위에 학습된 risk score `r_t`가 임계값 τ를 넘는 step에서만 selective reasoner + fallback primitive를 호출하는 wrapper를, ManiSkill3 contact-rich 제조 태스크에서 baseline / always-reasoning / risk-gated 세 조건으로 평가한다.

---

## 1. 저장소 구조

```
vla-risk-wrapper/
├── configs/
│   ├── base.yaml                # 전역 default
│   ├── env/
│   │   ├── peg_insertion.yaml
│   │   ├── assembling_kits.yaml
│   │   ├── pick_single_ycb.yaml
│   │   └── pick_cube.yaml
│   ├── policy/
│   │   ├── openvla.yaml
│   │   └── octo.yaml
│   ├── detector/
│   │   ├── mlp_hybrid.yaml
│   │   ├── mlp_hidden_only.yaml
│   │   ├── mlp_proprio_only.yaml
│   │   └── mlp_rgb_only.yaml
│   ├── reasoner/
│   │   ├── qwen25_vl_3b.yaml
│   │   ├── llava_next_7b.yaml
│   │   └── rule_based.yaml
│   └── experiment/
│       ├── phase1_baseline.yaml
│       ├── phase2_detector_train.yaml
│       ├── phase3_wrapper_eval.yaml
│       └── phase4_ablation.yaml
├── src/
│   ├── env/                     # ManiSkill 어댑터 + 평가 wrapper
│   │   ├── __init__.py
│   │   ├── factory.py           # make_env(cfg) -> gym.Env
│   │   ├── randomization.py     # 강한 randomization 옵션
│   │   └── observation.py       # obs dict 정규화
│   ├── policy/                  # Module A
│   │   ├── __init__.py
│   │   ├── base.py              # BasePolicy interface
│   │   ├── openvla_policy.py
│   │   ├── octo_policy.py
│   │   └── lora.py              # LoRA adapter helper
│   ├── detector/                # Module B
│   │   ├── __init__.py
│   │   ├── features.py          # build_features(step_dict) -> Tensor
│   │   ├── model.py             # RiskDetector MLP
│   │   ├── dataset.py           # StepDataset + DataLoader
│   │   ├── labeling.py          # horizon-based label 생성
│   │   ├── train.py             # 학습 루프
│   │   └── calibrate.py         # temperature scaling
│   ├── reasoner/                # Module C
│   │   ├── __init__.py
│   │   ├── base.py              # BaseReasoner interface
│   │   ├── vlm_reasoner.py      # LLaVA / Qwen2.5-VL
│   │   ├── rule_based.py
│   │   └── prompts.py
│   ├── primitives/              # Fallback 액션 라이브러리
│   │   ├── __init__.py
│   │   ├── base.py              # BasePrimitive interface
│   │   ├── re_grasp.py
│   │   ├── re_approach.py
│   │   ├── align_then_insert.py
│   │   ├── request_help.py
│   │   └── continue_.py
│   ├── wrapper/
│   │   ├── __init__.py
│   │   └── risk_gated.py        # RiskGatedWrapper end-to-end
│   ├── data/
│   │   ├── __init__.py
│   │   ├── collect.py           # baseline rollout -> step dataset
│   │   ├── schema.py            # pandera/pydantic schemas
│   │   └── io.py                # parquet/hdf5 read/write
│   ├── eval/
│   │   ├── __init__.py
│   │   ├── runner.py            # condition × seed × episode runner
│   │   ├── metrics.py           # AUROC, lead time, ECE, latency
│   │   ├── bootstrap.py         # 95% CI
│   │   └── pareto.py            # Pareto frontier plot helper
│   └── utils/
│       ├── __init__.py
│       ├── seeding.py
│       ├── logging.py           # 구조화 로깅 + Drive sync
│       ├── checkpoint.py        # resume helper
│       └── colab.py             # session keep-alive
├── scripts/
│   ├── phase1_run_baseline.py
│   ├── phase2_collect_data.py
│   ├── phase2_train_detector.py
│   ├── phase3_run_wrapper.py
│   ├── phase4_ablations.py
│   └── make_figures.py
├── notebooks/
│   ├── 00_env_smoke_test.ipynb
│   ├── 01_baseline_eval.ipynb
│   ├── 02_detector_train.ipynb
│   ├── 03_wrapper_eval.ipynb
│   └── 04_figures.ipynb
├── tests/
│   ├── test_env.py
│   ├── test_features.py
│   ├── test_labeling.py
│   ├── test_detector.py
│   ├── test_primitives.py
│   └── test_wrapper.py
├── data/                        # 자가 수집 데이터 (gitignore)
├── checkpoints/                 # 학습된 모델 (gitignore)
├── results/                     # CSV/JSON 로그 (gitignore)
├── requirements.txt
├── pyproject.toml
├── README.md
└── .gitignore
```

---

## 2. 환경과 의존성

### 2.1 Python 환경

- Python 3.10 (Colab default)
- CUDA 12.x, PyTorch 2.4+

### 2.2 핵심 의존성 (`requirements.txt`)

```
torch>=2.4.0
torchvision
mani-skill>=3.0.0
gymnasium>=0.29
transformers>=4.45.0
peft>=0.13.0
bitsandbytes>=0.43.0
accelerate>=0.34.0
hydra-core>=1.3.0
omegaconf>=2.3.0
pandas
pyarrow
numpy
scikit-learn
scipy
matplotlib
seaborn
tqdm
rich
wandb         # optional, 기본은 csv 로그
pydantic>=2.0
pandera>=0.20
pytest
```

### 2.3 Colab Drive 마운트 규약

```
/content/drive/MyDrive/vla-risk-wrapper/
├── data/         # 자가 수집 데이터셋 (parquet)
├── checkpoints/  # detector, LoRA adapter
├── results/      # 평가 CSV
└── figures/      # 최종 그림
```

학생의 Colab 노트북은 항상 이 경로를 `DRIVE_ROOT` 환경변수로 export한 뒤 `src/utils/logging.py`를 통해 사용한다.

---

## 3. 설정 파일 (Hydra)

### 3.1 `configs/base.yaml`

```yaml
defaults:
  - env: peg_insertion
  - policy: openvla
  - detector: mlp_hybrid
  - reasoner: qwen25_vl_3b
  - experiment: phase1_baseline
  - _self_

project_root: /content/drive/MyDrive/vla-risk-wrapper
seed: 42
device: cuda

logging:
  level: INFO
  console: true
  file: true
  drive_sync_every_n_episodes: 5

hardware:
  precision: bf16
  use_8bit: false
```

### 3.2 `configs/env/peg_insertion.yaml`

```yaml
id: PegInsertionSide-v1
obs_mode: rgb+depth+segmentation
control_mode: pd_ee_delta_pose
robot: panda
max_episode_steps: 100
horizon_H: 20            # detector horizon-based labeling 기준

randomization:
  strength: medium       # easy | medium | hard
  object_pose_xy_std: 0.04
  object_pose_yaw_std: 0.3
  lighting_jitter: 0.3
  distractor_count: 0
  camera_pose_jitter: 0.02

reward_mode: sparse
render_mode: null
```

태스크별 `horizon_H`는 PegInsertion=10, AssemblingKits=30, PickSingleYCB=20, PickCube=10으로 디폴트.

### 3.3 `configs/policy/openvla.yaml`

```yaml
name: openvla
hf_id: openvla/openvla-7b
action_space: ee_delta_pose_7d
freeze_backbone: true

lora:
  enabled: true
  r: 16
  alpha: 32
  dropout: 0.05
  target_modules: [q_proj, v_proj, o_proj]

quantization:
  enabled: true
  bits: 8

hidden_state:
  layer: -1
  pooling: mean       # mean | last_token | cls
  cache: true
```

### 3.4 `configs/detector/mlp_hybrid.yaml`

```yaml
name: mlp_hybrid

input:
  use_vla_hidden: true
  vla_hidden_dim: 4096
  vla_proj_dim: 256

  use_action_history: true
  action_history_K: 8
  action_dim: 7

  use_proprio: true
  proprio_dim: 8

  use_rgb_features: false
  rgb_feature_dim: 0

model:
  hidden_dim: 256
  num_layers: 3
  dropout: 0.1
  norm: layernorm
  activation: gelu

train:
  optimizer: adamw
  lr: 3.0e-4
  weight_decay: 1.0e-2
  batch_size: 256
  epochs: 30
  early_stopping_patience: 5
  loss: weighted_bce
  pos_weight: 4.0
  grad_clip: 1.0

calibration:
  method: temperature_scaling
  val_split_for_calib: 0.5
```

### 3.5 `configs/experiment/phase3_wrapper_eval.yaml`

```yaml
name: phase3_wrapper_eval

conditions:
  - id: C1_baseline
    wrapper: none
  - id: C2_always_reasoning
    wrapper: always
  - id: C3_risk_gated
    wrapper: risk_gated
    tau_sweep: [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

tasks:
  - peg_insertion
  - assembling_kits
  - pick_single_ycb

eval:
  n_seeds: 5
  n_episodes_per_seed: 100
  same_seed_across_conditions: true   # CRITICAL
  bootstrap_n: 1000
  confidence_level: 0.95

logging:
  per_step: true
  per_episode: true
```

---

## 4. 데이터 스키마

### 4.1 Per-step 레코드 (parquet)

자가 수집 시 매 step 다음 컬럼을 기록한다. **타입을 못박는 게 핵심**이다 — 학기 후반에 dtype 불일치로 모든 결과가 날아가는 사고 방지.

| 컬럼 | dtype | shape | 설명 |
|---|---|---|---|
| `episode_id` | string (uuid4) | scalar | episode 식별자 |
| `task_id` | string | scalar | "peg_insertion" 등 |
| `seed` | int32 | scalar | env seed |
| `step` | int32 | scalar | 0-indexed step |
| `vla_hidden` | float16 | (4096,) | VLA last layer hidden, mean-pooled |
| `action` | float32 | (7,) | 실행된 action |
| `action_history` | float32 | (K=8, 7) | 지난 K step의 action (padding 0) |
| `proprio` | float32 | (8,) | gripper, ee pose, contact proxy |
| `rgb_thumb` | uint8 | (64, 64, 3) | RGB 64×64 downsample (옵션) |
| `info_is_grasped` | bool | scalar | ManiSkill info dict |
| `info_is_success` | bool | scalar | ManiSkill info dict |
| `reward` | float32 | scalar | 환경 reward |
| `done` | bool | scalar | terminal flag |
| `truncated` | bool | scalar | timeout flag |

`vla_hidden`은 float16으로 저장하면 step당 8 KB. 360k step → ~3 GB. Drive 100 GB 안에서 여유 있음.

### 4.2 Per-episode 메타 레코드 (parquet)

| 컬럼 | dtype | 설명 |
|---|---|---|
| `episode_id` | string | |
| `task_id` | string | |
| `seed` | int32 | |
| `policy_id` | string | "openvla_lora_v1" 등 |
| `randomization_strength` | string | easy/medium/hard |
| `success` | bool | terminal 시점 |
| `fail_step` | int32 | -1 if success, else terminal step |
| `episode_length` | int32 | |
| `wall_clock_seconds` | float32 | |
| `mean_latency_ms_per_step` | float32 | |
| `condition` | string | C1 / C2 / C3 (Phase 3+ only) |
| `tau` | float32 | risk threshold (C3 only) |
| `reasoner_calls` | int32 | (C2/C3 only) |
| `primitive_history` | list[string] | 발동된 primitive 목록 (C2/C3 only) |
| `commit_sha` | string | 재현성 |
| `created_at` | timestamp | |

### 4.3 Detector 학습 레이블 (in-memory 변환)

`src/detector/labeling.py`가 per-step 레코드와 per-episode 메타를 join하여 다음을 생성:

```python
@dataclass
class StepSample:
    features: np.ndarray   # (D,) — concat된 입력
    label: int             # 0 or 1, horizon-based
    episode_id: str
    step: int
    task_id: str
    seed: int
```

규칙:

- `success == True` 인 episode: 모든 step `label = 0`
- `success == False`이고 `fail_step = t_f`: `step ∈ [t_f - H, t_f]` 구간 `label = 1`, 나머지 `label = 0`

### 4.4 Train/Val/Test 분할 — **seed 단위만**

`src/data/io.py:make_splits(meta_df, ratios=(0.7, 0.15, 0.15), seed=42)` 함수가 *seed 단위로* 분할한다. step 또는 episode 단위 random split은 **금지**. 잘못 split하면 detector AUROC가 비현실적으로 높게 나옴 (data leakage).

```python
def make_splits(meta_df, ratios=(0.7, 0.15, 0.15), seed=42):
    rng = np.random.default_rng(seed)
    seeds = sorted(meta_df["seed"].unique())
    rng.shuffle(seeds)
    n = len(seeds)
    n_tr = int(n * ratios[0])
    n_va = int(n * ratios[1])
    return {
        "train": seeds[:n_tr],
        "val":   seeds[n_tr:n_tr+n_va],
        "test":  seeds[n_tr+n_va:],
    }
```

---

## 5. 모듈 인터페이스

### 5.1 `BasePolicy` (Module A)

```python
class BasePolicy(ABC):
    @abstractmethod
    def reset(self, instruction: str) -> None: ...

    @abstractmethod
    def predict(self, obs: dict) -> PolicyOutput: ...

@dataclass
class PolicyOutput:
    action: np.ndarray              # (7,)
    hidden_state: np.ndarray        # (4096,) mean-pooled
    raw_logits: Optional[np.ndarray] = None
    latency_ms: float = 0.0
```

`OpenVLAPolicy`는 hf transformers + peft로 LoRA 적용, forward hook으로 hidden state 캡처.

### 5.2 `RiskDetector` (Module B)

```python
class RiskDetector(nn.Module):
    def __init__(self, cfg: DetectorConfig): ...

    def forward(
        self,
        vla_h: torch.Tensor,        # (B, 4096)
        action_hist: torch.Tensor,  # (B, K*7)
        proprio: torch.Tensor,      # (B, 8)
    ) -> torch.Tensor:              # (B,) logits
        ...

    def predict_proba(self, ...) -> torch.Tensor:  # calibrated 확률
        ...
```

calibrated 확률은 학습 후 `calibrate.py`에서 fit된 temperature `T`를 적용: `sigmoid(logit / T)`.

### 5.3 `BaseReasoner` (Module C)

```python
class BaseReasoner(ABC):
    @abstractmethod
    def diagnose(
        self,
        rgb: np.ndarray,            # (H, W, 3)
        instruction: str,
        recent_actions: np.ndarray, # (K, 7)
        risk_score: float,
    ) -> ReasonerOutput: ...

@dataclass
class ReasonerOutput:
    primitive_id: Literal[
        "re_grasp", "re_approach", "align_then_insert",
        "request_help", "continue"
    ]
    rationale: str
    latency_ms: float
```

VLM 출력은 JSON-only로 prompting하여 5개 primitive 중 하나의 id로 강제 파싱. 파싱 실패 시 `continue`로 fallback.

### 5.4 `BasePrimitive`

```python
class BasePrimitive(ABC):
    @abstractmethod
    def step(self, obs: dict, state: PrimitiveState) -> tuple[np.ndarray, PrimitiveState, bool]:
        """
        Returns: (action, next_state, is_done)
        is_done=True면 primitive 종료, baseline policy로 복귀
        """
        ...
```

primitive는 *multi-step* 실행 가능. 예: `re_approach`는 후퇴 5 step + 재진입 시작 — 그 동안 wrapper는 baseline VLA를 호출하지 않는다.

### 5.5 `RiskGatedWrapper`

```python
class RiskGatedWrapper:
    def __init__(
        self,
        policy: BasePolicy,
        detector: RiskDetector,
        reasoner: BaseReasoner,
        primitives: dict[str, BasePrimitive],
        tau: float,
        mode: Literal["none", "always", "risk_gated"],
    ): ...

    def reset(self, instruction: str) -> None: ...

    def step(self, obs: dict) -> WrapperStepInfo: ...

@dataclass
class WrapperStepInfo:
    action: np.ndarray
    risk_score: float
    reasoner_called: bool
    primitive_active: Optional[str]
    latency_breakdown_ms: dict[str, float]  # {"policy": 30, "detector": 0.5, "reasoner": 150, "primitive": 2}
```

`mode="none"` → 항상 baseline action
`mode="always"` → 매 step reasoner 호출
`mode="risk_gated"` → `r_t ≥ τ`일 때만 reasoner 호출

primitive가 실행 중이면 새 reasoner 호출 보류 (primitive 우선).

---

## 6. 평가 러너 — Phase 3의 핵심

```python
def run_condition(
    condition_id: str,
    task_id: str,
    n_seeds: int,
    n_episodes_per_seed: int,
    wrapper: RiskGatedWrapper,
    tau: Optional[float] = None,
) -> EvalResult:
    ...
```

**가장 중요한 불변식**: 같은 `(seed, episode_index)` 쌍은 모든 condition에서 *완전히 동일한 env reset*을 만들어야 한다. `gym.Env.reset(seed=...)`을 명시적으로 호출하고, `numpy.random` 및 `torch.manual_seed`도 같이 고정. paired comparison이 통계 검정의 근거가 된다.

```python
SEED_GRID = [(s, ep) for s in range(n_seeds) for ep in range(n_episodes_per_seed)]
```

세 condition × 같은 SEED_GRID로 돌린 결과만 비교한다.

---

## 7. 지표 함수

```python
# src/eval/metrics.py

def step_auroc(probs, labels) -> float: ...

def lead_time_at_recall(
    probs_by_traj: list[np.ndarray],
    fail_steps: list[int],
    target_recall: float = 0.9,
) -> tuple[float, float]:
    """Returns (tau_at_target_recall, mean_lead_time_steps)"""

def ece(probs, labels, n_bins: int = 10) -> float: ...

def success_rate_with_ci(successes: np.ndarray, n_bootstrap: int = 1000) -> tuple[float, float, float]:
    """Returns (mean, ci_low, ci_high)"""

def paired_bootstrap_diff(
    cond_a: np.ndarray,
    cond_b: np.ndarray,
    n_bootstrap: int = 1000,
) -> tuple[float, float, float, float]:
    """Returns (mean_diff, ci_low, ci_high, p_value)"""

def pareto_frontier(points: list[tuple[float, float]]) -> list[int]:
    """Returns indices of points on the frontier (min latency, max success)"""
```

---

## 8. 실험 조건과 산출물 매핑

| Phase | 스크립트 | 입력 | 산출물 |
|---|---|---|---|
| 1 | `phase1_run_baseline.py` | env, policy config | baseline meta parquet, csv 표 |
| 2-수집 | `phase2_collect_data.py` | env hard randomization | step parquet (~3 GB) + meta parquet |
| 2-학습 | `phase2_train_detector.py` | detector config + 데이터 | checkpoint, val_metrics.json, calibration.json |
| 3 | `phase3_run_wrapper.py` | wrapper config, τ 리스트 | per-condition meta parquet, latency 로그 |
| 4 | `phase4_ablations.py` | ablation matrix | ablation_table.csv |
| Fig | `make_figures.py` | results/ 전체 | figures/ pdf/png |

각 스크립트는 `python scripts/phase1_run_baseline.py --config-name=phase1_baseline` 형태로 Hydra override 가능.

---

## 9. 시드와 재현성

전역 seeding 헬퍼:

```python
# src/utils/seeding.py
def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

모든 스크립트 시작에서 `set_global_seed(cfg.seed)`. env reset은 별도로 `env.reset(seed=episode_seed)`.

매 episode 메타에 `commit_sha`를 기록 — Phase 3 시작 직전 git tag `v1.0-main-eval`을 박고, 이후 변경된 코드로 재현 시 즉시 인식.

---

## 10. Colab 운영 규칙

### 10.1 세션 종료 대응

- 매 5 episode마다 결과 row를 Drive의 parquet에 append (`pyarrow.dataset` partitioned write)
- detector 학습 중에는 매 epoch마다 checkpoint를 Drive에 저장
- Phase 3 평가 스크립트는 `--resume` 플래그로 SEED_GRID 중 미완료 항목만 처리

### 10.2 시간 예산

| 작업 | 예상 소요 | GPU |
|---|---|---|
| Phase 1 baseline 1 task × 5 seed × 100 ep | ~2시간 | T4 |
| OpenVLA LoRA adapt | ~5–10시간 | A100 (Pro 필수) |
| Phase 2 데이터 수집 (3 task × 1500 ep) | ~12시간 | T4, 분할 실행 |
| Detector 학습 30 epoch | ~1시간 | T4 |
| Phase 3 본실험 (3 task × 3 cond × 5 seed × 100 ep) | ~15–20시간 | T4 + A100 mix |
| Phase 4 ablation 전체 | ~10시간 | T4 |

Phase 3는 하루에 다 못 끝남 → 반드시 resume 구조.

---

## 11. 테스트와 CI

`pytest`로 다음 최소 테스트를 유지:

```python
# tests/test_labeling.py
def test_horizon_labeling_success_episode():
    """success episode는 모든 label=0"""

def test_horizon_labeling_failure_episode():
    """fail_step=70, H=20 → step 50~70 label=1, 나머지 0"""

def test_horizon_labeling_short_episode_fail_early():
    """fail_step=5, H=20 → step 0~5 label=1, 음수 인덱스 없음"""

# tests/test_detector.py
def test_detector_forward_shapes(): ...
def test_detector_overfits_small_batch():
    """8 sample에 100 epoch 돌려 train loss < 0.01"""

# tests/test_wrapper.py
def test_wrapper_none_mode_never_calls_reasoner(): ...
def test_wrapper_always_mode_calls_every_step(): ...
def test_wrapper_risk_gated_respects_tau(): ...

# tests/test_eval.py
def test_same_seed_grid_gives_same_initial_obs():
    """C1, C2, C3가 같은 (seed, ep)에서 reset 후 obs 동일"""
```

마지막 테스트가 *재현성의 기둥*이다. 통과하지 못하면 Phase 3 결과 비교가 무효.

---

## 12. 로깅 컨벤션

구조화 로깅 (`structlog` 또는 `logging` + JSONFormatter). 매 event는 다음 키를 포함:

```json
{
  "ts": "2025-09-15T03:14:22.391Z",
  "phase": 3,
  "condition": "C3_risk_gated",
  "task": "peg_insertion",
  "seed": 7,
  "episode": 42,
  "step": 18,
  "event": "reasoner_call",
  "risk_score": 0.71,
  "primitive": "re_approach",
  "latency_ms": 152.4
}
```

콘솔에는 사람이 읽을 수 있는 포맷, 파일에는 JSON Lines.

---

## 13. 실험을 *처음* 굴리기 전 체크리스트 (Day 0)

Claude Code에 첫 작업을 시키기 전 학생이 직접 확인:

1. `nvidia-smi`로 Colab GPU 종류 확인 (T4 또는 L4 또는 A100)
2. Drive 마운트, `DRIVE_ROOT` 디렉토리 4개 (data, checkpoints, results, figures) 생성
3. `mani-skill` 설치 후 `python -m mani_skill.examples.demo_random_action -e PickCube-v1` 한 번 굴려서 렌더링 확인
4. `openvla/openvla-7b` 또는 대체 체크포인트의 hf 인증 토큰 발급
5. 빈 git repo 생성, 위 디렉토리 구조 commit
6. `pytest -q` 실행하여 (지금은 모두 fail이지만) 수집 가능한 상태인지 확인

---

## 14. Claude Code에 작업 지시할 단위 — 권장 순서

> 한 번에 모든 모듈을 짜라고 하지 말 것. 모듈 단위로 끊어서 PR처럼 처리.

1. **PR1**: 디렉토리 구조 + `requirements.txt` + 빈 `__init__.py` + `configs/base.yaml` + `configs/env/peg_insertion.yaml` + smoke test 노트북
2. **PR2**: `src/env/factory.py`, `randomization.py`, `observation.py` + `tests/test_env.py`
3. **PR3**: `src/policy/base.py`, `openvla_policy.py` (LoRA 적용 포함) + 짧은 inference smoke test
4. **PR4**: `src/data/schema.py`, `io.py`, `collect.py` + `phase2_collect_data.py` 스크립트
5. **PR5**: `src/detector/{features, labeling, dataset, model, train, calibrate}.py` + 학습 스크립트 + 테스트
6. **PR6**: `src/primitives/*` + 단위 테스트
7. **PR7**: `src/reasoner/{base, rule_based, vlm_reasoner, prompts}.py` (먼저 rule_based로 wrapper 동작 검증)
8. **PR8**: `src/wrapper/risk_gated.py` + 통합 테스트 (same-seed-same-obs)
9. **PR9**: `src/eval/{runner, metrics, bootstrap, pareto}.py` + `phase3_run_wrapper.py`
10. **PR10**: `scripts/make_figures.py` + 결과 plotting

각 PR은 (a) 코드, (b) 테스트, (c) 짧은 README 업데이트를 포함하도록 Claude Code에 지시.

---

## 15. 빠뜨리기 쉬운 마지막 7가지

1. `seed leakage 검증`을 PR8 통합 테스트에서 반드시 추가 — 같은 (seed, ep)에서 세 condition의 초기 obs가 bit-identical인지 hash로 검사
2. `vla_hidden` 캐싱 — Phase 3에서 baseline action 계산 시 hidden state도 같이 캐시하면 detector 호출이 무료가 됨
3. `primitive 중첩 금지` — primitive 실행 중에는 새 reasoner 호출을 보류하는 락 필요
4. `latency 측정 위치 통일` — `time.perf_counter()`로 GPU sync 후 측정 (`torch.cuda.synchronize()` 필수)
5. `Drive write 빈도` — 매 step write는 I/O 폭증. 메모리 버퍼 후 episode 종료 시 batch write
6. `reasoner 출력 파싱 실패` — JSON 파싱 실패 시 무조건 `continue` primitive로 fallback, 실패 카운트는 별도 로그
7. `commit_sha 미기록` — Phase 3 시작 전 git tag, 매 episode 메타에 commit_sha — 한 번이라도 빠지면 재현 불가

---

## 16. 이후 의사결정이 필요한 항목

본 스펙을 그대로 가지고 가도 되지만, 학생이 Claude Code에 작업 지시하기 전에 다음 3개는 한 번 더 결정해 두면 좋다:

- **베이스 정책**: OpenVLA-7B (강력하지만 무거움) vs Octo-base (가볍고 fine-tune 쉬움). Colab 자원 확실치 않으면 **Octo 1차, OpenVLA 2차**가 안전.
- **Selective reasoner**: Qwen2.5-VL 3B (가벼움, latency 짧음) vs LLaVA-NEXT 7B (강력, 무거움). **rule_based로 먼저 wrapper 동작 검증 후 VLM 통합**을 권장.
- **데이터 수집 규모**: 3 task × 1500 ep는 욕심. 시간이 빠듯하면 메인 2 task × 1000 ep로 줄이고 PickSingleYCB는 Phase 4 ablation에만 사용.
