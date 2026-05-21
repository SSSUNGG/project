# Implementation Spec — Predictive Risk-Gated VLA Wrapper (Capstone Proposal 노선)

> **목적.** capstone_proposal.md에서 합의한 *학습된 risk detector + 실제 VLA + 제조 태스크* 노선을 Claude Code가 처음부터 끝까지 구현할 수 있는 자기완결적 명세로 풀어둔다.
>
> **이 문서를 적용하는 전제.** 학생은 이 노선을 1차로 시도하고, Phase 2 끝 (Week 8) 또는 Phase 3 중반 (Week 10–11)에 stop criteria에 따라 진행/피벗을 판단한다. 가벼운 rule-based 노선으로 피벗할 경우 별도 spec을 다시 만든다 — 두 문서를 섞지 않는다.

---

## 0. 한 줄 요약

frozen OpenVLA(또는 Octo) 정책 위에 *VLA 내부 hidden state·action history·proprioception을 입력으로 받는 학습된 MLP risk detector* 를 부착하여, `r_t ≥ τ`인 step에서만 VLM reasoner + fallback primitive를 호출하는 wrapper를 만든다. ManiSkill3의 contact-rich 제조 태스크 (PegInsertionSide, AssemblingKits) 에서 baseline / always-reasoning / risk-gated 세 조건을 동일 seed grid로 평가하여 success–latency Pareto frontier를 보고한다.

---

## 1. 연구 framing — Claude Code가 알아야 할 컨텍스트

### 1.1 차별화 포인트 (코드 설계 결정의 근거)

- **Predictive**, not reactive. detector는 *실패 발생 후*가 아니라 *실패 horizon H step 이내에 있는 상태*를 식별해야 한다. 이 가정이 horizon-based labeling, lead time at recall=0.9 지표, predictive feature 설계로 이어진다.
- **Sensor-free RGB-only.** force/tactile 입력 금지. proprioception은 로봇 자체 상태(gripper, ee pose, contact proxy)까지만 허용. ground-truth object pose는 *데이터 수집 시 label 생성용*으로만 쓰고, 추론 시 detector 입력에 들어가면 안 된다.
- **Manufacturing-first.** 평가 메시지의 중심은 PegInsertionSide·AssemblingKits. PickCube/StackCube는 sanity check, PickSingleYCB는 ablation.
- **Compute-aware.** 모든 latency는 동일 GPU(가능하면 Colab T4)에서 `torch.cuda.synchronize()` 이후 `time.perf_counter()`로 측정. 다른 환경 결과끼리 비교하지 않는다.

### 1.2 비교 조건

| 조건 ID | 이름 | 설명 | 역할 |
|---|---|---|---|
| `C1_baseline` | Baseline VLA | risk 판단 없이 VLA action 그대로 실행 | 좌하단 점 |
| `C2_always_reasoning` | Always-Reasoning | 매 step VLM reasoner + primitive 호출 | 우상단 점 (latency upper bound) |
| `C3_risk_gated` | Risk-Gated (ours) | `r_t ≥ τ`일 때만 reasoner 호출, τ sweep | 두 점 사이 곡선 |

C3 곡선이 C1–C2 직선보다 위쪽이면 Pareto 우위. 이 한 장이 메인 figure다.

---

## 2. 저장소 구조

```
vla-risk-wrapper/
├── configs/
│   ├── base.yaml
│   ├── env/
│   │   ├── peg_insertion.yaml
│   │   ├── assembling_kits.yaml
│   │   ├── pick_single_ycb.yaml
│   │   ├── pick_cube.yaml
│   │   └── stack_cube.yaml
│   ├── policy/
│   │   ├── openvla.yaml
│   │   └── octo.yaml
│   ├── detector/
│   │   ├── mlp_hybrid.yaml
│   │   ├── mlp_hidden_only.yaml
│   │   ├── mlp_action_only.yaml
│   │   └── mlp_proprio_only.yaml
│   ├── reasoner/
│   │   ├── qwen25_vl_3b.yaml
│   │   ├── llava_next_7b.yaml
│   │   └── rule_based.yaml
│   └── experiment/
│       ├── phase1_baseline.yaml
│       ├── phase2_collect_data.yaml
│       ├── phase2_train_detector.yaml
│       ├── phase3_wrapper_eval.yaml
│       └── phase4_ablation.yaml
├── src/
│   ├── env/
│   │   ├── __init__.py
│   │   ├── factory.py
│   │   ├── randomization.py
│   │   └── observation.py
│   ├── policy/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── openvla_policy.py
│   │   ├── octo_policy.py
│   │   └── lora.py
│   ├── detector/
│   │   ├── __init__.py
│   │   ├── features.py
│   │   ├── labeling.py
│   │   ├── dataset.py
│   │   ├── model.py
│   │   ├── train.py
│   │   └── calibrate.py
│   ├── reasoner/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── vlm_reasoner.py
│   │   ├── rule_based.py
│   │   └── prompts.py
│   ├── primitives/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── re_grasp.py
│   │   ├── re_approach.py
│   │   ├── align_then_insert.py
│   │   ├── request_help.py
│   │   └── continue_.py
│   ├── wrapper/
│   │   ├── __init__.py
│   │   └── risk_gated.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── schema.py
│   │   ├── io.py
│   │   └── collect.py
│   ├── eval/
│   │   ├── __init__.py
│   │   ├── runner.py
│   │   ├── metrics.py
│   │   ├── bootstrap.py
│   │   └── pareto.py
│   └── utils/
│       ├── __init__.py
│       ├── seeding.py
│       ├── logging.py
│       ├── checkpoint.py
│       └── colab.py
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
│   ├── test_wrapper.py
│   └── test_eval_reproducibility.py
├── data/                  # gitignore
├── checkpoints/           # gitignore
├── results/               # gitignore
├── requirements.txt
├── pyproject.toml
├── README.md
└── .gitignore
```

---

## 3. 환경과 의존성

### 3.1 Python 환경

- Python 3.10 (Colab default)
- CUDA 12.x, PyTorch 2.4+

### 3.2 `requirements.txt`

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
pydantic>=2.0
pandera>=0.20
pytest
```

### 3.3 Drive 경로 규약

`DRIVE_ROOT = /content/drive/MyDrive/vla-risk-wrapper` 를 환경변수로 export하고 모든 스크립트가 이 경로를 통해서만 I/O 수행:

```
DRIVE_ROOT/
├── data/          # 자가 수집 step parquet, episode meta parquet
├── checkpoints/   # detector .pt, LoRA adapter
├── results/       # phase별 평가 csv/parquet
└── figures/       # 최종 그림 pdf/png
```

---

## 4. 설정 파일 (Hydra)

### 4.1 `configs/base.yaml`

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

### 4.2 `configs/env/peg_insertion.yaml`

```yaml
id: PegInsertionSide-v1
obs_mode: rgb+depth+segmentation
control_mode: pd_ee_delta_pose
robot: panda
max_episode_steps: 100
horizon_H: 10            # detector horizon-based labeling 기준

randomization:
  strength: medium
  object_pose_xy_std: 0.04
  object_pose_yaw_std: 0.3
  lighting_jitter: 0.3
  distractor_count: 0
  camera_pose_jitter: 0.02

reward_mode: sparse
render_mode: null
```

태스크별 `horizon_H` 기본값:

| 태스크 | horizon_H | 이유 |
|---|---|---|
| PegInsertionSide | 10 | 짧은 contact-rich 단계 |
| AssemblingKits | 30 | 정렬·삽입 단계가 길다 |
| PickSingleYCB | 20 | 일반 길이 |
| PickCube, StackCube | 10 | 짧음 |

### 4.3 `configs/policy/openvla.yaml`

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
  pooling: mean        # mean | last_token | cls
  cache: true
```

### 4.4 `configs/detector/mlp_hybrid.yaml`

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

### 4.5 `configs/experiment/phase3_wrapper_eval.yaml`

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

## 5. 데이터 스키마

### 5.1 Per-step 레코드 (parquet)

| 컬럼 | dtype | shape | 설명 |
|---|---|---|---|
| `episode_id` | string (uuid4) | scalar | episode 식별자 |
| `task_id` | string | scalar | "peg_insertion" 등 |
| `seed` | int32 | scalar | env seed |
| `step` | int32 | scalar | 0-indexed |
| `vla_hidden` | float16 | (4096,) | VLA last layer hidden, mean-pooled |
| `action` | float32 | (7,) | 실행된 action |
| `action_history` | float32 | (K=8, 7) | 지난 K step action (padding 0) |
| `proprio` | float32 | (8,) | gripper, ee pose, contact proxy |
| `rgb_thumb` | uint8 | (64, 64, 3) | RGB 64×64 (옵션) |
| `info_is_grasped` | bool | scalar | ManiSkill info |
| `info_is_success` | bool | scalar | ManiSkill info |
| `reward` | float32 | scalar | |
| `done` | bool | scalar | |
| `truncated` | bool | scalar | |

`vla_hidden` float16 저장 → step당 8 KB. 360k step → ~3 GB. Drive 100 GB 내 여유.

### 5.2 Per-episode 메타 (parquet)

| 컬럼 | dtype | 설명 |
|---|---|---|
| `episode_id` | string | |
| `task_id` | string | |
| `seed` | int32 | |
| `policy_id` | string | "openvla_lora_v1" 등 |
| `randomization_strength` | string | easy / medium / hard |
| `success` | bool | terminal |
| `fail_step` | int32 | -1 if success |
| `episode_length` | int32 | |
| `wall_clock_seconds` | float32 | |
| `mean_latency_ms_per_step` | float32 | |
| `condition` | string | C1/C2/C3 (Phase 3+) |
| `tau` | float32 | C3만 |
| `reasoner_calls` | int32 | C2/C3만 |
| `primitive_history` | list[string] | C2/C3만 |
| `commit_sha` | string | 재현성 |
| `created_at` | timestamp | |

### 5.3 Detector 학습 sample

```python
@dataclass
class StepSample:
    features: np.ndarray   # (D,) concat된 입력
    label: int             # 0 or 1, horizon-based
    episode_id: str
    step: int
    task_id: str
    seed: int
```

### 5.4 Horizon-based labeling 규칙

- `success == True`: 모든 step `label = 0`
- `success == False`, `fail_step = t_f`: `step ∈ [max(0, t_f − H), t_f]` 구간 `label = 1`, 나머지 `label = 0`

H는 태스크별로 4.2의 표 값을 사용. H 자체를 ablation 변수로 둘 수 있도록 함수 시그니처는 `H`를 인자로 받는다.

### 5.5 Train/Val/Test 분할 — **seed 단위만**

```python
def make_splits(meta_df, ratios=(0.7, 0.15, 0.15), seed=42):
    """seed 단위 split. step/episode 단위 random split 금지 (leakage)."""
    rng = np.random.default_rng(seed)
    seeds = sorted(meta_df["seed"].unique())
    rng.shuffle(seeds)
    n = len(seeds)
    n_tr = int(n * ratios[0])
    n_va = int(n * ratios[1])
    return {"train": seeds[:n_tr], "val": seeds[n_tr:n_tr+n_va], "test": seeds[n_tr+n_va:]}
```

---

## 6. 모듈 인터페이스

### 6.1 `BasePolicy` (Module A)

```python
class BasePolicy(ABC):
    @abstractmethod
    def reset(self, instruction: str) -> None: ...

    @abstractmethod
    def predict(self, obs: dict) -> "PolicyOutput": ...

@dataclass
class PolicyOutput:
    action: np.ndarray              # (7,)
    hidden_state: np.ndarray        # (4096,) mean-pooled
    raw_logits: Optional[np.ndarray] = None
    latency_ms: float = 0.0
```

`OpenVLAPolicy` 구현 요구사항:

- `transformers.AutoModel.from_pretrained(openvla/openvla-7b)` + `peft.PeftModel`로 LoRA 적용
- 8-bit quantization은 `bitsandbytes`로
- forward hook으로 마지막 transformer block의 hidden state 캡처, mean-pool
- `torch.cuda.synchronize()` 후 `time.perf_counter()`로 latency 측정

### 6.2 `RiskDetector` (Module B)

```python
class RiskDetector(nn.Module):
    def __init__(self, cfg: DetectorConfig):
        super().__init__()
        self.vla_proj = nn.Linear(cfg.vla_hidden_dim, cfg.vla_proj_dim) if cfg.use_vla_hidden else None
        in_dim = (cfg.vla_proj_dim if cfg.use_vla_hidden else 0) \
               + (cfg.action_history_K * cfg.action_dim if cfg.use_action_history else 0) \
               + (cfg.proprio_dim if cfg.use_proprio else 0)
        layers = []
        for i in range(cfg.num_layers):
            out_dim = cfg.hidden_dim // (2 ** i) if i == cfg.num_layers - 1 else cfg.hidden_dim
            layers += [nn.Linear(in_dim if i == 0 else cfg.hidden_dim, out_dim),
                       nn.LayerNorm(out_dim),
                       nn.GELU(),
                       nn.Dropout(cfg.dropout)]
        layers += [nn.Linear(out_dim, 1)]
        self.head = nn.Sequential(*layers)
        self.temperature = nn.Parameter(torch.ones(1), requires_grad=False)

    def forward(self, vla_h, action_hist, proprio) -> torch.Tensor:
        feats = []
        if self.vla_proj is not None:
            feats.append(self.vla_proj(vla_h))
        if action_hist is not None:
            feats.append(action_hist.flatten(start_dim=1))
        if proprio is not None:
            feats.append(proprio)
        x = torch.cat(feats, dim=-1)
        return self.head(x).squeeze(-1)  # logits

    def predict_proba(self, vla_h, action_hist, proprio) -> torch.Tensor:
        logits = self.forward(vla_h, action_hist, proprio)
        return torch.sigmoid(logits / self.temperature)
```

### 6.3 `BaseReasoner` (Module C)

```python
class BaseReasoner(ABC):
    @abstractmethod
    def diagnose(
        self,
        rgb: np.ndarray,            # (H, W, 3)
        instruction: str,
        recent_actions: np.ndarray, # (K, 7)
        risk_score: float,
    ) -> "ReasonerOutput": ...

@dataclass
class ReasonerOutput:
    primitive_id: Literal[
        "re_grasp", "re_approach", "align_then_insert",
        "request_help", "continue"
    ]
    rationale: str
    latency_ms: float
```

`VLMReasoner`는 Qwen2.5-VL 3B 또는 LLaVA-NEXT 7B를 호출하되 출력 JSON-only 강제. 파싱 실패 시 무조건 `continue`로 fallback (이는 별도 카운터로 기록).

JSON 출력 예시 prompt (간략):

```
You are a manipulation failure analyzer. Given the current RGB image, instruction, and last 8 actions,
choose exactly ONE primitive from: re_grasp, re_approach, align_then_insert, request_help, continue.
Output strict JSON: {"primitive": "<id>", "rationale": "<one short sentence>"}.
```

### 6.4 `BasePrimitive`

```python
class BasePrimitive(ABC):
    @abstractmethod
    def step(self, obs: dict, state: "PrimitiveState") -> tuple[np.ndarray, "PrimitiveState", bool]:
        """Returns: (action, next_state, is_done). is_done=True면 종료, baseline 복귀."""
```

각 primitive는 *multi-step* 실행 가능. 예: `re_approach`는 5 step 후퇴 + 5 step 재진입. 실행 중에는 wrapper가 baseline VLA를 호출하지 않는다.

5개 primitive 동작 정의:

| primitive | 동작 | 예상 step 수 |
|---|---|---|
| `re_grasp` | gripper open → 5 cm 상승 → 재하강 → gripper close | 6–8 |
| `re_approach` | 10 cm 후퇴 → 재진입 시작 | 8–10 |
| `align_then_insert` | 시각 정렬(yaw 보정 ±5°) → 천천히 삽입 | 6–10 |
| `request_help` | gripper open → 안전 위치로 이동 → episode abort flag | 3–5 |
| `continue` | 즉시 종료, baseline action 1 step 실행 후 복귀 | 1 |

### 6.5 `RiskGatedWrapper`

```python
class RiskGatedWrapper:
    def __init__(
        self,
        policy: BasePolicy,
        detector: Optional[RiskDetector],
        reasoner: BaseReasoner,
        primitives: dict[str, BasePrimitive],
        tau: float,
        mode: Literal["none", "always", "risk_gated"],
    ): ...

    def reset(self, instruction: str) -> None: ...

    def step(self, obs: dict) -> "WrapperStepInfo": ...

@dataclass
class WrapperStepInfo:
    action: np.ndarray
    risk_score: float
    reasoner_called: bool
    primitive_active: Optional[str]
    latency_breakdown_ms: dict[str, float]
```

mode 동작:

- `none`: detector·reasoner 우회, baseline action 그대로
- `always`: detector 우회, 매 step reasoner 호출 (단 primitive 실행 중에는 보류)
- `risk_gated`: `r_t ≥ τ`이고 primitive 비활성일 때만 reasoner 호출

primitive 실행 중 락:

```python
if self._primitive_state is not None:
    action, self._primitive_state, done = self._active_primitive.step(obs, self._primitive_state)
    if done:
        self._active_primitive = None
        self._primitive_state = None
    return action  # reasoner 호출 안 함
```

---

## 7. 평가 러너 — Phase 3의 핵심

### 7.1 seed grid 불변식

```python
SEED_GRID = [(s, ep) for s in range(n_seeds) for ep in range(n_episodes_per_seed)]
```

세 condition × 같은 SEED_GRID로 돌린 결과만 비교. 같은 `(seed, ep)`에서 세 condition의 초기 obs는 **bit-identical**이어야 한다. 테스트 `test_eval_reproducibility.py`가 매 PR에서 이를 검증.

```python
def run_condition(
    condition_id: str,
    task_id: str,
    seed_grid: list[tuple[int, int]],
    wrapper: RiskGatedWrapper,
    tau: Optional[float] = None,
) -> "EvalResult":
    ...
```

### 7.2 latency 측정 규칙

매 step 다음 4 구간을 *분리*하여 측정:

```python
import time

torch.cuda.synchronize()
t0 = time.perf_counter()
policy_out = policy.predict(obs)
torch.cuda.synchronize()
t1 = time.perf_counter()
risk = detector.predict_proba(...)
torch.cuda.synchronize()
t2 = time.perf_counter()
if reasoner_called:
    reasoner_out = reasoner.diagnose(...)
    torch.cuda.synchronize()
t3 = time.perf_counter()
# primitive step
t4 = time.perf_counter()

latency_breakdown_ms = {
    "policy":    (t1 - t0) * 1000,
    "detector":  (t2 - t1) * 1000,
    "reasoner":  (t3 - t2) * 1000 if reasoner_called else 0.0,
    "primitive": (t4 - t3) * 1000,
}
```

mean wall-clock latency per step은 위 4개의 합의 평균.

---

## 8. 지표 함수

```python
# src/eval/metrics.py

def step_auroc(probs: np.ndarray, labels: np.ndarray) -> float: ...

def lead_time_at_recall(
    probs_by_traj: list[np.ndarray],
    fail_steps: list[int],
    target_recall: float = 0.9,
) -> tuple[float, float]:
    """Returns (tau_at_target_recall, mean_lead_time_steps)"""

def ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float: ...

def success_rate_with_ci(
    successes: np.ndarray, n_bootstrap: int = 1000
) -> tuple[float, float, float]:
    """Returns (mean, ci_low, ci_high)"""

def paired_bootstrap_diff(
    cond_a: np.ndarray, cond_b: np.ndarray, n_bootstrap: int = 1000
) -> tuple[float, float, float, float]:
    """Returns (mean_diff, ci_low, ci_high, p_value).
    cond_a, cond_b는 같은 (seed, ep) 인덱싱이어야 함."""

def pareto_frontier(points: list[tuple[float, float]]) -> list[int]:
    """Returns indices on frontier (min latency, max success)."""
```

### 8.1 보고 표 형식

Phase 2 detector 보고:

```
task            | AUROC          | lead_time@0.9 | ECE
peg_insertion   | 0.82 ± 0.02    | 14.3 step     | 0.07
assembling_kits | 0.78 ± 0.03    | 22.1 step     | 0.09
```

Phase 3 wrapper 보고:

```
task            | cond              | success         | latency/step | reasoner_calls/ep
peg_insertion   | C1                | 0.42 ± 0.04     |  32 ms       |   0
peg_insertion   | C2                | 0.61 ± 0.04     | 215 ms       |  78
peg_insertion   | C3 (τ=0.5)        | 0.58 ± 0.05     |  61 ms       |   9
```

### 8.2 메인 figure

x축: 평균 wall-clock latency per step (ms, log scale 옵션)
y축: success rate (95% CI 음영)

- C1, C2: 점 + errorbar
- C3: τ ∈ {0.2, ..., 0.8}을 점선으로 잇는 곡선

곡선이 C1–C2 직선 위에 있으면 Pareto 우위.

---

## 9. Phase별 실행 계획

### Phase 1 — 인프라 (Week 1–4)

목표: baseline VLA가 ManiSkill에서 동작, 평가 파이프라인 완성.

작업:

1. ManiSkill3 Colab quickstart 실행, 5개 환경 (PegInsertion, AssemblingKits, PickSingleYCB, PickCube, StackCube) 동작 검증
2. OpenVLA-7B 또는 Octo 로드 + LoRA adapt
3. 평가 스크립트 작성: seed × episode 매트릭스 CSV 로그
4. 5 seed × 100 episode baseline 수치

산출물: `results/phase1_baseline.parquet`, baseline 수치 표, Colab 노트북 1세트

**Stop criteria (Phase 1 끝):**

- ✓ 5개 환경 모두 1 episode 이상 정상 실행
- ✓ baseline 성공률이 PegInsertion 30–70%, AssemblingKits 20–60% 범위 (너무 높거나 너무 낮으면 randomization 조정)
- ✗ OpenVLA LoRA adapt에 4주 다 써도 baseline 성공률 < 10% → **Octo로 즉시 교체**, 그래도 안 되면 capstone 노선 포기 신호

### Phase 2 — 데이터 수집 + Detector 학습 (Week 5–8)

목표: AUROC ≥ 0.75, lead time ≥ 10 step 의 detector.

작업:

1. randomization 강화로 실패율 30–50%
2. `phase2_collect_data.py`로 step parquet + meta parquet 축적
3. horizon-based labeling
4. `RiskDetector` MLP 학습 + temperature scaling
5. val AUROC, lead time at recall=0.9, ECE 보고
6. seed 단위 70/15/15 분할

산출물: `data/step_*.parquet`, `data/meta_*.parquet`, `checkpoints/detector_*.pt`, `results/phase2_metrics.json`, ROC/lead-time/calibration plot

**Stop criteria (Phase 2 끝, Week 8):** — 첫 번째 피벗 판단 시점

- ✓ AUROC ≥ 0.75 (둘 중 하나라도 만족) AND lead time ≥ 10 step → 계속
- △ 0.70 ≤ AUROC < 0.75 → feature 추가 (RGB 64×64 thumb), focal loss, 1주 더 시도
- ✗ AUROC < 0.70 OR lead time < 5 step → **노선 재고**. 학생은 이 시점에서 implementation_spec(가벼운 rule-based) 노선으로 피벗할지 판단

### Phase 3 — Wrapper 통합 + 본실험 (Week 9–12)

목표: 3 condition Pareto curve 1장.

작업:

1. VLM reasoner 통합 (Qwen2.5-VL 3B 1차, 무거우면 rule-based reasoner로 임시)
2. 5개 primitive 구현
3. 동일 seed grid로 C1, C2, C3 × τ sweep 실행
4. Pareto figure, 95% bootstrap CI 표

산출물: `results/phase3_*.parquet`, 메인 figure, condition별 표

**Stop criteria (Phase 3 중반, Week 10–11):** — 두 번째 피벗 판단 시점

- ✓ C3 성공률이 C1 보다 +5%p 이상, latency가 C2 의 50% 이하 → 차별화 메시지 성립, Phase 4 진입
- △ C3 ≈ C1 또는 C3 ≈ C2 (Pareto 우위 없음) → τ 추가 sweep, primitive 디버깅 1주
- ✗ Wrapper가 항상 baseline보다 나쁨 → primitive 동작 오류 가능성. fallback selector를 oracle로 대체 (정답 primitive 알려주는 mock) 한 후에도 안 되면 노선 포기

### Phase 4 — Ablation, 보고서 (Week 13–16)

작업:

1. Detector ablation (hidden-only / action-only / proprio-only / hybrid)
2. Manufacturing vs general Δsuccess
3. Primitive library ablation (5 vs 3 vs single retry)
4. Horizon H sweep ({10, 20, 30})
5. τ fine sweep ({0.2, 0.3, ..., 0.8})
6. 보고서, 발표자료, 코드 정리

산출물: 졸업 보고서 (한/영), 발표자료, GitHub 저장소

---

## 10. 시드와 재현성

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

모든 스크립트 시작에서 `set_global_seed(cfg.seed)`. env reset은 별도로 `env.reset(seed=episode_seed)`. 매 episode meta에 `commit_sha` 기록. Phase 3 시작 직전 git tag `v1.0-main-eval`.

---

## 11. Colab 운영 규칙

### 11.1 세션 종료 대응

- 매 5 episode마다 결과 row를 Drive parquet에 append (`pyarrow.dataset` partitioned write)
- detector 학습 중 매 epoch마다 checkpoint Drive 저장
- Phase 3 평가 스크립트는 `--resume` 플래그로 SEED_GRID 중 미완료 항목만 처리

### 11.2 시간 예산

| 작업 | 예상 소요 | GPU |
|---|---|---|
| Phase 1 baseline (3 task × 5 seed × 100 ep) | ~6시간 | T4 |
| OpenVLA LoRA adapt (1회) | ~5–10시간 | A100 (Pro 필수) |
| Phase 2 데이터 수집 (3 task × 1500 ep) | ~12시간 | T4, 분할 |
| Detector 학습 30 epoch | ~1시간 | T4 |
| Phase 3 본실험 (3 task × 3 cond × 5 seed × 100 ep, C3는 τ 7개) | ~30–40시간 | T4 + A100 mix |
| Phase 4 ablation 전체 | ~15시간 | T4 |

Phase 3는 절대 하루에 못 끝남 → resume 구조 필수.

---

## 12. 테스트

```python
# tests/test_labeling.py
def test_horizon_labeling_success_episode():
    """success episode는 모든 label=0"""

def test_horizon_labeling_failure_episode():
    """fail_step=70, H=20 → step 50~70 label=1"""

def test_horizon_labeling_short_episode_fail_early():
    """fail_step=5, H=20 → step 0~5 label=1, 음수 인덱스 없음"""

# tests/test_detector.py
def test_detector_forward_shapes(): ...
def test_detector_overfits_small_batch():
    """8 sample × 100 epoch → train loss < 0.01"""
def test_detector_trivial_always_one_gives_auroc_05():
    """sanity: 모두 1 출력은 AUROC=0.5"""

# tests/test_wrapper.py
def test_wrapper_none_mode_never_calls_reasoner(): ...
def test_wrapper_always_mode_calls_every_step_except_primitive(): ...
def test_wrapper_risk_gated_respects_tau(): ...
def test_primitive_lock_blocks_new_reasoner_call(): ...

# tests/test_eval_reproducibility.py — 가장 중요한 테스트
def test_same_seed_grid_gives_bit_identical_initial_obs():
    """C1, C2, C3 wrapper로 같은 (seed, ep) reset 후 obs hash 동일"""
```

마지막 테스트가 통과하지 못하면 Phase 3 비교 결과 전체가 무효. PR8 통합 단계에서 반드시 추가.

---

## 13. 로깅 컨벤션

JSON Lines 파일 로깅:

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
  "latency_ms": 152.4,
  "commit_sha": "a3f2c1d"
}
```

이벤트 타입: `episode_start`, `episode_end`, `reasoner_call`, `primitive_start`, `primitive_end`, `policy_step`, `error`.

---

## 14. Day 0 체크리스트

Claude Code 첫 명령 *전*에 학생이 직접:

1. `nvidia-smi`로 Colab GPU 확인 (T4 / L4 / A100 중 무엇인지)
2. Drive 마운트, `DRIVE_ROOT` 하위 4 디렉토리 (data, checkpoints, results, figures) 생성
3. `pip install mani-skill && python -m mani_skill.examples.demo_random_action -e PickCube-v1` 로 렌더링 확인
4. Hugging Face 토큰 발급, `openvla/openvla-7b` 또는 `rail-berkeley/octo-base` 다운로드 권한 확인
5. 빈 git repo 생성, 위 디렉토리 구조 commit
6. `pytest -q` 실행 (지금은 다 fail, 수집 가능 상태인지만 확인)

---

## 15. Claude Code 작업 단위 (PR 분할)

> 한 번에 모든 모듈 짜라고 하지 말 것. 모듈 단위로 끊어서 PR처럼 처리.

1. **PR1**: 디렉토리 구조 + `requirements.txt` + `__init__.py` + `configs/base.yaml` + `configs/env/peg_insertion.yaml` + smoke test notebook
2. **PR2**: `src/env/{factory, randomization, observation}.py` + `tests/test_env.py`
3. **PR3**: `src/policy/{base, openvla_policy, octo_policy, lora}.py` + 추론 smoke test
4. **PR4**: `src/data/{schema, io, collect}.py` + `scripts/phase2_collect_data.py`
5. **PR5**: `src/detector/{features, labeling, dataset, model, train, calibrate}.py` + `scripts/phase2_train_detector.py` + 테스트
6. **PR6**: `src/primitives/*` + 단위 테스트
7. **PR7**: `src/reasoner/{base, rule_based, vlm_reasoner, prompts}.py`. rule_based 먼저로 wrapper 동작 검증 가능하게.
8. **PR8**: `src/wrapper/risk_gated.py` + `tests/test_eval_reproducibility.py` (bit-identical obs 검증)
9. **PR9**: `src/eval/{runner, metrics, bootstrap, pareto}.py` + `scripts/phase3_run_wrapper.py`
10. **PR10**: `scripts/make_figures.py` + 결과 plot

각 PR은 (a) 코드, (b) 테스트, (c) README 업데이트를 포함하도록 Claude Code에 명시.

---

## 16. 빠뜨리기 쉬운 항목

1. **seed leakage 검증** — PR8 통합 테스트에 obs hash 동일 검사 필수
2. **vla_hidden 캐싱** — Phase 3 step에서 baseline action 계산 시 hidden state 같이 캐시 → detector 호출 0 비용
3. **primitive 중첩 금지** — 실행 중 새 reasoner 호출 보류 락
4. **latency GPU sync** — `torch.cuda.synchronize()` 안 부르면 측정값 거짓말
5. **Drive write 빈도** — step마다 쓰지 말 것. memory buffer 후 episode 종료 시 batch
6. **reasoner 파싱 실패** — JSON 파싱 실패 시 무조건 `continue`, 실패 카운트 별도 로그
7. **commit_sha 미기록** — Phase 3 직전 git tag, 매 episode meta에 sha 포함

---

## 17. Day 0 직전 3가지 결정

학생이 Claude Code에 PR1 명령하기 전 다음 3개를 못박아 두면 망설임 없이 진행 가능:

### 17.1 베이스 정책

| 후보 | 장점 | 단점 | 권장 |
|---|---|---|---|
| OpenVLA-7B | 강력, 메인 framing에 정합 | LoRA에 A100 5–10시간, VRAM 24 GB+ | Pro+ 또는 외부 A100 있을 때 |
| **Octo-base** | 가벼움 (~90M param), Colab Free에서 fine-tune 가능, ManiSkill 어댑터 풍부 | OpenVLA보다 약함, framing에서 "VLA-style"로 약간 양보 | **1차 권장**. OpenVLA는 시간 남으면 2차 |

→ **권장: Octo로 시작, Phase 2 끝나서 시간 여유 있으면 OpenVLA로 교체**

### 17.2 Selective reasoner

| 후보 | 장점 | 단점 | 권장 |
|---|---|---|---|
| LLaVA-NEXT 7B | 강력 | latency ~300 ms, VRAM 14 GB+ | C2 latency upper bound가 너무 커짐 |
| **Qwen2.5-VL 3B** | 가벼움, latency ~150 ms | LLaVA-NEXT보다 약함 | **VLM 1차 권장** |
| Rule-based reasoner | 즉시 동작, latency 무시 | 본 framing에서 reasoner가 VLM이라는 가정 약화 | wrapper 동작 검증용으로만 |

→ **권장: PR7에서 rule-based 먼저, PR9 즈음에 Qwen2.5-VL 3B로 교체. LLaVA-NEXT는 ablation에서만**

### 17.3 데이터 수집 규모

| 옵션 | 규모 | 시간 |
|---|---|---|
| 풀 | 3 task × 1500 ep × ~80 step = 360k step | ~12시간 수집 |
| **축소** | **2 task (PegInsertion, AssemblingKits) × 1000 ep × ~80 step = 160k step** | **~6시간** |
| 최소 | 1 task × 500 ep | ~2시간 |

→ **권장: 축소부터 시작. Phase 2 stop criteria 통과하면 풀로 확장**

---

## 18. 본 노선 실패 판정 — Plan B 트리거

다음 중 하나라도 발생 시 학생은 implementation_spec(rule-based 가벼운 노선)으로 피벗을 진지하게 고려:

1. Phase 1 끝 (Week 4)에 OpenVLA·Octo 둘 다 baseline 성공률 < 10%
2. Phase 2 끝 (Week 8)에 detector AUROC < 0.70
3. Phase 3 중반 (Week 10–11)에 oracle primitive로도 wrapper 효과 없음
4. Colab Pro/Pro+ 자원이 phase 3 본실험 완료 전 소진

피벗 결정 시 본 spec은 폐기하고 implementation_spec 기반 새 spec을 다시 만든다. 두 spec을 섞지 않는다 — 한 노선의 결과만 보고서에 담는 것이 학부 캡스톤에서 훨씬 강한 메시지를 만든다.

---

## 19. 본 spec과 capstone_proposal.md의 정합성

본 문서는 capstone_proposal.md의 §6.1 시스템 아키텍처, §6.2 모듈 구성, §6.3 Risk Detector 상세, §7 Phase 계획, §8 평가 방법을 코드 단위로 풀어쓴 것이다. 두 문서가 충돌하면 본 spec(implementation)이 우선이고, 학생은 충돌 발견 시 capstone_proposal.md를 본 spec에 맞춰 갱신한다.
