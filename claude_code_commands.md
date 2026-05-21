# Claude Code 명령 문구 모음

> 본 문서는 `capstone_implementation_spec.md`를 spec 파일로 두고 Claude Code에 PR 단위로 작업을 지시하기 위한 *복붙용* 문구 모음이다. 학생은 PR1부터 순서대로, 한 번에 하나씩 사용한다.

---

## 사용 방법 (Day 0)

1. 로컬 또는 Colab에서 빈 git 저장소 생성:
   ```bash
   mkdir vla-risk-wrapper && cd vla-risk-wrapper
   git init
   ```

2. 두 파일을 저장소 루트에 복사:
   - `capstone_implementation_spec.md` — 풀스펙
   - `CLAUDE.md` — 아래 §1 내용을 그대로 저장
   - 두 파일 다 `git add . && git commit -m "init: spec and project context"` 로 커밋

3. Claude Code 실행:
   ```bash
   claude
   ```
   또는 VS Code / JetBrains / Slack 통합 사용. Claude Code가 자동으로 `CLAUDE.md`를 컨텍스트로 읽는다.

4. 매 PR마다 아래 §3 ~ §12의 명령을 *복붙*. 한 PR이 끝나서 학생이 검토·머지한 후 다음 PR로 넘어간다.

5. 각 PR이 끝나면 학생이 직접:
   - `pytest -q` 실행
   - 새 파일 커밋 (`git add . && git commit`)
   - PR 번호와 결과를 메모

---

## 1. `CLAUDE.md` 내용 (프로젝트 루트에 저장)

다음 내용을 그대로 `CLAUDE.md` 파일로 저장한다. Claude Code는 매 세션마다 이 파일을 자동으로 읽는다.

```markdown
# 프로젝트 컨텍스트: VLA Risk-Aware Selective Recovery Wrapper

## 한 줄 요약
ManiSkill3 contact-rich 제조 태스크에서, frozen VLA 정책(Octo 또는 OpenVLA) 위에 학습된 risk detector를 부착하여 `r_t ≥ τ`인 step에서만 VLM reasoner + fallback primitive를 호출하는 wrapper를 만든다. baseline / always-reasoning / risk-gated 세 조건을 동일 seed grid로 평가하여 success–latency Pareto frontier를 보고한다.

## 가장 중요한 spec 문서
`capstone_implementation_spec.md` — 모든 모듈의 인터페이스, 데이터 스키마, 설정 파일 포맷, PR 분할 순서가 박혀 있다. 본 문서와 spec이 충돌하면 spec이 우선.

## 절대 잊지 말 것 (top-level invariants)
1. **두 노선을 섞지 않는다.** 본 프로젝트는 *학습된 ML detector + 실제 VLA* 노선이다. rule-based 노선이나 simulator state 기반 detector를 임의로 추가하지 말 것. spec과 충돌 시 학생에게 질문.
2. **Seed 단위 split만 사용.** detector 학습 데이터 train/val/test 분할은 반드시 seed 단위. step/episode 단위 random split은 leakage를 만든다.
3. **Latency 측정 전에 `torch.cuda.synchronize()`.** GPU 비동기 실행 때문에 sync 안 하면 측정값이 거짓말이 된다.
4. **세 condition은 같은 seed grid에서만 비교.** Phase 3에서 C1/C2/C3가 같은 `(seed, episode_index)`에서 bit-identical 초기 obs를 받아야 한다. 이걸 못 지키면 비교 결과 전체가 무효.
5. **Drive write는 batch.** step마다 Drive에 쓰지 말고, episode 종료 시 buffer flush.

## 작업 스타일
- 한 번에 한 PR만 처리한다. 학생이 명시한 PR 범위를 넘어서 작업하지 말 것.
- 매 PR은 (a) 코드, (b) 테스트, (c) README/CHANGELOG 업데이트를 포함해야 한다.
- 새 파일을 만들 때는 spec의 §2 디렉토리 구조를 따른다. 임의로 새 디렉토리를 만들지 말 것.
- 함수 시그니처는 spec의 §6 모듈 인터페이스를 그대로 따른다.
- 외부 라이브러리를 새로 추가할 일이 있으면 `requirements.txt`를 함께 업데이트하고 학생에게 알린다.
- 설정값(threshold, learning rate 등)을 코드에 하드코딩하지 말 것. 모두 `configs/*.yaml`에서 읽어온다.

## Colab 환경 가정
- Python 3.10, PyTorch 2.4+, CUDA 12.x
- T4 (15 GB) 또는 L4 또는 A100 (40 GB) — 셋 다 동작해야 한다
- Drive 마운트 경로: `/content/drive/MyDrive/vla-risk-wrapper`
- 세션 종료 가능성을 항상 가정. checkpoint·resume 구조 필수.

## 베이스 정책 결정
1차로 **Octo-base**를 사용한다. OpenVLA-7B는 Phase 2 stop criteria 통과 후 시간이 남으면 교체한다. PR3 시점에 학생이 별도 지시 없으면 Octo 우선.

## Selective reasoner 결정
PR7에서 **rule-based reasoner**를 먼저 구현하여 wrapper 동작을 검증한다. PR9 즈음에 **Qwen2.5-VL 3B**로 교체한다. LLaVA-NEXT 7B는 ablation 용도로만 남긴다.

## 학생에게 질문해야 할 시점
다음 상황에서는 작업을 멈추고 학생에게 질문할 것:
- spec과 명확히 충돌하는 결정이 필요할 때
- 새로운 외부 데이터셋이나 모델 체크포인트가 필요할 때
- 한 PR이 spec의 시간 예산을 2배 이상 초과할 때
- 테스트가 통과하지 않는 원인이 spec 자체의 설계 결함인 것 같을 때
```

---

## 2. Day 0 직전 확인 명령 (선택)

학생이 직접 환경을 확인했다면 건너뛰어도 됨. 자신 없으면 첫 명령으로 던져볼 것:

```
@capstone_implementation_spec.md 의 §14 Day 0 체크리스트를 확인하는 짧은 bash 스크립트
scripts/check_environment.sh 를 만들어줘.

요구사항:
- nvidia-smi 출력
- Python 버전 확인 (3.10+ 권장)
- ManiSkill 설치 여부 (import mani_skill 시도)
- Drive 마운트 경로 존재 여부
- Hugging Face 토큰 환경변수(HF_TOKEN) 설정 여부

각 항목을 ✓/✗로 표시하고, 마지막에 모두 ✓이면 "Day 0 ready"를 출력해줘.
파일은 만들기만 하고 실행은 하지 마. 학생이 직접 실행할 거야.
```

---

## 3. PR1 명령 — 디렉토리 구조 + 설정 + smoke test

복사해서 그대로 던질 것:

```
@capstone_implementation_spec.md 의 §2와 §4를 참조해서 PR1을 수행해줘.

# PR1 범위
다음을 만든다. spec에 명시된 것만 만들고, 그 외 파일은 만들지 마.

## 1. 디렉토리 구조
spec §2의 트리 구조 그대로. 각 디렉토리에 빈 __init__.py 추가.
다음 디렉토리는 .gitkeep만 두고 비워둔다: data/, checkpoints/, results/, notebooks/

## 2. 의존성 파일
- requirements.txt: spec §3.2 그대로
- pyproject.toml: 최소한의 setuptools 설정, project name="vla-risk-wrapper", python_requires=">=3.10"
- .gitignore: __pycache__, *.pyc, .ipynb_checkpoints, data/, checkpoints/, results/, .env, *.log

## 3. 설정 파일 (Hydra)
spec §4를 그대로 따라:
- configs/base.yaml (§4.1)
- configs/env/peg_insertion.yaml (§4.2)
- configs/env/assembling_kits.yaml (peg_insertion 기반, id=AssemblingKits-v1, horizon_H=30)
- configs/env/pick_single_ycb.yaml (id=PickSingleYCB-v1, horizon_H=20)
- configs/env/pick_cube.yaml (id=PickCube-v1, horizon_H=10)
- configs/env/stack_cube.yaml (id=StackCube-v1, horizon_H=10)
- configs/policy/octo.yaml (hf_id=rail-berkeley/octo-base, LoRA disabled by default — 가벼운 모델이므로)
- configs/policy/openvla.yaml (§4.3)
- configs/detector/mlp_hybrid.yaml (§4.4)
- configs/detector/mlp_hidden_only.yaml (use_action_history=false, use_proprio=false)
- configs/detector/mlp_action_only.yaml (use_vla_hidden=false, use_proprio=false)
- configs/detector/mlp_proprio_only.yaml (use_vla_hidden=false, use_action_history=false)
- configs/reasoner/rule_based.yaml (name=rule_based, 별다른 파라미터 없음)
- configs/reasoner/qwen25_vl_3b.yaml (hf_id=Qwen/Qwen2.5-VL-3B-Instruct, max_new_tokens=128)
- configs/reasoner/llava_next_7b.yaml (hf_id=llava-hf/llava-v1.6-mistral-7b-hf, max_new_tokens=128)
- configs/experiment/phase1_baseline.yaml, phase2_collect_data.yaml, phase2_train_detector.yaml, phase3_wrapper_eval.yaml (§4.5), phase4_ablation.yaml

## 4. Smoke test notebook
notebooks/00_env_smoke_test.ipynb 를 만든다. 다음 셀을 포함:
1. Drive 마운트 (Colab일 때만)
2. `import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))`
3. `import mani_skill; import gymnasium as gym; env = gym.make("PickCube-v1"); obs, _ = env.reset(seed=0); print(obs.keys()); env.close()`
4. `from omegaconf import OmegaConf; cfg = OmegaConf.load("configs/base.yaml"); print(OmegaConf.to_yaml(cfg))`
5. 마지막 셀: "Smoke test passed" 출력

## 5. README.md
프로젝트 설명 (한국어 OK), 디렉토리 구조 요약, Day 0 체크리스트 링크, PR1 완료 표시.

## 6. CHANGELOG.md
새 파일 생성. "## PR1 (YYYY-MM-DD): scaffolding" 섹션 추가.

# Acceptance criteria
- `pytest --collect-only` 가 에러 없이 통과 (테스트가 없어도 됨)
- `python -c "from omegaconf import OmegaConf; OmegaConf.load('configs/base.yaml')"` 가 에러 없이 통과
- 디렉토리 구조가 spec §2와 정확히 일치
- notebook 4번 셀의 yaml 출력이 base + env + policy + detector + reasoner + experiment 모두 포함

# 하지 말 것
- ManiSkill 실제 호출 코드는 notebook에만, src/ 안에는 넣지 마.
- src/env/, src/policy/ 등 안의 모듈 구현은 PR2부터. PR1에서는 __init__.py만.
- 새 외부 의존성 추가하지 마.

작업 시작.
```

---

## 4. PR2 명령 — Env 어댑터

PR1이 머지된 후 사용:

```
@capstone_implementation_spec.md 의 §2, §4.2, §5.1 (per-step 레코드의 proprio 정의) 을 참조해서 PR2를 수행해줘.

# PR2 범위
src/env/ 모듈 구현 + 테스트.

## 1. src/env/factory.py
```python
def make_env(cfg: DictConfig, seed: int) -> gym.Env: ...
```
- ManiSkill3 환경을 만들고 반환
- obs_mode, control_mode, robot 등은 cfg.env에서 읽음
- cfg.env.randomization 설정을 적용 (PR2에서는 hook만 — 실제 randomization 로직은 randomization.py에서)
- seed로 reset 후 반환하지 말고, env 객체만 반환. 학생이 명시적으로 reset 호출.

## 2. src/env/randomization.py
```python
def apply_randomization(env, cfg_random: DictConfig) -> None: ...
```
- object pose, lighting, distractor, camera pose jitter를 적용
- ManiSkill의 reconfigure 또는 env.unwrapped 접근 사용
- 강도가 hard / medium / easy면 미리 정의된 std 값으로 매핑
- 구현이 ManiSkill API상 불가능한 부분은 TODO 주석과 함께 no-op으로 두고 학생에게 보고

## 3. src/env/observation.py
```python
def normalize_obs(obs: dict, env) -> dict: ...
def extract_proprio(obs: dict, env) -> np.ndarray:  # shape (8,)
    """gripper_open(1) + ee_pose(6) + contact_proxy(1)"""
```
- ManiSkill obs dict를 일관된 포맷으로 정규화
- proprio는 spec §5.1의 정의를 따름

## 4. tests/test_env.py
- test_make_env_returns_gym_env: PickCube-v1로 env 생성 후 obs key 검증
- test_seed_reproducibility: 같은 seed로 두 번 reset하면 obs가 bit-identical
- test_proprio_shape: extract_proprio 출력이 (8,) float32

# Acceptance criteria
- pytest tests/test_env.py 가 모두 통과
- PickCube-v1, PegInsertionSide-v1 두 환경에서 1 episode 랜덤 액션 실행 가능 (수동 확인용 짧은 스크립트 scripts/_smoke_env.py 추가 OK)

# 하지 말 것
- Policy, detector, wrapper 코드 작성 금지
- 실제 random 적용 로직이 ManiSkill API 한계로 불가능하면 학생에게 보고

작업 시작.
```

---

## 5. PR3 명령 — Policy (Octo)

```
@capstone_implementation_spec.md 의 §6.1, §4.3 (policy config) 을 참조해서 PR3을 수행해줘.

# PR3 범위
src/policy/ 모듈 구현. Octo 우선, OpenVLA는 인터페이스만.

## 1. src/policy/base.py
spec §6.1의 BasePolicy ABC와 PolicyOutput dataclass 그대로 구현.

## 2. src/policy/octo_policy.py
```python
class OctoPolicy(BasePolicy): ...
```
- rail-berkeley/octo-base 체크포인트 로드
- predict(obs) 호출 시:
  - VLA action 생성
  - 마지막 transformer block의 hidden state를 forward hook으로 캡처
  - mean-pool (또는 cfg에 따라 last_token / cls)
  - PolicyOutput(action, hidden_state, latency_ms) 반환
- latency 측정 시 torch.cuda.synchronize() 전후 호출
- cfg에서 freeze_backbone=true이면 requires_grad=False

## 3. src/policy/openvla_policy.py
- 클래스 이름과 시그니처만 구현
- 본문은 NotImplementedError("OpenVLA는 Phase 2 stop criteria 통과 후 활성화. 그때까지 OctoPolicy를 사용.") raise
- Hf id, LoRA config 등은 cfg에서 읽어두기만 함 (실제 호출 안 함)

## 4. src/policy/lora.py
- apply_lora(model, cfg) 헬퍼. peft.LoraConfig로 wrapping.
- Octo가 PEFT와 호환되지 않으면 그 사실을 docstring에 명시하고 빈 함수로 둠.

## 5. 테스트
tests/test_policy.py:
- test_octo_loads: 모델 로드가 OOM 없이 됨 (skip 가능하게 @pytest.mark.slow)
- test_octo_predict_shapes: PolicyOutput.action shape (7,), hidden_state shape (D,)
- test_octo_hidden_state_is_captured: hidden_state가 모두 0이 아님
- test_openvla_raises_not_implemented

# Acceptance criteria
- Octo가 Colab T4에서 OOM 없이 로드되고 1 step 추론 가능
- pytest -m "not slow" tests/test_policy.py 가 통과
- PolicyOutput.latency_ms > 0

# 하지 말 것
- OpenVLA 실제 구현 금지 (이번 PR 범위 밖)
- Octo fine-tune 코드 금지 (frozen으로 사용)

작업 시작.
```

---

## 6. PR4 명령 — 데이터 수집

```
@capstone_implementation_spec.md 의 §5 데이터 스키마 전체와 §6.1 (PolicyOutput) 을 참조해서 PR4를 수행해줘.

# PR4 범위
src/data/ 모듈과 phase2 데이터 수집 스크립트.

## 1. src/data/schema.py
pydantic 모델 또는 pandera schema로 spec §5.1, §5.2의 컬럼을 정의.
StepRecord, EpisodeMeta 두 모델.

## 2. src/data/io.py
- write_step_batch(records: list[StepRecord], path: str): parquet으로 append write (pyarrow.dataset)
- write_episode_meta(meta: EpisodeMeta, path: str)
- read_steps(path: str, filters: dict | None) -> pd.DataFrame
- read_meta(path: str) -> pd.DataFrame
- make_splits(meta_df, ratios, seed): spec §5.5 그대로 (seed 단위 분할 — step/episode random split 금지)

## 3. src/data/collect.py
```python
def collect_episode(env, policy: BasePolicy, instruction: str, seed: int, cfg: DictConfig) -> tuple[list[StepRecord], EpisodeMeta]:
    ...
```
- env reset(seed=seed), reset history buffer
- 매 step:
  - obs를 받아서 policy.predict(obs) 호출
  - StepRecord 생성 (vla_hidden은 float16으로 캐스팅)
  - action_history 버퍼 업데이트 (padding 0)
  - env.step
- terminal에서 fail_step 계산 (success이면 -1, else terminal step)
- horizon-based label은 여기서 만들지 말 것 — labeling은 detector 학습 시점에 한다 (PR5)

## 4. scripts/phase2_collect_data.py
- Hydra 진입점, --config-name=phase2_collect_data
- task / seed / n_episodes 루프 돌면서 collect_episode 호출
- 매 5 episode마다 Drive에 parquet flush (drive_sync_every_n_episodes=5)
- 중단됐다 재시작할 수 있도록 --resume 플래그 (이미 처리한 episode_id는 건너뜀)

## 5. 테스트
tests/test_data.py:
- test_step_record_schema: 잘못된 dtype은 reject
- test_parquet_round_trip: write → read 후 동일
- test_make_splits_no_seed_overlap: train/val/test의 seed 교집합이 빈 집합
- test_collect_episode_returns_correct_step_count: 10 step만 가는 dummy env로 검증

## 6. 작은 dry-run 스크립트
scripts/_dryrun_collect.py: PickCube-v1에서 3 episode 수집해서 parquet 생성. 사람이 눈으로 확인용.

# Acceptance criteria
- pytest tests/test_data.py 통과
- _dryrun_collect.py 실행 후 data/ 디렉토리에 step_*.parquet, meta_*.parquet 생성
- parquet 파일을 pandas로 읽어서 vla_hidden 컬럼이 (4096,) float16인지 확인 가능

# 하지 말 것
- 실제 detector 학습 코드 작성 금지 (PR5)
- 매 step Drive write 금지 (반드시 buffer 후 batch)

작업 시작.
```

---

## 7. PR5 명령 — Risk Detector

```
@capstone_implementation_spec.md 의 §5.3 (StepSample), §5.4 (horizon labeling), §6.2 (RiskDetector), §4.4 (detector config) 를 참조해서 PR5를 수행해줘.

# PR5 범위
src/detector/ 모듈 전체와 학습 스크립트.

## 1. src/detector/features.py
```python
def build_features(step_record: StepRecord, cfg: DetectorConfig) -> np.ndarray: ...
```
- cfg.use_vla_hidden, use_action_history, use_proprio 플래그에 따라 입력 벡터를 concat
- VLA hidden은 float16 → float32 캐스팅
- action_history는 (K*7,) 로 flatten
- 출력 shape는 cfg에서 계산된 in_dim과 일치해야 함

## 2. src/detector/labeling.py
spec §5.4의 규칙을 정확히 구현. 함수 시그니처:
```python
def horizon_label(success: bool, fail_step: int, episode_length: int, H: int) -> np.ndarray:
    """Returns label array of shape (episode_length,) with values in {0, 1}."""
```
- success=True: 전부 0
- success=False, fail_step=t_f: [max(0, t_f-H), t_f] 구간 1, 나머지 0

## 3. src/detector/dataset.py
- StepDataset(torch.utils.data.Dataset): step parquet과 meta parquet을 join, horizon_label 적용
- 인덱싱 시 StepSample (spec §5.3) 반환
- DataLoader 헬퍼: make_dataloader(split_seeds, cfg) → DataLoader

## 4. src/detector/model.py
spec §6.2의 RiskDetector nn.Module 정확히 구현.
predict_proba는 temperature scaling 적용.

## 5. src/detector/train.py
```python
def train_detector(cfg: DictConfig) -> Path: ...
```
- weighted BCE loss (pos_weight from cfg)
- AdamW, early stopping on val AUROC
- 매 epoch 끝에 val AUROC, loss를 콘솔과 results/phase2_metrics.json에 기록
- best checkpoint를 checkpoints/detector_{task}_{name}.pt에 저장

## 6. src/detector/calibrate.py
- val set의 절반으로 temperature scaling
- T를 model.temperature buffer에 저장 후 checkpoint 다시 저장
- 학습 후 자동 호출

## 7. scripts/phase2_train_detector.py
- Hydra 진입점, --config-name=phase2_train_detector
- task별로 train_detector + calibrate 호출
- 출력 metric: AUROC ± bootstrap CI, lead_time at recall=0.9, ECE — 세 개 모두 results/phase2_metrics.json에 저장

## 8. 테스트
tests/test_labeling.py:
- test_horizon_labeling_success_episode (success → 모두 0)
- test_horizon_labeling_failure_episode (fail_step=70, H=20 → step 50~70 label=1)
- test_horizon_labeling_short_episode_fail_early (fail_step=5, H=20 → step 0~5 label=1, 음수 인덱스 안 생김)

tests/test_detector.py:
- test_detector_forward_shapes
- test_detector_overfits_small_batch (8 sample 100 epoch → train loss < 0.01)
- test_detector_trivial_always_one_gives_auroc_05 (sanity)
- test_temperature_scaling_reduces_ece

# Acceptance criteria
- 모든 테스트 통과
- PR4의 _dryrun_collect 데이터로 train_detector를 5 epoch 돌려서 loss가 감소하는지 확인
- val AUROC 출력이 0.5 ~ 1.0 사이의 합리적 숫자

# 하지 말 것
- Wrapper, reasoner, primitive 코드 금지
- 학습 데이터 크기 너무 작아서 AUROC 못 측정되는 경우, 그 사실을 알리고 작은 synthetic 데이터로 단위 테스트만 통과시켜.

작업 시작.
```

---

## 8. PR6 명령 — Fallback Primitives

```
@capstone_implementation_spec.md 의 §6.4 (BasePrimitive, 5개 primitive 정의) 를 참조해서 PR6을 수행해줘.

# PR6 범위
src/primitives/ 모듈과 단위 테스트.

## 1. src/primitives/base.py
spec §6.4의 BasePrimitive ABC, PrimitiveState dataclass.

## 2. 5개 primitive 구현 (각각 별도 파일)
각 primitive는 다음 시그니처:
```python
class ReGrasp(BasePrimitive):
    def step(self, obs: dict, state: PrimitiveState) -> tuple[np.ndarray, PrimitiveState, bool]:
        ...
```

primitive별 동작 (spec §6.4 표):
- re_grasp.py: gripper open → 5cm 상승 → 재하강 → gripper close (6-8 step)
- re_approach.py: 10cm 후퇴 → 재진입 시작 (8-10 step)
- align_then_insert.py: yaw 보정 ±5° → 천천히 삽입 (6-10 step)
- request_help.py: gripper open → 안전 위치 이동 → abort flag (3-5 step)
- continue_.py: 1 step만 baseline 그대로 → done=True

action space는 7-DoF EE delta pose (pd_ee_delta_pose) 기준. PrimitiveState로 현재 phase, step count, target offset 등 추적.

## 3. tests/test_primitives.py
각 primitive에 대해:
- test_X_terminates_within_max_steps: 정해진 step 안에 done=True
- test_X_action_shape: 모든 step에서 (7,) float32 반환
- test_continue_terminates_in_one_step
- test_request_help_sets_abort_in_state

## 4. notebooks/05_primitives_smoke.ipynb (옵션)
PegInsertionSide-v1 환경에서 각 primitive를 격리 실행해서 시각적으로 동작 확인.

# Acceptance criteria
- pytest tests/test_primitives.py 통과
- 모든 primitive가 max 15 step 이내에 done=True

# 하지 못 할 것
- VLA policy 호출 금지 (primitive는 baseline VLA와 독립)
- ManiSkill API 변경 금지

작업 시작.
```

---

## 9. PR7 명령 — Reasoner (rule-based 먼저)

```
@capstone_implementation_spec.md 의 §6.3 (BaseReasoner, JSON 출력 강제), §4 reasoner configs 참조. PR7 수행.

# PR7 범위
src/reasoner/ 모듈. rule-based 먼저, VLM은 인터페이스만.

## 1. src/reasoner/base.py
spec §6.3의 BaseReasoner ABC, ReasonerOutput dataclass.

## 2. src/reasoner/rule_based.py
RuleBasedReasoner:
- risk_score 기반의 간단한 if-else 결정 트리
- ee_pose의 z좌표가 낮고 grasp 실패 신호면 re_grasp
- recent_actions의 progress가 작으면 re_approach
- risk_score가 매우 높으면 (>0.85) request_help
- 그 외 continue
- 이건 wrapper end-to-end 동작 검증용. 실제 본 실험에서는 VLM으로 대체.
- 본문 로직은 짧고 명시적으로. cfg 파라미터로 임계값 조정.

## 3. src/reasoner/vlm_reasoner.py
VLMReasoner:
- cfg.hf_id로 모델 로드 (Qwen2.5-VL 3B 우선)
- diagnose 호출 시:
  - prompts.py의 SYSTEM_PROMPT + USER_TEMPLATE 사용
  - JSON-only 강제 (output에서 첫 JSON 객체 추출)
  - 파싱 실패 시 ReasonerOutput("continue", "parse_fail", latency_ms)로 fallback. 실패 카운트는 별도 로깅.
- latency 측정 시 torch.cuda.synchronize()

## 4. src/reasoner/prompts.py
SYSTEM_PROMPT: spec §6.3의 example prompt 그대로 + 강한 JSON-only 강제 문구.
USER_TEMPLATE: f"Instruction: {instruction}\nLast actions: {actions}\nRisk: {risk:.2f}\n"

## 5. 테스트
tests/test_reasoner.py:
- test_rule_based_returns_valid_primitive_id
- test_vlm_parse_fail_falls_back_to_continue (mock으로 invalid JSON 반환)
- test_vlm_output_is_in_allowed_set

# Acceptance criteria
- rule-based reasoner가 임의의 risk_score에서 합법적인 primitive_id 반환
- VLM reasoner는 model load skip (@pytest.mark.slow)했을 때 mock으로 동작 검증

# 하지 말 것
- Qwen2.5-VL 실제 로드를 통과 조건으로 강제하지 마. Colab Free에서는 부담.
- LLaVA-NEXT는 클래스 placeholder만 두고 NotImplementedError.

작업 시작.
```

---

## 10. PR8 명령 — Wrapper 통합 + 재현성 테스트

```
@capstone_implementation_spec.md 의 §6.5 (RiskGatedWrapper), §7.1 (seed grid 불변식), §12 (테스트) 를 참조해서 PR8을 수행해줘.

# PR8 범위
이 PR이 본 프로젝트의 가장 중요한 PR이다. 재현성 테스트를 통과하지 못하면 Phase 3 결과 전체가 무효.

## 1. src/wrapper/risk_gated.py
spec §6.5의 RiskGatedWrapper와 WrapperStepInfo 정확히 구현.

핵심 동작:
- mode="none": baseline VLA action만
- mode="always": 매 step reasoner 호출 (단 primitive 실행 중에는 보류)
- mode="risk_gated": detector(r_t) ≥ tau일 때만 reasoner 호출 (primitive 실행 중에는 보류)
- primitive lock: self._active_primitive와 self._primitive_state로 관리. primitive 실행 중 새 reasoner 호출 차단.
- latency_breakdown_ms: spec §7.2의 4 구간 정확히 측정 (torch.cuda.synchronize 필수)

## 2. tests/test_wrapper.py
- test_wrapper_none_mode_never_calls_reasoner
- test_wrapper_always_mode_calls_every_step_except_primitive
- test_wrapper_risk_gated_respects_tau (mock detector로 r_t를 0.3 / 0.7로 조절)
- test_primitive_lock_blocks_new_reasoner_call
- test_latency_breakdown_has_all_four_keys

## 3. tests/test_eval_reproducibility.py (★ 가장 중요)
- test_same_seed_grid_gives_bit_identical_initial_obs:
  - 세 wrapper (mode=none, always, risk_gated)를 만들고
  - 같은 (seed=42, ep_idx=0)으로 env reset
  - 세 wrapper의 첫 obs를 hashlib.sha256으로 hash해서 모두 동일한지 검증
- test_random_state_does_not_leak_between_conditions:
  - C1 wrapper로 1 episode 굴린 후 C2 wrapper를 reset해도 영향 없음

# Acceptance criteria
- 모든 wrapper 테스트 통과
- 재현성 테스트가 *확실히* 통과 — 통과 못 하면 즉시 학생에게 보고
- PickCube-v1에서 3개 mode 각각 1 episode 굴려서 결과 dict이 합리적인지 (success bool, latency > 0)

# 하지 말 것
- Reasoner나 detector를 임의로 교체하지 마. 학생이 mock 객체를 주입할 수 있는 구조여야 함 (의존성 주입).
- 재현성 테스트를 통과시키기 위해 무리한 hack 금지. 통과 못 하면 그 사실을 보고.

작업 시작.
```

---

## 11. PR9 명령 — Evaluation runner

```
@capstone_implementation_spec.md 의 §7 평가 러너, §8 지표 함수를 참조해서 PR9를 수행해줘.

# PR9 범위
src/eval/ 모듈 전체와 phase3 실행 스크립트.

## 1. src/eval/runner.py
```python
def run_condition(condition_id, task_id, seed_grid, wrapper, tau=None) -> EvalResult: ...
def run_phase3(cfg) -> dict[str, EvalResult]: ...
```
- SEED_GRID = [(s, ep) for s in range(n_seeds) for ep in range(n_episodes_per_seed)]
- 세 condition × 같은 seed_grid 강제
- 매 episode 종료 시 EpisodeMeta 저장 (Drive parquet append)
- --resume 플래그로 미완료 (seed, ep) 만 처리

## 2. src/eval/metrics.py
spec §8의 5개 함수 정확히:
- step_auroc
- lead_time_at_recall (Returns (tau_at_recall, mean_lead_time))
- ece
- success_rate_with_ci (bootstrap)
- paired_bootstrap_diff (같은 seed grid 가정)
- pareto_frontier

## 3. src/eval/bootstrap.py
- bootstrap_mean_ci 헬퍼
- B=1000, confidence_level=0.95 기본값

## 4. src/eval/pareto.py
- aggregate_for_pareto: condition별 (mean_latency_ms_per_step, mean_success_rate, ci) 추출
- mark_frontier_points

## 5. scripts/phase3_run_wrapper.py
- Hydra 진입점, --config-name=phase3_wrapper_eval
- 3 task × 3 condition (C3는 tau_sweep 7개)
- Drive append 저장, resume 지원
- 종료 시 results/phase3_summary.csv 출력

## 6. 테스트
tests/test_eval.py:
- test_paired_bootstrap_same_input_gives_zero_diff
- test_pareto_frontier_simple_case
- test_lead_time_recovers_known_value

# Acceptance criteria
- pytest tests/test_eval.py 통과
- PickCube-v1로 small smoke run (2 seed × 5 episode × 3 condition) 가능
- results/ 디렉토리에 phase3_*.parquet 생성

# 하지 말 것
- 그림 그리는 코드 금지 (PR10)
- 다른 condition 간 다른 seed grid 비교 코드 작성 금지

작업 시작.
```

---

## 12. PR10 명령 — Figures

```
@capstone_implementation_spec.md 의 §8.2 메인 figure를 참조해서 PR10을 수행해줘.

# PR10 범위
scripts/make_figures.py 와 figure 헬퍼.

## 1. scripts/make_figures.py
results/ 의 parquet들을 읽어서 figures/ 에 다음 그림들 저장:

- fig01_main_pareto.{pdf,png}: spec §8.2 메인 figure
  - x축: mean wall-clock latency per step (ms, log scale)
  - y축: success rate (95% CI 음영)
  - C1 점, C2 점, C3 곡선 (tau 표시)
  - 태스크별로 subplot
- fig02_detector_roc.{pdf,png}: detector ROC curve, 태스크별 색상
- fig03_lead_time_dist.{pdf,png}: failure trajectory별 first-trigger lead time 분포 (boxplot)
- fig04_calibration.{pdf,png}: detector calibration plot, ECE 표시
- fig05_ablation_features.{pdf,png}: detector feature ablation AUROC bar chart
- fig06_primitive_history.{pdf,png}: condition별 primitive 발동 횟수 stacked bar

## 2. 시각화 규약
- matplotlib + seaborn
- font_size=12, dpi=200
- 색상: 색맹 친화적 palette (seaborn colorblind)
- 모든 그림은 pdf와 png 둘 다 저장

## 3. 테이블
results/phase3_summary_table.csv: spec §8.1의 표 포맷 그대로 (task, cond, success±CI, latency, reasoner_calls)

# Acceptance criteria
- 모든 그림이 figures/에 생성
- 결과 데이터가 부족하면 그 자리에 "Insufficient data" placeholder 그림 출력 (에러 raise 금지)
- LaTeX-friendly: 그림 폭 6 inch, 글꼴 sans-serif

# 하지 말 것
- 데이터를 임의로 변환·보정하지 마. parquet 그대로 사용.
- 보고서 작성 금지 (이건 학생 작업)

작업 시작.
```

---

## 13. 검증·디버깅 명령 (PR 사이에 사용)

### 13.1 학생이 직접 확인하는 명령

```
지금까지 만들어진 코드 전체를 git ls-files 로 보여주고, spec §2 디렉토리 구조와 일치하는지 점검해줘. 빠진 파일이나 spec에 없는 파일을 표로 정리해줘.
```

### 13.2 한 PR이 막혔을 때

```
PR{N}의 작업이 막혔어. 지금 상태에서 다음을 알려줘:
1. 어떤 파일이 완성됐고 어떤 파일이 미완성인가
2. 막힌 원인 (라이브러리 호환성, spec 모호함, API 변경 등)
3. 학생에게 결정이 필요한 항목 (있다면 yes/no 또는 A/B 형태로)
코드는 더 만들지 마. 보고만.
```

### 13.3 spec과 코드가 어긋났는지 점검

```
@capstone_implementation_spec.md 의 §6 (모듈 인터페이스) 와 현재 src/ 코드의 클래스/함수 시그니처가 일치하는지 점검해줘.
어긋난 부분이 있으면 표로 정리하고, spec을 따를 것인지 코드를 유지할 것인지 학생에게 질문해줘.
```

### 13.4 Phase 종료 시 stop criteria 점검

```
@capstone_implementation_spec.md 의 §9 Phase {N} Stop criteria를 기준으로 results/ 의 데이터를 점검해줘.
✓ / △ / ✗ 로 각 기준을 표시하고, 다음 phase로 진행할지 1주 더 시도할지 피벗할지 권장을 한 줄로.
```

---

## 14. 막판 팁

- **PR 사이에 학생이 직접 검토**: Claude Code가 만든 코드를 그대로 머지하지 말고, 적어도 (a) 새 파일이 spec §2와 맞는지, (b) 새 함수 시그니처가 spec §6과 맞는지, (c) 테스트가 실제로 의미 있는지 확인.
- **막히면 무리하지 않기**: Claude Code가 한 PR을 두 번 시도해도 acceptance criteria 못 채우면, 그 PR을 작게 쪼개서 다시 던지거나 학생이 직접 일부 작성.
- **모델 로드는 cache 적극 활용**: `HF_HOME=/content/drive/MyDrive/hf_cache` 환경변수로 Drive에 캐시하면 세션 종료 후에도 재다운로드 없음.
- **Phase 2 stop criteria가 가장 중요**: AUROC ≥ 0.75 통과 못 하면 Phase 3·4는 시간 낭비. Week 8에 솔직하게 판단.
- **항상 git commit**: Claude Code가 만든 코드를 검토 후 즉시 commit. push는 PR 단위로.

---

## 부록 — 자주 쓰는 짧은 보조 명령

- "현재 코드의 cyclomatic complexity가 가장 높은 함수 5개를 보여줘"
- "tests/ 디렉토리의 모든 테스트를 카테고리(unit / integration / slow)로 분류해줘"
- "지난 PR의 CHANGELOG entry를 보고, 다음 PR 시작 전 정합성 점검 체크리스트를 5개 항목으로 만들어줘"
- "results/phase3_*.parquet 의 컬럼 dtype을 spec §5.2와 비교해줘"
