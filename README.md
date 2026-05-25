# Risk-Gated VLA — Phase 1 실행 가이드

## 실행 환경

로컬 개발 → GitHub push → **Colab GPU**에서 실행 (로컬 Windows에서는 Vulkan 제약)

## Colab 실행 (권장)

1. `phase1_colab.ipynb` 열기 (런타임: GPU T4 이상)
2. 셀 순서대로 실행
3. 게이트 1(결정성 테스트) 통과 확인 후 Phase 2 진행

## CLI 실행 (GPU 서버)

```bash
# 의존성 설치
pip install -r requirements.txt

# asset 다운로드 (처음 1회)
python -m mani_skill.utils.download_asset "PickCube-v1" -y

# Phase 1: obs 구조 확인 + 베이스라인 실행
python -m experiments.phase1_baseline --task PickCube-v1 --n_seeds 5 --eps 10 --debug_obs

# collect 모드 (실패율 30~50% 타겟, Phase 2 데이터 수집용)
python -m experiments.phase1_baseline --task PickCube-v1 --n_seeds 5 --eps 10 --randomization collect

# 결과 저장
python -m experiments.phase1_baseline --task PickCube-v1 --n_seeds 5 --eps 10 --save

# 결정성 테스트 (게이트 1)
python -m pytest tests/test_determinism.py -v
```

## Phase 1 목표 판정 기준

| Task              | eval 성공률 목표 |
|-------------------|----------------|
| PickCube-v1       | 40 ~ 70%       |
| StackCube-v1      | 40 ~ 70%       |
| PickSingleYCB-v1  | 30 ~ 60%       |

성공률이 목표 대역을 벗어나면:
- **너무 높음** → `--randomization collect` 사용, 또는 `RANDOMIZATION_PRESETS["collect"]["obj_xy_std"]`를 0.10~0.15로 조정
- **너무 낮음** → `obj_xy_std`를 0.03 이하로 낮춤

## 디렉토리 구조

```
risk_gated_vla/
├── phase1_colab.ipynb          ← Colab 실행용 노트북 (Phase 1)
├── requirements.txt
├── rgvla/
│   ├── interfaces.py           타입 정의
│   ├── envs/make_env.py        환경 생성 + obs 헬퍼
│   ├── policy/
│   │   └── baseline_scripted.py  FSM 베이스라인 (frozen)
│   └── utils/
│       ├── resources.py        GPU/노선 감지
│       ├── timing.py           step 구간 타이밍
│       └── seeding.py          seed grid 생성
├── experiments/
│   └── phase1_baseline.py      Phase 1 CLI 스크립트
├── tests/
│   └── test_determinism.py     게이트 1 (결정성 테스트)
└── results/                    산출물
```

## Phase 진행 순서 (spec §13)

| Phase | 내용 | 게이트 |
|-------|------|--------|
| **1** | 인프라 + 베이스라인 실행 ← **지금** | 결정성 테스트 통과 |
| 2 | 데이터 수집 + horizon labeling | 실패율 30~50% 확인 |
| 3 | Risk detector 학습 | AUROC ≥ 0.70 |
| 4 | Wrapper 통합 + Pareto figure | C3 > C1-C2 직선 |
| 5 | Ablation + 보고서 | — |
| 6 | Medium-B (GPU 확보 시) | — |
