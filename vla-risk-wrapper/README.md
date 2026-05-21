# VLA Risk-Aware Selective Recovery Wrapper

## 한 줄 요약

ManiSkill3 contact-rich 제조 태스크에서, frozen VLA 정책(Octo 또는 OpenVLA) 위에 학습된 risk detector를 부착하여 `r_t ≥ τ`인 step에서만 VLM reasoner + fallback primitive를 호출하는 wrapper. baseline / always-reasoning / risk-gated 세 조건을 동일 seed grid로 평가하여 success–latency Pareto frontier를 보고한다.

## 디렉토리 구조

```
vla-risk-wrapper/
├── configs/          # Hydra YAML configs (env, policy, detector, reasoner, experiment)
├── src/
│   ├── env/          # ManiSkill3 환경 어댑터 (factory, randomization, observation)
│   ├── policy/       # VLA 정책 (Octo, OpenVLA placeholder)
│   ├── detector/     # Risk MLP (features, labeling, dataset, model, train, calibrate)
│   ├── reasoner/     # Rule-based + VLM reasoner (Qwen2.5-VL)
│   ├── primitives/   # Fallback primitives (re_grasp, re_approach, etc.)
│   ├── wrapper/      # RiskGatedWrapper (핵심 모듈)
│   ├── data/         # 데이터 스키마, I/O, 수집
│   ├── eval/         # 평가 러너, 지표, bootstrap, Pareto
│   └── utils/        # 시드, 로깅, 체크포인트, Colab 유틸
├── scripts/          # 실행 스크립트 (phase1~4)
├── notebooks/        # Colab 노트북
├── tests/            # pytest 테스트
└── capstone_implementation_spec.md  # 풀 스펙 문서
```

## 실행 순서

### Day 0 체크리스트

```bash
# 1. GPU 확인
bash scripts/check_environment.sh

# 2. ManiSkill 테스트
python -m mani_skill.examples.demo_random_action -e PickCube-v1

# 3. pytest 수집 확인
pytest --collect-only
```

### Phase 1 — Baseline

```bash
python scripts/phase1_run_baseline.py env=peg_insertion
```

### Phase 2 — 데이터 수집 + Detector 학습

```bash
python scripts/phase2_collect_data.py
python scripts/phase2_train_detector.py
```

### Phase 3 — Wrapper 평가

```bash
python scripts/phase3_run_wrapper.py
python scripts/phase3_run_wrapper.py --resume  # 중단 후 재시작
```

### Phase 4 — 분석 + 그림

```bash
python scripts/phase4_ablations.py
python scripts/make_figures.py
```

## 테스트

```bash
pytest -q                        # 빠른 테스트 (slow 제외)
pytest -m "not slow" tests/      # 명시적
pytest tests/test_labeling.py    # 특정 모듈만
```

## 주요 설계 결정

| 항목 | 결정 |
|---|---|
| 베이스 정책 | Octo-base (Phase 1–2), OpenVLA-7B (Phase 2 통과 후) |
| Risk detector | MLP (3-layer, 256 hidden) + temperature scaling |
| Reasoner | Rule-based (Phase 7–8), Qwen2.5-VL 3B (Phase 9+) |
| 데이터 split | Seed 단위 (70/15/15) — step/episode split 금지 |
| Latency 측정 | `torch.cuda.synchronize()` 전후 `time.perf_counter()` |
| 재현성 보장 | 세 condition 동일 seed grid, obs hash 검증 테스트 포함 |

## 환경 변수

```bash
export DRIVE_ROOT=/content/drive/MyDrive/vla-risk-wrapper
export HF_TOKEN=hf_...
export HF_HOME=/content/drive/MyDrive/hf_cache  # 세션 종료 후 재다운로드 방지
```

## Stop Criteria 요약

| Phase 끝 | 기준 | 판단 |
|---|---|---|
| Phase 1 (Week 4) | PegInsertion 성공률 30–70% | ✓ 진행 / ✗ Octo→OpenVLA 전환 |
| Phase 2 (Week 8) | AUROC ≥ 0.75 AND lead time ≥ 10 step | ✓ 진행 / ✗ 피벗 검토 |
| Phase 3 (Week 10–11) | C3 > C1 +5%p AND latency < C2×50% | ✓ Phase 4 / ✗ 디버깅 1주 |

자세한 내용은 `capstone_implementation_spec.md` 참조.
