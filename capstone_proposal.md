# 졸업프로젝트 제안서

## 1. 연구 제목

**한국어**
사전적 위험 예측 기반 선택적 추론 래퍼를 통한 제조형 Vision-Language-Action 조작 정책의 효율–성능 트레이드오프 분석

**English**
Predictive Risk-Gated Selective Reasoning Wrapper for Vision-Language-Action Manipulation Policies on Contact-Rich Manufacturing Tasks

---

## 2. 연구 배경 및 목표

### 2.1 배경

OpenVLA, π0, Octo로 대표되는 Vision-Language-Action (VLA) 모델은 RGB 이미지와 언어 지시문을 입력받아 직접 로봇 액션을 출력하는 end-to-end 정책으로, 일반화된 조작 능력을 보이고 있다. 그러나 두 가지 한계가 분명하다.

1. PegInsertionSide, AssemblingKits 같은 contact-rich 제조 태스크에서는 정렬 오류·삽입 실패·grasp 슬립 등으로 인해 실패율이 여전히 높다.
2. ECoT 계열처럼 매 step chain-of-thought reasoning을 수행하는 방식은 latency가 커서 실시간 제어에 부적합하며, 그렇다고 reasoning을 끄면 (1)의 실패를 회복할 수단이 없다.

본 연구는 frozen VLA 위에 *학습된 사전적(predictive) 위험 점수가 임계값을 넘을 때만 느린 추론기를 호출하는 wrapper* 를 부착하여, 성공률을 유지하면서 reasoning 호출과 wall-clock latency를 명시적으로 통제하는 방식을 제안한다. 평가는 ManiSkill3 위의 contact-rich 제조 태스크에서 수행한다.

### 2.2 연구 질문 (Research Questions)

- **RQ1 (Detection).** 학습된 risk score가 실패를 *사전에* 예측할 수 있는가? — AUROC, lead time at recall=0.9
- **RQ2 (Selective wrapper).** risk-gated 호출이 always-reasoning 대비 성공률을 유지하면서 호출 빈도와 wall-clock latency를 줄이는가? — success–latency Pareto frontier
- **RQ3 (Manufacturing specificity).** contact-rich 제조 태스크에서 wrapper 효과가 일반 manipulation 대비 큰가? — task category 간 Δsuccess 비교
- **RQ4 (Detector ablation).** detector 입력 feature로 VLA 내부 hidden state, action history, RGB heuristic, hybrid 중 어떤 조합이 가장 효과적인가?

### 2.3 기대 기여

1. VLA에 추가 센서 없이 RGB만으로 부착 가능한 sensor-free wrapper 설계
2. predictive risk score를 통해 latency budget을 명시적으로 제어하는 selective reasoner gating 메커니즘 제시
3. ManiSkill3 contact-rich 제조 태스크에서 baseline 대비 성공률 향상 및 always-reasoning 대비 latency–Pareto 우위 입증
4. detector 입력 feature ablation을 통한 "어떤 신호가 실패 예측에 유용한가" 정량 분석

---

## 3. 선행 연구 및 본 연구의 차별점

### 3.1 직접 관련 선행 연구 요약

| 연구 | 핵심 아이디어 | 본 연구와 유사도 | 본 연구의 차별 |
|---|---|---|---|
| FailSafe (2510.01642) | ManiSkill 위 VLA failure recovery wrapper | 매우 높음 (~85%) | reactive correction; predictive gating 부재; latency Pareto 미보고 |
| SAFE (2506.09937, NeurIPS 2025) | VLA internal feature 기반 failure detector | 높음 (~70%) | detector만 다루고 closed-loop recovery 없음 |
| SC-VLA (2405.17418) | VLA 내부 fast–slow dual system | 높음 (~90%) | 실패 발생 후 reactive 보정; 본 연구는 사전 예측 trigger |
| AHA (2410.00371, ICLR 2025) | failure reasoning VLM (NVIDIA) | 중간 (~60%) | TAMP planning feedback용; 실시간 manipulation closed-loop 아님 |
| RoboFAC (2505.12224) | failure analysis VLM + correction | 중간 (~65%) | 일반 manipulation 중심, 제조 contact-rich 미평가 |
| ECoT (2407.08693, CoRL 2024) | 매 step embodied CoT | 낮음 (~40%) | 본 연구의 *always-reasoning* 비교 baseline |
| REFLECT (2306.15724, CoRL 2023) | LLM 기반 실패 요약·교정 | 낮음 (~45%) | 본 연구 패러다임의 historical reference |
| ForceVLA (2505.22159), VTLA (2505.09577) | force/tactile 센서 추가로 contact-rich 강화 | 낮음 (~30%) | 본 연구는 sensor-free, RGB만 사용 |

### 3.2 본 연구의 4가지 차별 포인트 (누적)

1. **Predictive risk-gating**
   기존 SC-VLA, FailSafe는 *실패 발생 후* 보정을 수행하는 reactive 구조이다. 본 연구는 실패 시점까지의 horizon H 안에서 발생할 실패를 step 단위로 예측하여 reasoner를 *사전* 호출한다. *lead time at recall=0.9* 를 정량 차별화 지표로 사용한다.

2. **Manufacturing-specific tasks**
   LIBERO·BridgeV2 위주의 일반 manipulation이 아닌, PegInsertionSide·AssemblingKits 같은 contact-rich industrial task에서 1차 평가한다.

3. **Compute-aware Pareto 평가**
   τ sweep으로 reasoning 호출 빈도와 wall-clock latency의 Pareto frontier를 명시적으로 그려 baseline, always-reasoning과 비교한다. 학부 캡스톤에서 명시적 latency 곡선을 보고한 VLA 회복 연구는 아직 드물다.

4. **RGB-only wrapper**
   ForceVLA·VTLA가 force/tactile 센서를 추가하여 contact-rich를 보강한다면, 본 연구는 추가 센서 없이 wrapper만 부착하여 같은 태스크의 robustness 개선 가능성을 검증한다. "deployable on edge" 메시지의 근거가 된다.

### 3.3 한 줄 framing

기존 selective reasoning VLA가 실패 *발생 후* reactive correction에 의존하는 반면, 본 연구는 OpenVLA-class 정책 위에 학습된 *사전적 risk detector* 를 두어 contact-rich manufacturing 태스크에서 reasoning 호출 빈도를 명시적으로 통제하면서 성공률–지연 Pareto frontier를 baseline 대비 개선한다.

---

## 4. 시뮬레이터 및 환경

### 4.1 시뮬레이터 선정

**ManiSkill3 (메인) + LIBERO (보조)** 조합을 선정한다.

ManiSkill3 선정 근거:

- Colab Free/Pro에서 공식 quickstart 노트북이 동작 (T4 ~15 GB VRAM 환경에서 검증됨)
- contact-rich 제조 태스크가 표준 환경으로 포함 — PegInsertionSide, AssemblingKits, PickSingleYCB, PlugCharger
- OpenVLA, Octo, RDT-1B 등 VLA-style 정책의 ManiSkill 적응판이 공식 baseline으로 제공
- 모션플래닝·teleop·RL 기반 demonstration이 공개되어 behavior cloning 학습 가능
- 본 연구의 가장 가까운 선행 연구인 FailSafe가 ManiSkill을 사용 → 직접 비교 가능

검토된 대안과 탈락 사유:

| 대안 | 탈락 사유 |
|---|---|
| Isaac Lab / Factory | contact-rich physics 충실도 최고이나 Colab 안정 동작 비공식 |
| RoboSuite | 가벼우나 manufacturing 태스크가 ManiSkill 대비 부족 |
| LIBERO 단독 | VLA 표준 벤치이나 manufacturing 태스크 부재 — 보조 일반화 평가로 사용 |
| MetaWorld | 언어 conditioning 약함, VLA 적합도 낮음 |
| RLBench | Colab 헤드리스 설치 난이도 높음 |

### 4.2 사용 태스크

| 태스크 | 분류 | 역할 | 주요 실패 모드 |
|---|---|---|---|
| PickCube, StackCube | 일반 manipulation | Phase 1 sanity check | grasp 슬립 |
| **PegInsertionSide** | contact-rich 삽입 | 메인 평가 ★ | 정렬 오류, 삽입 실패 |
| **AssemblingKits** | 조립 | 메인 평가 ★ | wrong-slot, orientation 오차 |
| PickSingleYCB | semantic disambiguation | RQ3 보조 ablation | wrong-object selection |
| LIBERO 1 suite | 일반화 평가 | 외부 비교 baseline | 다양 |

---

## 5. 데이터셋

### 5.1 베이스 정책 적응 데이터 (Module A)

- 원천: ManiSkill3 공식 demonstration (motion planning, RL, teleop으로 수집)
- 규모: 태스크당 1,000~3,000 trajectory
- 용도: OpenVLA-7B 또는 Octo 체크포인트 위 LoRA fine-tune으로 ManiSkill action space에 맞춤
- 학습 자원: Colab Pro A100 1–2회, LoRA + 8-bit quantization

### 5.2 Risk Detector 학습 데이터 (Module B, 자가 수집)

- 베이스 정책을 강하게 randomize된 환경에서 실행 (object pose, lighting, distractor, viewpoint shift)
- 실패율을 30–50% 영역으로 의도적으로 유지 (실패 데이터 부족 방지)
- 매 step `(VLA hidden state h_t, action history K=8, proprioception, episode_id, t)` 저장
- 종료 후 *horizon-based labeling* (default H=20)
  - 성공 episode: 모든 step label = 0
  - 실패 episode: [t_fail − H, t_fail] 구간 label = 1, 나머지 label = 0
- 목표 규모: 3 태스크 × 1,500 episode × 평균 80 step ≈ 360k step
- 분할: **seed 단위 70/15/15** (step random split 금지 — leakage 방지)

### 5.3 평가 데이터 (Phase 3)

- baseline, always-reasoning, risk-gated 세 조건을 *동일 seed × 동일 episode* 매트릭스로 평가
- 태스크당 5 seed × 100 episode = 500 episode/조건
- 총 3 메인 태스크 × 3 조건 × 500 episode = 4,500 episode

---

## 6. 연구 프레임워크

### 6.1 시스템 아키텍처 (Per-step 흐름)

```
관측 (RGB + 언어)
    │
    ▼
[Module A] frozen VLA  ──►  candidate action a_t + hidden state h_t
                                       │
                                       ▼
[Module B] Risk Detector(h_t, action_hist, proprio)  ──►  r_t ∈ [0,1]
                                       │
                          ┌────────────┴────────────┐
                  r_t < τ │                         │ r_t ≥ τ
                          ▼                         ▼
                  Execute a_t              [Module C] Reasoner VLM
                  (fast path)              + Fallback Primitive
                                                    │
                                                    ▼
                                          Selected primitive 실행
                          └────────────┬────────────┘
                                       ▼
                              ManiSkill env step
                                       │
                                       ▼
                              Logging (success, latency,
                              reasoner call, r_t, primitive)
```

### 6.2 모듈 구성

| 모듈 | 역할 | 학습/구성 | 추론 비용 |
|---|---|---|---|
| **A. Base VLA** | 후보 action 생성 | OpenVLA-7B / Octo + LoRA adapt | ~30 ms / step |
| **B. Risk Detector** | per-step risk score r_t | 자가 수집 데이터로 학습된 MLP (~1.5M param) | <1 ms / step |
| **C. Selective Reasoner + Fallback Library** | risk 발생 시 진단 + primitive 선택 | LLaVA-NEXT 7B 또는 Qwen2.5-VL 3B + 5 primitive | ~150–300 ms / 호출 |

5개 Fallback Primitive (산업 매뉴얼 대응):

1. `re_grasp` — 현재 grip 풀고 재파지
2. `re_approach` — 10 cm 후퇴 후 재진입
3. `align_then_insert` — 시각 정렬 보정 후 삽입
4. `request_help` — 포기 / abort
5. `continue` — 위험 오탐, baseline action 그대로 유지

### 6.3 Risk Detector 상세

입력 feature (320차원):

- VLA hidden state mean-pool (4096차원, 첫 layer에서 256으로 축소)
- Action history K=8 step (8 × 7 = 56차원)
- Proprioception: gripper state, end-effector pose, contact proxy (8차원)

구조: 3-layer MLP + LayerNorm + GELU + Dropout(0.1), sigmoid output

학습:

- Loss: weighted BCE (`pos_weight=4.0`) 또는 focal loss (γ=2)
- Optimizer: AdamW, lr=3e-4, weight_decay=1e-2
- Batch: 256 step, 20–30 epoch, val AUROC로 early stopping
- Calibration: val set 기반 temperature scaling

---

## 7. 연구 단계별 상세 계획 (16주)

### Phase 1 (Week 1–4) — 인프라 및 베이스라인 확립

**목표.** baseline VLA가 ManiSkill에서 동작하고, 평가 파이프라인 (seed × episode × 지표 CSV 로그)이 완성된 상태.

작업:

- ManiSkill3 Colab quickstart 실행, PegInsertionSide, AssemblingKits, PickSingleYCB, PickCube, StackCube 5개 환경 동작 검증
- OpenVLA-7B 체크포인트 로드, ManiSkill action space에 맞춰 LoRA head 적응 학습 (Colab Pro A100)
- 평가 스크립트 작성: seed × episode 매트릭스로 success rate, episode length, mean wall-clock latency per step을 CSV로 로그
- 5 seed × 100 episode baseline 수치 확보

산출물: baseline 성능 표 1장, 재현 가능한 Colab 노트북 1세트, Drive 기반 결과 저장 구조

위험: Colab 세션 종료 — 매 N episode마다 결과 append 저장으로 대응

### Phase 2 (Week 5–8) — 실패 데이터 수집 및 Risk Detector 학습

**목표.** AUROC ≥ 0.75, lead time ≥ 10 step의 detector가 학습된 상태.

작업:

- 환경 randomization 강도 조절 — 실패율을 30–50% 영역으로 끌어올림
- 데이터 수집 스크립트: 매 step `(h_t, action_history, proprio, t, episode_id, success_flag)` 저장
- horizon-based labeling (H=20 기본, 태스크별 H 조정)
- Risk Detector MLP 학습 — Section 6.3 설계 적용
- val AUROC, lead time at recall=0.9, ECE 보고
- seed 단위 70/15/15 train/val/test 분할

산출물: detector checkpoint, ROC curve, lead time curve, calibration plot

위험: detector AUROC < 0.7 — feature 추가 (visual heuristic), focal loss, 데이터 추가 수집으로 대응

### Phase 3 (Week 9–12) — Wrapper 통합 및 본실험

**목표.** 3가지 조건의 Pareto curve 1장으로 main result 완성.

작업:

- Selective reasoner VLM 통합 (LLaVA-NEXT 7B 1차 시도, 무거우면 Qwen2.5-VL 3B로 downgrade)
- 5개 fallback primitive를 ManiSkill action sequence로 구현
- 3가지 조건 평가, 동일 seed × episode 매트릭스:

| 조건 | 설명 |
|---|---|
| **C1 (baseline)** | VLA만 실행 |
| **C2 (always-reasoning)** | 매 step VLM critic + primitive 호출 — latency upper bound |
| **C3 (risk-gated, ours)** | τ ∈ {0.3, 0.5, 0.7} sweep |

- 메인 figure: x = mean wall-clock latency per step, y = success rate
  - C1 좌하단 점, C2 우상단 점, C3 τ 변화에 따라 두 점 사이의 곡선
  - 곡선이 두 점을 잇는 직선보다 위쪽이면 Pareto 우위

산출물: 메인 Pareto figure, 조건별 결과 표 (95% bootstrap CI 포함)

위험: VLM이 너무 무거워 Colab 세션 한계 초과 — Qwen2.5-VL 3B 또는 rule-based primitive selector로 대체

### Phase 4 (Week 13–16) — Ablation, 분석, 보고서

**목표.** 차별화 4 포인트가 모두 정량 입증된 상태.

작업:

- **Detector ablation (RQ4)**: hidden state only / action history only / RGB heuristic only / hybrid 4조합 비교
- **Manufacturing vs general (RQ3)**: PegInsertionSide·AssemblingKits 결과를 PickSingleYCB·LIBERO와 비교 → Δsuccess 표
- **Primitive library ablation**: 5개 vs 3개 vs single retry
- **Horizon H sweep**: H ∈ {10, 20, 30}
- **τ fine sweep**: τ ∈ {0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8}
- 모든 결과에 95% bootstrap CI (B=1000) 추가
- 보고서 작성, 발표자료 제작, 코드 저장소 정리, README 작성

산출물: 졸업 보고서 (한/영), 발표자료, GitHub 코드 저장소

---

## 8. 평가 방법

### 8.1 주요 지표

| RQ | 지표 | 보고 방식 |
|---|---|---|
| RQ1 (Detection) | step-level AUROC | mean ± 95% bootstrap CI |
| | lead time at recall=0.9 | mean step count, 분포 boxplot |
| | ECE (10 bin) | scalar |
| RQ2 (Wrapper) | success rate | mean ± 95% CI |
| | reasoner call frequency | per episode 평균 |
| | mean wall-clock latency per step | ms, Colab T4 기준 |
| | fallback precision / recall | scalar |
| RQ3 (Manufacturing) | Δsuccess (C3 − C1) | task category별 비교 |
| RQ4 (Ablation) | feature 조합별 AUROC | 표 |

### 8.2 실험 조건

3가지 조건을 *동일한 seed, 동일한 reset, 동일한 episode* 매트릭스로 평가한다. 비교는 절대 다른 seed에서 수행하지 않는다.

- **C1 baseline**: VLA만
- **C2 always-reasoning**: 매 step VLM critic + primitive (latency upper bound)
- **C3 risk-gated**: τ sweep

### 8.3 통계 처리

- 모든 success rate에 95% bootstrap CI (B=1000)
- detector AUROC에도 bootstrap CI
- 두 조건 비교 시 paired permutation test 또는 paired bootstrap
- 같은 seed에서의 차이만 비교 (paired 비교)

### 8.4 Main Figure (제출 보고서의 핵심 그림)

x축: 평균 wall-clock latency per step (ms)
y축: success rate (95% CI 음영)

- C1: 좌하단 한 점
- C2: 우상단 한 점
- C3: τ 변화에 따라 두 점 사이를 잇는 곡선

해당 곡선이 C1–C2를 잇는 직선 위쪽에 있으면 "더 적은 latency로 더 높은 성공률"이라는 한 줄 메시지가 그림 한 장으로 전달된다.

---

## 9. 자원 및 제약

| 항목 | 내용 |
|---|---|
| 컴퓨팅 | Colab Free + Pro (T4 ~15 GB, L4/A100 occasional) |
| Storage | Google Drive 100 GB 가정 (자가 수집 데이터셋 ~10–20 GB) |
| 모델 | 공개 OpenVLA-7B, Octo, LLaVA-NEXT 7B, Qwen2.5-VL 3B |
| 외부 데이터셋 | ManiSkill3 demonstration (공개) |
| 일정 | 16주 학기 |
| 코드 공개 | GitHub 저장소 (졸업 후 공개 예정) |

---

## 10. 위험 요소 및 대응

| 위험 | 대응 |
|---|---|
| OpenVLA fine-tune이 Colab 자원 부족 | LoRA + 8-bit quantization, 또는 Octo로 대체 |
| 실패 데이터 부족 (성공률 > 90%) | 환경 randomization 강도 증가, distractor 추가, H 증가 |
| detector AUROC < 0.7 | feature 추가, focal loss, 데이터 추가 수집 |
| Seed leakage로 비현실적 AUROC | seed 단위 train/val/test 분할 엄수, "always 1" trivial check |
| Colab 세션 종료로 학습 중단 | 매 N episode마다 Drive에 checkpoint append, restart-resume 구조 |
| Selective reasoner VLM이 너무 무거움 | Qwen2.5-VL 3B로 downgrade, 또는 rule-based fallback selector |
| FailSafe와 결과가 비슷해 차별화 약함 | predictive lead time, latency Pareto, manufacturing Δsuccess 3 지표를 명시 |

---

## 11. 참고문헌

[1] Lin et al. (2025). *FailSafe: Reasoning and Recovery from Failures in Vision-Language-Action Models.* arXiv:2510.01642.
[2] Gu et al. (2025). *SAFE: Multitask Failure Detection for Vision-Language-Action Models.* NeurIPS 2025. arXiv:2506.09937.
[3] Li, Liu et al. (2024). *A Self-Correcting Vision-Language-Action Model for Fast and Slow System Manipulation.* arXiv:2405.17418.
[4] Duan et al. (2025). *AHA: A Vision-Language-Model for Detecting and Reasoning Over Failures in Robotic Manipulation.* ICLR 2025. arXiv:2410.00371.
[5] Wang et al. (2025). *RoboFAC: A Comprehensive Framework for Robotic Failure Analysis and Correction.* arXiv:2505.12224.
[6] Zawalski et al. (2024). *Robotic Control via Embodied Chain-of-Thought Reasoning.* CoRL 2024. arXiv:2407.08693.
[7] Liu et al. (2023). *REFLECT: Summarizing Robot Experiences for Failure Explanation and Correction.* CoRL 2023. arXiv:2306.15724.
[8] Yu et al. (2025). *ForceVLA: Enhancing VLA Models with a Force-aware MoE for Contact-rich Manipulation.* arXiv:2505.22159.
[9] Hao et al. (2024). *ManiSkill3: GPU Parallelized Robotics Simulation and Rendering for Generalizable Embodied AI.* arXiv:2410.00425.
[10] Kim et al. (2024). *OpenVLA: An Open-Source Vision-Language-Action Model.* CoRL 2024. arXiv:2406.09246.

---

## 부록 — 16주 일정 요약

| 주차 | Phase | 주요 작업 | 마일스톤 |
|---|---|---|---|
| W1–2 | 1 | ManiSkill 환경 구축, OpenVLA 로드 | 5개 환경 동작 확인 |
| W3–4 | 1 | LoRA adapt, 평가 스크립트, baseline 측정 | 5 seed × 100 ep baseline 표 |
| W5–6 | 2 | 실패 데이터 수집 파이프라인, randomization | 360k step 데이터셋 |
| W7–8 | 2 | Risk Detector 학습, AUROC/lead-time 보고 | detector checkpoint |
| W9–10 | 3 | VLM 통합, primitive 구현 | wrapper end-to-end 동작 |
| W11–12 | 3 | 3 조건 본실험, τ sweep | 메인 Pareto figure |
| W13 | 4 | Detector ablation, H sweep | ablation 표 |
| W14 | 4 | Manufacturing Δsuccess, primitive ablation | RQ3 표 |
| W15 | 4 | 통계 정리, 그림 최종화 | 모든 figure 완성 |
| W16 | 4 | 보고서, 발표자료, 코드 정리 | 제출 |
