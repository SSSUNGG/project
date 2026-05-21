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
