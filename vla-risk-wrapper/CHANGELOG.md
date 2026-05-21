# CHANGELOG

## PR10 (2026-05-19): make_figures + src/utils/ + remaining scripts

- Added `src/utils/{seeding,logging,checkpoint,colab}.py`
- Added `scripts/make_figures.py` (6 figures: Pareto, ROC, lead-time, calibration, ablation, primitive history)
- Added `scripts/phase1_run_baseline.py`
- Added `scripts/phase2_collect_data.py`
- Added `scripts/phase2_train_detector.py`
- Added `scripts/phase3_run_wrapper.py`
- Added `scripts/phase4_ablations.py`
- Added `scripts/check_environment.sh`
- Added `scripts/_dryrun_collect.py`
- Added `notebooks/00_env_smoke_test.ipynb`
- Added `README.md`, `CHANGELOG.md`

## PR9 (2026-05-19): Evaluation runner + metrics + scripts

- Added `src/eval/{runner,metrics,bootstrap,pareto}.py`
- Added `tests/test_eval.py`
- `runner.py`: run_condition(), run_phase3(), seed-grid enforcement, resume support
- `metrics.py`: step_auroc, lead_time_at_recall, ece, success_rate_with_ci, paired_bootstrap_diff, pareto_frontier
- `bootstrap.py`: bootstrap_mean_ci
- `pareto.py`: aggregate_for_pareto, mark_frontier_points

## PR8 (2026-05-19): RiskGatedWrapper + reproducibility tests

- Added `src/wrapper/risk_gated.py`
- Added `tests/test_wrapper.py`
- Added `tests/test_eval_reproducibility.py`
- Implements mode=none/always/risk_gated with primitive lock
- Latency measured in 4 segments with torch.cuda.synchronize()

## PR7 (2026-05-19): Reasoner modules

- Added `src/reasoner/{base,rule_based,vlm_reasoner,prompts}.py`
- Added `tests/test_reasoner.py`
- RuleBasedReasoner: threshold-based decision tree for wrapper validation
- VLMReasoner: Qwen2.5-VL 3B, JSON-only output, parse-fail fallback to 'continue'

## PR6 (2026-05-19): Fallback primitives

- Added `src/primitives/{base,re_grasp,re_approach,align_then_insert,request_help,continue_}.py`
- Added `tests/test_primitives.py`
- All 5 primitives complete within 15 steps

## PR5 (2026-05-19): Risk detector

- Added `src/detector/{features,labeling,dataset,model,train,calibrate}.py`
- Added `tests/test_labeling.py`, `tests/test_detector.py`
- Horizon-based labeling per spec §5.4
- Temperature scaling calibration

## PR4 (2026-05-19): Data collection

- Added `src/data/{schema,io,collect}.py`
- Added `tests/test_data.py`
- StepRecord + EpisodeMeta with parquet I/O
- Seed-unit split (no step/episode leakage)

## PR3 (2026-05-19): Policy modules

- Added `src/policy/{base,octo_policy,openvla_policy,lora}.py`
- Added `tests/test_policy.py`
- OctoPolicy: hidden state extraction, dummy mode fallback
- OpenVLAPolicy: NotImplementedError guard (Phase 2+)

## PR2 (2026-05-19): Environment adapters

- Added `src/env/{factory,randomization,observation}.py`
- Added `tests/test_env.py`
- extract_proprio() returns (8,) float32

## PR1 (2026-05-19): Scaffolding

- Created full directory structure per spec §2
- Added `requirements.txt`, `pyproject.toml`, `.gitignore`
- Added all Hydra YAML configs (env, policy, detector, reasoner, experiment)
- Added `CLAUDE.md`
