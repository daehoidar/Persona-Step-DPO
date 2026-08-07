# CODEMAP — 파일별 위치·역할

"내가 X를 바꾸려면 어느 파일을 봐야 하나"에 빠르게 답하기 위한 인덱스. 실행 순서와 흐름은 [PIPELINE.md](PIPELINE.md)를 참조.

> 모든 경로는 레포 루트 기준.

## 용어 대응표

코드는 초기 명칭(`persona`, `belief`, `bc_`)을 그대로 쓴다. **파일명·yaml 키·jsonl 필드값이라 바꾸면 스크립트와 기존 산출물이 전부 깨지므로 유지했다.** 논문·포스터 용어와의 대응은 다음과 같다.

| 논문·포스터 | 코드 식별자 |
|---|---|
| SLC-StepDPO | `bc_stepdpo`, `BC-StepDPO` (구 Belief-Conditional StepDPO) |
| 학생 수준 조건 `c` | `persona`, `persona_id`, `persona_tag` |
| 교육과정 제약 프로파일 `κ_c` | `personas.json`의 `forbidden_terms` / `preferred_terms` / `term_evidence` |
| Type-1 (same-level pair) | `pair_type: "step_pair"` |
| Type-2 (cross-level flip pair) | `pair_type: "belief_flip_pair"` |
| `reject_level` | `reject_persona` |
| Explanation Match | `persona_consistency`, `step_persona_err_rate` |
| Belief-Flip | `belief_flip_accuracy` |

---

## 빠른 길찾기

| 하고 싶은 일 | 봐야 할 파일 |
|---|---|
| 수준 조건 추가하거나 어휘 수정 | `personas.json` |
| 손실 수식이 코드로 어떻게 구현됐는지 | `losses/bc_stepdpo_loss.py` |
| GPT judge 프롬프트 수정 | `judge_prompts.py` |
| 수준 적합성 판정 로직 (정규식 → LLM cascade) | `persona_verifier.py` |
| 선호쌍이 어떻게 만들어지는지 | `data_pipeline_stepdpo/3_build_pairs.py` |
| 학습 하이퍼파라미터·ablation toggle | `configs/default.yaml` (상단 주석에 조합표) |
| 포스터 결과표 숫자의 출처 | `eval/*.json` + `data_pipeline/aggregate_results.py` |
| 전체 파이프라인 한 번에 실행 | `scripts/bc_stepdpo_pipeline_slurm.sh` |
| 실험 이력·과거 결과 | `docs/RESULTS_SUMMARY_2026-06-18.md`, `docs/HANDOFF.md` |

---

## 루트 모듈

### `personas.json`
**학생 수준 조건 6종 정의.** 학년 3종(초·중·고) × 성취 2종(상·하).

각 수준마다 `tag`(`<elem_low>` 등 모델이 보는 식별 토큰), `vocabulary_guide` / `explanation_style`(생성 시 참고), `forbidden_terms` / `preferred_terms`(어휘 제약), `exemplar_standards`(교육과정 진술 발췌), `term_evidence`(각 어휘의 최초 도입 학년·코드)를 갖는다.

데이터 생성·judge·학습이 모두 이 파일을 통해 수준을 인식한다. **수정하면 모든 stage 결과에 영향.**

### `persona_verifier.py`
**수준 적합성 판정 3-stage cascade.**

| 단계 | 방식 | 동작 |
|---|---|---|
| Stage A | 정규식 | `forbidden_terms` 단어경계 매치 → `reject_persona` 즉결, 교육과정 코드 자동 첨부. 통과 시 escalate (false-negative 가능) |
| Stage B | 로컬 LLM | 다른 family base 모델로 verdict + confidence. threshold 이상이면 확정. 기본 off |
| Stage C | GPT-4o-mini | 최종 판정 |

### `judge_prompts.py`
GPT judge용 system prompt 3종. 모두 수준 정의와 교육과정 evidence를 자동 주입한다.

| 프롬프트 | 사용처 |
|---|---|
| `GENERATOR_SYSTEM` | Stage 1 — 수준 조건 풀이 합성 |
| `STEP_JUDGE_SYSTEM` | Stage 3 — 각 step의 수준 적합성 평가 |
| `CROSS_BELIEF_CHECK_SYSTEM` | Stage 3 — 같은 step이 두 수준에서 라벨이 뒤집히는지 확인 (Type-2 핵심) |

### `openai_client.py`
API 키 failover. `OPENAI_API_KEY`가 할당량 소진되면 `OPENAI_API_KEY_FALLBACK`(쉼표 구분 다중)으로 자동 전환. 일시적 rate-limit은 호출부 재시도에 위임.

### `inference_backend.py`
vLLM 미설치 환경(Mac M-series 등)용 transformers fallback. `TransformersLLM` / `TransformersSamplingParams`가 vLLM 동명 클래스와 같은 인터페이스를 제공하며, 샘플링·평가 스크립트가 try/except로 자동 전환한다.

### `utils.py`
공용 헬퍼. `load_personas(path)`, `parse_steps(text)`("Step 1: …" 형식을 리스트로 분리), `extract_gsm8k_answer(text)`(`#### 정답` 추출).

---

## `losses/` — 손실 함수

### `losses/bc_stepdpo_loss.py`
**SLC-StepDPO 손실 구현.** 포스터의 손실이 여기 있다.

- `step_logprob(model, …)` — prefix 토큰을 제외하고 **step 토큰만** log 확률 합산. step-level masking의 실체
- `bc_stepdpo_loss(policy, ref, batch, beta)` — `-log σ(β·Δθ)` 산출, Type-1 / Type-2 분리 모니터링

### `losses/bc_fullstepdpo_loss.py`
체인 전체에 per-step reward를 가중 합산하는 변형(`R̂θ(y|x,c) = Σ αᵢ·β·log[πθ/πref]`). PRM 계열 실험용이며 **포스터 결과에는 쓰이지 않았다.**

---

## `data_pipeline/` — 공통 stage + 평가·집계

| 파일 | 역할 |
|---|---|
| `0_seed_problems.py` | Stage 0. MetaMathQA-40K → `GSM_` 필터 → dedupe → 6수준 복제 |
| `1_synthesize_sft.py` | Stage 1. GPT-4o로 수준 조건 풀이 합성 |
| `1_5_split_train_test.py` | Stage 1.5. **`problem_id` 단위 train/test 분리.** held-out 주장의 근거 |
| `2_train_sft.py` | Stage 2. Qwen3-1.7B + LoRA SFT → π_ref |
| `merge_adapter.py` | Stage 2.5. LoRA 머지 → standalone. 샘플링 백엔드가 어댑터를 직접 못 읽어서 필수 |
| `shared_sampling.py` | Stage 3a. π_ref K개 샘플 + 수준 라벨 cascade. 여러 모드가 이 산출물을 공유 |
| `rejudge_contaminated.py` | judge 실패(429·timeout)로 `persona_ok`가 강제 기록된 오염 라벨만 재판정 |
| `augment_type2.py` | Type-2 증량. 후보 수준 제한을 풀고 모든 후보에 cross-check 병렬 수행 |
| `5_evaluate.py` | Stage 5. Final Acc / Step Acc / Explanation Match. **캠페인 스크립트가 실제로 쓰는 평가기** |
| `eval_belief_flip.py` | Belief-Flip. 저수준·고수준 풀이를 각각 생성해 3조건 만족 문제를 카운트 |
| `aggregate_results.py` | `eval/*.json` → 4지표 결과표 이미지 |
| `make_result_figures.py`, `make_heldout_figures.py`, `make_compare_figure.py` | 그림 생성 |

---

## `data_pipeline_stepdpo/` — 선호쌍 구성 + 학습

| 파일 | 역할 |
|---|---|
| `3_build_pairs.py` | Stage 3b. 수학 single-step judge + Type-1 / Type-2 페어 구성. Stage 3a 산출물을 받는 shared 모드 권장 |
| `3_5_analyze_flip_rate.py` | Type-1 중 `reject_persona` 비율, Type-2 개수·수준쌍 매트릭스 |
| `4_train_bc_stepdpo.py` | Stage 4. **SLC-StepDPO 본 학습.** `--lambda-sft`로 anchor 조절 |
| `run_pipeline.sh`, `README.md` | 디렉토리 단독 실행 가이드 |

---

## `data_pipeline_fullstepdpo/` — PRM 계열 (미완)

자가 지도 PRM을 학습해 모든 step에 per-step reward를 부여하는 경로. 외부 GPT 의존 없음. **데이터 골격 단계이며 포스터 결과와 무관하다.**

`3a_mc_rollout_label.py`(MC rollout으로 step value 라벨) → `3b_train_prm.py`(PRM 학습) → `3c_score_and_pack.py`(reward 패킹) → `4_train_fullstepdpo.py`.

---

## `evaluation/` — 보조 평가

| 파일 | 역할 |
|---|---|
| `5_evaluate.py` | 평가기 **변형본**. Belief-Flip을 held-out Type-2 페어의 logprob win rate로 계산 |
| `5_5_make_eval_subset.py` | 평가 서브셋 구성 |
| `6_check_format.py` | 출력 형식(단계 분리·Final 표기) 준수율 |
| `6_5_dump_generations.py` | 생성 결과 덤프 |
| `7_logprob_analysis.py` | logprob 기반 분석 |
| `7_5_rejudge_persona.py` | 수준 라벨 재판정 |
| `8_summarize_bc_smoke.py` | 파이프라인 실행 결과 정리 문서 자동 생성 |

> ⚠️ `5_evaluate.py`가 **두 곳에 있다.** 포스터 Table 1은 `data_pipeline/5_evaluate.py` + `data_pipeline/eval_belief_flip.py` 조합으로 산출됐다 (Belief-Flip = 생성 기반 flip 카운트, 분모 60문제). `evaluation/5_evaluate.py`는 logprob win rate 방식이라 **값이 다르다.** 혼동 주의.

---

## `configs/` — 학습 설정

| 파일 | 용도 |
|---|---|
| `default.yaml` | 기본 설정. 상단 주석에 ablation toggle 조합표 |
| `bc_full_run.yaml` | 풀스케일 본 실험 |
| `bc_retrain.yaml`, `bc_retrain_v3.yaml` | 재학습 preset |
| `abl_vanilla_dpo.yaml` | ablation — step masking 없음 |
| `abl_step_dpo.yaml` | ablation — StepDPO 재현 (`disable_type2: true`) |
| `abl_type1_only.yaml` | ablation — Type-1만 |
| `step_dpo.yaml` | StepDPO 모드 preset |
| `bc_smoke.yaml`, `sft_smoke.yaml`, `smoke.yaml` | 소규모 검증 |

toggle 3종: `disable_step_mask`(step masking 해제), `disable_belief_token`(prompt에서 수준 태그 제거), `disable_type2`(Type-2 제외).

---

## `scripts/` — 실행·그림

**파이프라인**

| 스크립트 | 용도 |
|---|---|
| `bc_stepdpo_pipeline_slurm.sh` | **메인 오케스트레이터.** 머지 → 샘플링 → 페어 → 학습 |
| `sft_slurm.sh` | Stage 2 SFT 단독 |
| `build_pairs_full_slurm.sh` | Stage 3 페어 구성 단독 |
| `runpod_pipeline.sh` | RunPod 환경용 |

**실험 캠페인**

| 스크립트 | 용도 |
|---|---|
| `ablation_campaign_slurm.sh` | Vanilla DPO / StepDPO / Type-1-only 학습 → 머지 → 5모델 평가 → 집계 |
| `run_lambda_sweep_slurm.sh` | λ_sft sweep |
| `run_slc_n60.sh` | SLC 모델 n=60 평가 |
| `run_flip_only.sh`, `reeval_flip_heldout_slurm.sh` | Belief-Flip 재평가 |
| `overnight_strengthen_slurm.sh`, `run_b_campaign_slurm.sh`, `retrain_bc_slurm.sh`, `fullstepdpo_campaign_slurm.sh` | 장기 실험 |
| `eval_format_slurm.sh`, `logprob_slurm.sh` | 보조 평가 |
| `watchdog_bc_pipe.sh`, `watchdog_ovn.sh`, `watchdog_runb.sh` | 장기 job 감시 |

**포스터 그림**

`fig_table_lambda.py`(λ sweep 결과표), `fig_qual_frac.py`(초등 분수 비유 예시), `fig_qual_high.py`(고등 공식 예시), `gen_qual_examples.py`(정성 예시 생성).

> `*_slurm.sh`는 서버 경로 `~/project/Persona-Step-DPO`를 `cd` 한다. 레포명을 바꿨다면 서버 클론 디렉토리명도 확인할 것.

---

## 데이터·결과

| 경로 | 내용 |
|---|---|
| `curriculum/achievement_standards_2022.json` | 2022 개정 수학과 교육과정 성취기준. 학년군 5개(초1-2, 초3-4, 초5-6, 중, 고) × 영역별 |
| `eval/*.json` | 평가 결과 원본. `bc_s0.01_c0.0`, `bc_s0.03_c0`, `slc_s0.01_flip`, `slc_s0.03_flip` |
| `docs/figures_final/` | 포스터·보고용 최종 그림 |
| `docs/figures/` | 초기 비교 그림 |
| `docs/poster/` | 학회 발표 포스터 (PDF + PNG) |

## 문서

| 파일 | 내용 |
|---|---|
| `README.md` | 방법론 개요·포스터·결과표·인용 |
| `PIPELINE.md` | Stage 0→5 실행 흐름 |
| `CODEMAP.md` | 본 파일 |
| `docs/RESULTS_SUMMARY_2026-06-18.md` | **전체 ablation 결과 원본.** 누수 정정 이력 포함 |
| `docs/HANDOFF.md` | 실험 인수인계 기록 |
| `docs/SFT_형식검증_결과보고.md`, `docs/SFT_출력전체.md` | SFT 출력 형식 검증 |
| `docs/DATA_PIPELINE_NOTION.md`, `docs/FULLSTEPDPO_PLAN.md`, `docs/CHANGES_TO_MERGE.md` | 설계 노트 |
| `docs/학교서버_실행가이드.md` | 학교 서버 운용 가이드 |

> `docs/` 내부 문서는 **작성 시점의 기록**이라 구 명칭(BC-StepDPO, persona, Full Step-DPO)이 그대로 남아 있다. 소급 수정하지 않았다.

## `archive/` · `tests/`

`archive/`는 폐기된 파일 보관소다 (`root_app.py`, `root_derive_persona_evidence.py`, 구 `tests/` 하니스, 특허 도면 스크립트 등). 현재 파이프라인은 아무것도 참조하지 않으며, 이력 보존 목적으로만 남겨두었다.

`tests/sft_vs_trained.py`가 유일하게 남은 테스트로, SFT 모델과 학습 모델의 출력을 비교한다.
