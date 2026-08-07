# SLC-StepDPO 데이터 파이프라인

Stage 0 → 5 전체 흐름을 정리한다. README가 "무엇을 왜"라면, 본 문서는 "어떤 파일이 무엇을 만드는지"에 집중한다. 파일별 역할 인덱스는 [CODEMAP.md](CODEMAP.md)를 참조.

## 용어 대응표

레포 코드는 초기 명칭(`persona`, `belief`, `bc_`)을 그대로 쓴다. 논문·포스터 용어와의 대응은 다음과 같다. **코드 식별자는 바꾸지 않았다** — 파일명·yaml 키·jsonl 필드값이라 바꾸면 스크립트와 기존 산출물이 전부 깨진다.

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

## 0. 전체 흐름

```
 Stage 0   data_pipeline/0_seed_problems.py          MetaMathQA-40K → GSM_ 필터 → ×6 수준
              ↓ seed_problems.jsonl
 Stage 1   data_pipeline/1_synthesize_sft.py         GPT-4o 수준 조건 풀이 합성
              ↓ sft_data.jsonl
 Stage 1.5 data_pipeline/1_5_split_train_test.py     problem_id 단위 train/test 분리 ★누수 차단
              ↓ sft_train.jsonl / sft_test.jsonl
 Stage 2   data_pipeline/2_train_sft.py              Qwen3-1.7B + LoRA → π_ref
              ↓ checkpoints/sft_ref/
 Stage 2.5 data_pipeline/merge_adapter.py            LoRA 머지 → standalone (샘플링 백엔드용)
              ↓ checkpoints/*_merged/
 Stage 3a  data_pipeline/shared_sampling.py          π_ref K개 샘플 + 수준 라벨 cascade
              ↓ samples_with_persona_labels.jsonl
 Stage 3b  data_pipeline_stepdpo/3_build_pairs.py    수학 judge + Type-1 / Type-2 페어
              ↓ preference_pairs.jsonl
 Stage 3.5 data_pipeline_stepdpo/3_5_analyze_flip_rate.py   flip rate 통계
           data_pipeline/augment_type2.py            Type-2 증량 (선택)
              ↓
 Stage 4   data_pipeline_stepdpo/4_train_bc_stepdpo.py  + losses/bc_stepdpo_loss.py
              ↓ checkpoints/bc_stepdpo/
 Stage 5   data_pipeline/5_evaluate.py               Final / Step / Explanation Match
           data_pipeline/eval_belief_flip.py         Belief-Flip
           data_pipeline/aggregate_results.py        eval/*.json → 결과표
```

Stage 0~2.5는 공통이다. 분기는 Stage 3에서 일어난다.

| | **StepDPO (baseline)** | **SLC-StepDPO (제안)** | **PRM 계열 (미완)** |
|---|---|---|---|
| Stage 3 | `data_pipeline_stepdpo/3_build_pairs.py` (`--disable-type2`) | 같은 파일, Type-1 + Type-2 | `data_pipeline_fullstepdpo/3a·3b·3c` |
| 페어 종류 | `step_pair`만 | `step_pair` + `belief_flip_pair` | 체인 단위 per-step reward |
| 학습 config | `configs/abl_step_dpo.yaml` | `configs/default.yaml`, `configs/bc_full_run.yaml` | — |
| 상태 | 완료 | 완료 (포스터 결과) | 데이터 골격만 |

---

## 1. Stage 0 — Seed Problem Sampling

**파일**: `data_pipeline/0_seed_problems.py`

**입력**: [meta-math/MetaMathQA-40K](https://huggingface.co/datasets/meta-math/MetaMathQA-40K) — GSM8K와 MATH를 AnsAug / Rephrased / FOBAR / SV 4가지로 증강한 40K 행

**처리**
1. `type` 컬럼이 `GSM_`로 시작하는 행만 필터 (MATH 계열 제외)
2. `query` 컬럼 기준 dedupe
3. 풀에서 N개 무작위 추출 → 6개 수준 조건에 복제 배정

기본값은 `--n-problems 1500`, `--seed 42`. 포스터의 "1,500 problems from MetaMathQA-40K"가 이 단계다.

**출력**: `data_pipeline/output/seed_problems.jsonl` — N × 6 행

```json
{
  "problem_id": "metamath_42",
  "persona": "elem_low",
  "question": "Janet's ducks lay 16 eggs per day. ...",
  "gt_answer": "18",
  "augmentation_type": "GSM_AnsAug"
}
```

---

## 2. Stage 1 — SFT 데이터 합성

**파일**: `data_pipeline/1_synthesize_sft.py`

각 `(problem, 수준조건)` 행에 대해 **GPT-4o**로 `κ_c` 제약을 지키는 풀이를 `--solutions-per-row`만큼 합성한다. 프롬프트는 `judge_prompts.GENERATOR_SYSTEM`이고, 수준별 어휘 제약과 교육과정 근거가 system prompt에 자동 주입된다.

**출력**: `data_pipeline/output/sft_data.jsonl`

```json
{
  "problem_id": "metamath_42",
  "persona_id": "elem_low",
  "persona_tag": "<elem_low>",
  "solution_text": "Step 1: ...",
  "steps": ["Step 1: ...", "Step 2: ..."]
}
```

---

## 3. Stage 1.5 — Train / Test 분할

**파일**: `data_pipeline/1_5_split_train_test.py`

**같은 base 문제의 다른 증강본·다른 수준 변형이 train과 test에 동시에 들어가지 않도록 `problem_id` 단위로 분리한다.** MetaMathQA는 한 GSM8K 문제를 4가지로 증강하므로, 행 단위로 자르면 같은 문제가 양쪽에 걸린다.

포스터 Table 1의 *"Held-out evaluation with no train/eval overlap"* 이 이 단계에 근거한다. 초기 실험에서 이 분리가 없어 Final Acc가 90.3으로 부풀려졌고, 분리 후 73.9로 정정됐다 (`docs/RESULTS_SUMMARY_2026-06-18.md`).

---

## 4. Stage 2 — SFT 학습 → π_ref

**파일**: `data_pipeline/2_train_sft.py`

`Qwen/Qwen3-1.7B`에 LoRA로 SFT. 수준 분기는 prompt 포맷(`<elem_low>\nProblem: …`)으로 학습된다.

**출력**: `checkpoints/sft_ref/` (LoRA adapter)

> 스모크 테스트에서는 `Qwen/Qwen3-0.6B`를 쓰기도 하지만, **포스터에 보고된 모든 수치는 Qwen3-1.7B 기준**이다.

### Stage 2.5 — 어댑터 머지

**파일**: `data_pipeline/merge_adapter.py`

샘플링 백엔드(vLLM / `inference_backend.TransformersLLM`)가 LoRA 어댑터를 직접 로드하지 못한다. base + adapter를 머지해 standalone 모델로 저장해야 이후 단계가 돈다.

---

## 5. Stage 3a — 공유 샘플링 + 수준 라벨

**파일**: `data_pipeline/shared_sampling.py`

π_ref에서 각 `(problem, 수준조건)`에 K개 풀이를 샘플링하고, 각 step에 수준 적합성 라벨을 붙인다. 여러 모드가 이 산출물을 **공유**하므로 샘플링과 judge 호출이 1회로 끝난다.

라벨링은 `persona_verifier.py`의 3-stage cascade다.

| 단계 | 방식 | 역할 |
|---|---|---|
| Stage A | 정규식 | `forbidden_terms` 단어경계 매치 → `reject_persona` 즉결. 교육과정 코드 자동 첨부 |
| Stage B | 로컬 LLM | 다른 family base 모델로 verdict + confidence (기본 off, `--disable-stage-b`) |
| Stage C | GPT-4o-mini | 최종 판정. few-shot + 교육과정 근거 주입 |

**출력**: `data_pipeline/output/samples_with_persona_labels.jsonl`

> judge 실패(429·timeout)로 `persona_ok`가 강제 기록된 오염 라벨은 `data_pipeline/rejudge_contaminated.py`로 재판정한다.

---

## 6. Stage 3b — 선호쌍 구성

**파일**: `data_pipeline_stepdpo/3_build_pairs.py`

Stage 3a 산출물에 **수학 single-step judge**(GPT-4o-mini)를 돌려 각 step에 수학 라벨을 채우고, 두 종류의 페어를 만든다.

- **Type-1 `step_pair`** — 같은 수준 조건·같은 prefix 위에서 `acceptable` step vs 거절된 step (`reject_math` 또는 `reject_persona`)
- **Type-2 `belief_flip_pair`** — `reject_persona`로 거절된 step이 *다른* 수준 조건에서는 `acceptable`인지 cross-check. 확인되면 방향이 반대인 두 페어를 함께 emit

**출력**: `data_pipeline/output/preference_pairs.jsonl` — `pair_type` 필드로 구분

```json
{
  "problem_id": "metamath_42",
  "persona_id": "elem_low",
  "persona_tag": "<elem_low>",
  "prefix_steps": ["Step 1: ...", "Step 2: ..."],
  "step_win":  "Step 3: ...",
  "step_lose": "Step 3: ...",
  "pair_type": "belief_flip_pair",
  "reject_type": "reject_persona",
  "flip_persona_id": "high_high"
}
```

### Stage 3.5 — flip rate 분석 / Type-2 증량

- `data_pipeline_stepdpo/3_5_analyze_flip_rate.py` — Type-1 중 `reject_persona` 비율, Type-2 개수·수준쌍 매트릭스 산출
- `data_pipeline/augment_type2.py` — 후보 수준 제한을 풀고 **모든** 후보에 대해 cross-check를 병렬 수행해 Type-2를 증량 (선택)

---

## 7. Stage 4 — SLC-StepDPO 학습

**파일**: `data_pipeline_stepdpo/4_train_bc_stepdpo.py` + `losses/bc_stepdpo_loss.py`

```
L_Total = -E[log σ(β·Δθ)] + λ_sft · ( -E[log πθ(s_w | x, c, prefix)] )

Δθ = [log πθ(s_w | x,c,prefix) - log π_ref(s_w | x,c,prefix)]
   - [log πθ(s_l | x,c,prefix) - log π_ref(s_l | x,c,prefix)]
```

`step_logprob()`이 prefix 토큰을 제외하고 **step 토큰만** log 확률을 합산한다 — 이게 step-level masking이고, StepDPO의 핵심을 그대로 유지한 부분이다. Type-1·Type-2 모두 같은 `Δθ`로 채점되며 손실 함수는 하나다.

policy는 π_ref + LoRA, reference는 π_ref frozen. `--lambda-sft`로 anchor 가중치를 조절한다.

### 주요 config

| 파일 | 용도 |
|---|---|
| `configs/default.yaml` | 기본 학습 설정. 상단 주석에 ablation toggle 조합표 |
| `configs/bc_full_run.yaml` | 풀스케일 본 실험 |
| `configs/abl_vanilla_dpo.yaml` | ablation — step masking 없음 |
| `configs/abl_step_dpo.yaml` | ablation — StepDPO 재현 (`disable_type2: true`) |
| `configs/abl_type1_only.yaml` | ablation — Type-1만 |
| `configs/bc_smoke.yaml`, `configs/sft_smoke.yaml`, `configs/smoke.yaml` | 소규모 검증 |

toggle 3종의 의미: `disable_step_mask`(step masking 해제), `disable_belief_token`(prompt에서 수준 태그 제거), `disable_type2`(Type-2 페어 제외).

---

## 8. Stage 5 — 평가

| 파일 | 산출 지표 |
|---|---|
| `data_pipeline/5_evaluate.py` | Final Acc. / Step Acc. / Explanation Match |
| `data_pipeline/eval_belief_flip.py` | Belief-Flip |
| `data_pipeline/aggregate_results.py` | `eval/*.json` 취합 → 결과표 이미지 |
| `evaluation/6_check_format.py` | 출력 형식 준수율 |
| `evaluation/7_logprob_analysis.py` | logprob 기반 win rate 분석 |

**judge 모델 구분이 중요하다.** 데이터 구축 단계는 `gpt-4o-mini` + 정규식 보강 + few-shot을 쓰지만, **최종 평가 judge는 반드시 `gpt-4o`** 다. 포스터 Table 1은 gpt-4o judge 기준이다.

`eval_belief_flip.py`는 각 문제에 대해 저수준(`--persona-low`, 기본 `elem_low`)과 고수준(`--persona-high`, 기본 `high_high`) 풀이를 각각 생성한 뒤, ① 저수준 풀이가 저수준에 적합 ② 고수준 풀이가 고수준에 적합 ③ 고수준 풀이가 저수준에는 부적합 — 세 조건을 모두 만족한 문제를 flip 성공으로 센다. 분모가 문제 수(60)라서 1문제가 약 1.7%p다.

---

## 9. 실행

### 전체 파이프라인 (SLURM)

```bash
export OPENAI_API_KEY=sk-...

MAX_ROWS=0 K_SAMPLES=8 \
CONFIG=configs/default.yaml \
OUT=checkpoints/bc_stepdpo \
ADAPTER=checkpoints/sft_qwen3_1.7b_eos \
  sbatch scripts/bc_stepdpo_pipeline_slurm.sh
```

Stage 2.5 머지 → 3a 샘플링 → 3b 페어 → 4 학습을 한 번에 돈다. 스모크는 `MAX_ROWS=12 K_SAMPLES=4 CONFIG=configs/bc_smoke.yaml`.

### λ_sft sweep

```bash
sbatch scripts/run_lambda_sweep_slurm.sh
```

### Ablation 캠페인

```bash
sbatch scripts/ablation_campaign_slurm.sh
```

Vanilla DPO / StepDPO / Type-1-only 학습 → 머지 → 5모델 평가 → 집계까지 자동.

### 기타 스크립트

| 스크립트 | 용도 |
|---|---|
| `scripts/sft_slurm.sh` | Stage 2 SFT 단독 |
| `scripts/build_pairs_full_slurm.sh` | Stage 3 페어 구성 단독 |
| `scripts/run_flip_only.sh` | Belief-Flip 재평가만 |
| `scripts/reeval_flip_heldout_slurm.sh` | held-out flip 재평가 |
| `scripts/fig_table_lambda.py`, `fig_qual_frac.py`, `fig_qual_high.py` | 포스터 그림 생성 |
| `scripts/watchdog_*.sh` | 장기 job 감시 |

> `scripts/*_slurm.sh`는 서버 경로 `~/project/Persona-Step-DPO`를 `cd` 한다. 레포명을 바꿨다면 서버 클론 디렉토리명도 함께 확인할 것.

---

## 10. API 키 운용

`openai_client.py`가 키 failover를 처리한다. `OPENAI_API_KEY`(1순위)가 할당량 소진되면 `OPENAI_API_KEY_FALLBACK`(쉼표로 여러 개)으로 자동 전환한다. 일시적 rate-limit은 호출부 재시도 로직이 담당.

---

## 11. 참고

- 원본 StepDPO 논문: Lai et al., arXiv:2406.18629 — **원저자 표기는 `Step-DPO`(하이픈 포함)**. 본 레포는 포스터 표기를 따라 `StepDPO`로 통일했으니, 외부 인용 시에는 원저자 표기를 쓸 것
- GDPO (belief 변수 도입): Yao et al., ICLR 2025, arXiv:2412.20299
- 교육과정 원본: 2022 개정 수학과 교육과정 (별책 8, 교육부 고시 제2022-33호) → `curriculum/achievement_standards_2022.json`
- 실험 기록: `docs/RESULTS_SUMMARY_2026-06-18.md`, `docs/HANDOFF.md`
