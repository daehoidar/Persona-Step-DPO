# SLC-StepDPO

**Student Level Conditioned StepDPO for Math Tutor LLMs**
2026 한국인공지능학회 하계학술대회 포스터 발표 (2026.08, 경주)

수학 튜터 LLM이 수학적으로 정확하면서 동시에 학생 수준에 맞는 풀이 단계를 생성하도록 학습하는 선호학습 방법이다. Base 모델은 Qwen3-1.7B(LoRA)이고, 학생 수준은 2022 개정 수학과 교육과정을 따른다.

## Poster

[![SLC-StepDPO poster](docs/poster/SLC-StepDPO_poster_KAIA2026.png)](docs/poster/SLC-StepDPO_poster_KAIA2026.pdf)

## Method

수학 튜터는 두 요건을 동시에 만족해야 한다. 각 풀이 단계가 수학적으로 타당해야 하고, 동시에 학생이 배운 범위의 어휘로 설명해야 한다. 정답이지만 너무 어려운 풀이나, 쉽지만 틀린 풀이는 둘 다 충분하지 않다.

SLC-StepDPO는 StepDPO를 학생 수준 조건 $c$ 로 조건화해 이 두 요건을 하나의 손실로 최적화한다. $c$ 는 학년 세 종류와 성취 두 종류를 조합한 여섯 수준이며, 각 수준은 2022 개정 교육과정에서 유도한 제약 프로파일 $\kappa_c$ 에 대응한다.

문제 $x$, 수준 조건 $c$, 앞선 단계 $s_{1:k-1}$, 선호 단계 $s_w$, 비선호 단계 $s_l$ 에 대해 단계 단위 선호 점수를 다음과 같이 정의한다.

$$\Delta_\theta = \log \frac{\pi_\theta(s_w \mid x, c, s_{1:k-1})}{\pi_{ref}(s_w \mid x, c, s_{1:k-1})} - \log \frac{\pi_\theta(s_l \mid x, c, s_{1:k-1})}{\pi_{ref}(s_l \mid x, c, s_{1:k-1})}$$

전체 손실은 선호 손실과 SFT 앵커의 합이다.

$$\mathcal{L} = -\mathbb{E}\left[\log \sigma(\beta \Delta_\theta)\right] - \lambda_{sft} \mathbb{E}\left[\log \pi_\theta(s_w \mid x, c, s_{1:k-1})\right]$$

여기서 $\pi_{ref}$ 는 SFT로 고정된 기준 모델이고 $\pi_\theta$ 는 학습 대상이다. 앵커 항은 선호 단계에 확률 질량을 붙잡아 정답 정확도가 떨어지는 것을 막으며, $\lambda_{sft}$ 가 그 강도를 조절한다.

선호쌍은 두 단계로 만든다. Type 1은 같은 수준 안에서 적절한 단계와 거절된 단계를 짝지은 것이고, 거절 사유는 수학 오류이거나 수준 불일치다.

Type 2는 Type 1에서 파생된다. Type 1 쌍 가운데 수준 불일치로 거절된 단계만 골라, 그 단계가 다른 수준에서는 적절한지 확인한다. 뒤집힘이 확인되면 방향이 반대인 두 쌍을 함께 만든다. 같은 단계가 원래 수준에서는 비선호이고 다른 수준에서는 선호가 되므로, 승패가 텍스트가 아니라 조건에 의해 결정된다. 수학 오류로 거절된 단계는 어느 수준에서도 적절할 수 없으므로 대상이 아니다. 두 종류 모두 같은 스키마와 같은 손실을 쓴다.

수준별 어휘 제약과 교육과정 근거는 `personas.json`에 정의되어 있다. 데이터 구성 절차는 위 포스터에 정리되어 있다.

## Pipeline

1. MetaMathQA-40K에서 GSM8K 계열 문제 1,500개를 뽑아 여섯 수준에 배정한다.
2. GPT-4o가 각 문제와 수준 조건에 대해 제약 프로파일을 지키는 풀이를 생성한다.
3. 같은 문제의 변형이 학습과 평가에 함께 들어가지 않도록 문제 단위로 나눈다.
4. 생성된 풀이로 Qwen3-1.7B를 미세조정해 기준 모델 $\pi_{ref}$ 를 얻는다.
5. $\pi_{ref}$ 에서 풀이를 여러 개 샘플링하고 각 단계를 적절 또는 거절로 라벨링한다. 거절 사유는 수학 오류와 수준 불일치로 구분한다.
6. 같은 수준 안에서 적절한 단계와 거절된 단계를 짝지어 Type 1 쌍을 만든다.
7. Type 1 쌍 가운데 수준 불일치로 거절된 단계가 다른 수준에서는 적절한지 확인하고, 뒤집힘이 확인되면 방향이 반대인 Type 2 쌍 두 개를 만든다.
8. 두 종류를 합친 집합으로 SLC-StepDPO를 학습하고 GPT-4o 심판으로 네 지표를 측정한다.

선호 데이터 생성에는 GPT-4o-mini를, 최종 평가 심판에는 GPT-4o를 사용했다. 단계별 실행 가이드는 [PIPELINE.md](PIPELINE.md), 파일별 역할은 [CODEMAP.md](CODEMAP.md)에 있다.

## Results

Held-out 평가. 학습/평가 데이터 중복 없음. 360개 풀이(60문제 × 6수준)를 GPT-4o judge로 채점했다.

| Model | Final Acc. | Step Acc. | Explanation Match | Belief-Flip |
|---|:---:|:---:|:---:|:---:|
| SFT (baseline) | 73.9 | 91.5 | 79.5 | 8.3 |
| StepDPO | 72.2 | 90.9 | 79.1 | 10.0 |
| SLC-StepDPO (λ_sft = 0) | 72.8 | 91.6 | **81.7** | 10.0 |
| SLC-StepDPO (λ_sft = 0.01) | 73.3 | **92.0** | 80.9 | 10.0 |
| SLC-StepDPO (λ_sft = 0.03) | **76.1** | 91.4 | 78.6 | **15.0** |

포스터에는 λ_sft = 0.01 설정을 보고했다.

SLC-StepDPO는 비교 대상인 SFT·StepDPO 대비 Step Acc와 Explanation Match에서 우위를 보인다. 다만 **지표별 최고값이 서로 다른 λ_sft에서 나오며, 네 지표를 동시에 최적화하는 단일 설정은 없다.** λ_sft는 정답 정확도(Final Acc)와 수준 적합성(Explanation Match) 사이의 트레이드오프를 결정한다.

Belief-Flip은 60문제 기준 비율이므로 8.3 / 10.0 / 15.0은 각각 5 / 6 / 9문제에 해당한다. 표본이 작아 해석에 주의가 필요하다.

결과 표 그림은 `docs/figures_final/fig_table_lambda.png`, 수준별 풀이 차이의 정성 예시는 `fig_qual_frac.png`(초등, 분수 비유)와 `fig_qual_high.png`(고등, 공식)에 있다.

### Evaluation metrics

| 지표 | 정의 |
|---|---|
| Final Acc. | 최종 정답 일치 비율 |
| Step Acc. | 수학적으로 타당한 step 비율 (전체 step 누적) |
| Explanation Match | 대상 수준의 어휘와 개념을 지킨 step 비율 |
| Belief-Flip | 저수준·고수준 풀이가 각자 수준에 맞으면서 서로 구별되는 비율 |

## 실행

```bash
pip install -r requirements.txt
# 학습 및 평가 상세 절차는 PIPELINE.md 참조
```

핵심 코드는 `data_pipeline_stepdpo/4_train_bc_stepdpo.py`(SLC-StepDPO 학습, `--lambda-sft`로 조절), `data_pipeline/5_evaluate.py`(평가), `scripts/run_lambda_sweep_slurm.sh`(λ_sft sweep)이다.

## Citation

```
ChaeWon An, Minwoo Park, Yejin Yoon, Jihye Kim, Jong-June Jeon.
"Student Level Conditioned StepDPO for Math Tutor LLMs."
2026 한국인공지능학회 하계학술대회, 경주, 2026. 8. (포스터 발표)
```

## Acknowledgments

This research was supported by the National Research Foundation of Korea (NRF) grant funded by the Ministry of Science and ICT (MSIT) (No. RS-2022-NR068754).
