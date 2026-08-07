# archive

폐기된 파일 보관소. **현재 파이프라인은 여기 있는 어떤 파일도 참조하지 않는다.** 이력 보존 목적으로만 남겨두었으므로, 새로 작업할 때 참고하지 말 것.

| 파일 | 원래 위치 | 폐기 사유 |
|---|---|---|
| `root_app.py` | 루트 `app.py` | Gradio 데모 스텁. 구현 미완 |
| `root_derive_persona_evidence.py` | 루트 `derive_persona_evidence.py` | 교육과정 evidence 자동 주입. `personas.json`에 결과가 반영된 뒤 불필요 |
| `scripts_make_patent_figures.py` | `scripts/` | 특허 도면 생성. 논문 그림과 무관 |
| `tests_README.md` | `tests/README.md` | 구 sanity test 하니스 문서 |
| `tests_run_sft_data.sh` | `tests/` | Phase A 하니스 |
| `tests_run_sft_train.sh` | `tests/` | Phase B 하니스 |
| `tests_run_pairs.sh` | `tests/` | Phase C 하니스 |
| `tests_summarize.py` | `tests/` | REPORT.md 자동 생성기 |
| `tests_persona_compare.py` | `tests/` | 수준별 출력 비교 |
| `tests_smoke_inference.py` | `tests/` | 추론 스모크 테스트 |

구 sanity test 하니스는 파이프라인이 `scripts/*_slurm.sh` 기반으로 재편되면서 사용하지 않게 되었다. 현재 남은 테스트는 `tests/sft_vs_trained.py` 하나다.

> 이 디렉토리는 이전에 `useless/`라는 이름이었다. 그 안에 있던 `data_pipeline_stepdpo_make_pilot_pairs.py`는 `scripts/logprob_slurm.sh`가 실제로 사용 중이어서 `scripts/make_pilot_pairs.py`로 옮겼다.
