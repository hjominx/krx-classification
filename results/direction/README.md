# 결과: 방향 예측 (5초 뒤 상승 / 보합 / 하락)

코드는 `src/direction/`(과제 전용)과 `src/common/`(두 과제 공통)에 있음. 문서는 [`DIRECTION.md`](../../DIRECTION.md)에 정리했음.

## 1. 결과 기록 (`*.jsonl`)

| 단계 | 파일 | 코드 |
|---|---|---|
| Round 1: 모델·설정 선택, 최종 평가 | `select.jsonl`, `final.jsonl`, `deep_test.jsonl`, `deep_*_epochs.jsonl` | `select_models.py`, `final_eval.py`, `deep.py` |
| Round 1: conv+GRU 튜닝·보정·긴 학습 | `tune_gru.jsonl`, `tune_gru_final.jsonl`, `gru_flat.jsonl`, `gru_full.jsonl` | `tune_gru.py`, `gru_flat.py`, `gru_full.py` |
| Round 1: 개선 시도, 체결 데이터, 원시 호가 | `ensemble.jsonl`, `norm_wf.jsonl`, `norm_diag.jsonl`, `trades.jsonl`, `gru_tr.jsonl`, `raw_cmp.jsonl`, `cross.jsonl` | `ensemble.py`, `norm_wf.py`, `norm_diag.py`, `exp_trades.py`, `gru_tr.py`, `raw_cmp.py`, `cross_venue.py` |
| Round 1: 다음 변화 방향, 확신할 때만 답하기, 백테스트 | `next_move.jsonl`, `next_move_sel.jsonl`, `next_move_tr.jsonl`, `backtest.jsonl`, `swing.jsonl` | `next_move*.py`, `backtest.py`, `swing.py` |
| Round 2: 피처 127개, GBM·딥러닝·결합 | `v2.jsonl`, `v2_deep.jsonl`, `v2_ens.jsonl`, `v2_final.jsonl` | `src/common/exp_v2.py`, `src/common/deep_v2.py`, `ens_v2.py`, `final_v2.py` |
| Round 3: 시드 추가, 학습 변형, stacking | `r3_deep.jsonl`, `r3_gbm.jsonl`, `r3_post.jsonl`, `r3_check.jsonl`, `r3_final.jsonl`, `r3_cost.jsonl` | `deep_r3.py`, `gbm_r3.py`, `post_r3.py`, `check_r3.py`, `final_r3.py`, `cost_r3.py` |
| MLPLOB·TLOB 비교 (중단한 방향) | `lob_models.jsonl`, `lob_cost.jsonl`, `cpu_gpu.jsonl` | `lob_models.py`, `lob_cost.py`, `cpu_gpu.py` |

- Apple GPU로 다시 돌린 MLPLOB·TLOB 비교는 `results_gpu/direction/`에 있음.
- Round 5·7에서 방향도 함께 시험한 기록(공동 학습 `joint`, 시장 피처 GBM `dir_mkt`)은 그 라운드 코드가 쓰는 `results/volatility/`에 있음.

## 2. 모델 가중치

| 파일 | 내용 | 불러오는 코드 |
|---|---|---|
| `v2_deep_<구성>_s<시드>.pt` | Round 2 딥러닝 (conv+GRU, `mt`, `tcn` 등) | `src/common/deep_v2.py`의 `make()` |
| `r3_deep_<구성>_s<시드>.pt` | Round 3 딥러닝 (시드 5~9, `mt_st2` 등) | `src/direction/deep_r3.py`, `final_r3.py` |
| `r3_gbm_<구성>.pkl` | Round 3 GBM (`lag`, `rec`) | `src/direction/gbm_r3.py`, `final_r3.py` |

- `.pt`는 `state`(가중치), `mu`·`sd`(학습셋 표준화 값), `cfg`(구성), `cols`(입력 열), `epochs`를 담은 사전임.
- `.pkl`은 학습된 GBM과 검증셋에서 고른 트리 수(`best_iter`)를 담음.
- Round 1·2 GBM과 로지스틱은 같은 설정으로 다시 학습하면 결과가 똑같아서 따로 저장하지 않았음.
- MLPLOB·TLOB 가중치는 올리지 않았음.

## 3. 최종 예측 (`predictions/`)

정답 라벨(`y`)은 원본 데이터에서 만든 것이라 빼고, 모델 출력과 위치 정보만 올렸음. 채점하려면 원본 데이터로 라벨을 다시 만들어야 함.

| 파일 | 단계 | 구간 | 내용 |
|---|---|---|---|
| `test_pred_gru_tuned_H5.npz` | Round 1 conv+GRU (튜닝) | 테스트 | `prob`(하락·보합·상승 확률), `day`, `minute` |
| `v2_test_primary.npz` | Round 2 결합 (GBM + `mt` + `tcn`) | 테스트 | `prob`, `day`, `row` |
| `r3_val_stack_lr.npz` | Round 3 stacking (날짜 하나씩 빼고 학습한 예측) | 검증 | `prob`, `day`, `row` |
| `r3_test.npz` | Round 3 stacking (최종) | 테스트 | `prob`, `day`, `row` |

- `day`는 거래일(YYYYMMDD), `row`는 그날 호가 스냅샷의 순번임.
- 최종 클래스는 확률에 각 라운드의 보합 배수를 곱해 가장 큰 쪽임: Round 2는 [1, 1.1, 1], Round 3는 [1, 1.65, 1.025].
