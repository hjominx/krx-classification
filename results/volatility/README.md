# 결과: 변동성 예측 (다음 60초 잔잔 / 보통 / 격렬)

코드는 `src/volatility/`(과제 전용)와 `src/common/`(두 과제 공통)에 있음. 문서는 [`VOLATILITY.md`](../../VOLATILITY.md)에 정리했음.

## 1. 결과 기록 (`*.jsonl`)

| 단계 | 파일 | 코드 |
|---|---|---|
| Round 1: 정답(RV60)과 3분위 경계, 기본 모델 | `vol_cls.jsonl` | `vol_cls.py` |
| Round 2: 피처 145개, GBM 회귀 + 경계값 | `vol_v2.jsonl` | `vol_v2.py` |
| Round 3: stacking | `vol_r3.jsonl` | `vol_r3.py` |
| Round 4: conv+GRU 회귀 | `vol_r4.jsonl` | `vol_r4.py` |
| Round 5: 보조 정답·공동 학습 등 | `r5.jsonl`, `r5_combine.jsonl` | `r5.py`, `r5_combine.py` |
| Round 6: GBM 회귀 개선 시도 | `gbr_r6.jsonl` | `gbr_r6.py` |
| Round 7: 시장 전체 정보 피처 | `mkt_r7.jsonl` | `src/common/market_feat.py`, `mkt_r7.py` |
| 체결 비용 시뮬레이션 | `exec_sim.jsonl`, `exec_sim2.jsonl` | `exec_sim.py`, `exec_sim2.py` |
| 잔잔 → 격렬 전환 점검 (저장된 예측 분석) | `transition.jsonl` | `transition.py` |

- `r5.jsonl`에는 방향과 변동성을 함께 학습한 모델(`joint`)과 방향 대조군(`mt_allv`)의 방향 결과도 있음.
- `mkt_r7.jsonl`에는 방향 GBM에 시장 피처를 넣은 짝 비교(`dir_base`, `dir_mkt`)도 있음.

## 2. 모델 가중치

| 파일 | 내용 | 불러오는 코드 |
|---|---|---|
| `vol_v2_gru_s<시드>.pt` | Round 2 conv+GRU 분류 | `vol_v2.py`의 `gru_apply()` |
| `vol_r4_convgru_reg_s<시드>.pt` | Round 4 conv+GRU 회귀 | `vol_r4.py` |
| `r5_<구성>_s<시드>_<vol 또는 dir>.pt` | Round 5 conv+GRU (`vreg_mh`, `vreg_lr`, `joint` 등) | `r5.py`, `r5_combine.py`의 `test_bag()` |
| `r6_gbr_<구성>.pkl` | Round 6 GBM 회귀 변형 | `gbr_r6.py` |
| `r7_<구성>.pkl` | Round 7 GBM (시장 피처 있음·없음, 변동성 `vol_*`·방향 `dir_*`) | `mkt_r7.py` |

- `.pt`는 `state`(가중치), `mu`·`sd`(학습셋 표준화 값)를 담고, 회귀 모델은 정답 표준화 값(`zm`, `zs`)도 담음. Round 5는 `cfg`, 출력 개수(`nout`), 변동성 시간(`hv`)도 담음.
- `.pkl`은 학습된 GBM과 검증셋에서 고른 트리 수(`best_iter`)를 담음.
- Round 2 GBM 회귀는 같은 설정으로 다시 학습하면 Round 2 테스트 예측과 100% 같아서 따로 저장하지 않았음.

## 3. 최종 예측 (`predictions/`)

정답 라벨(`y`)은 원본 데이터에서 만든 것이라 빼고, 모델 출력과 위치 정보만 올렸음. 채점하려면 원본 데이터로 라벨을 다시 만들어야 함.
클래스는 0 = 잔잔, 1 = 보통, 2 = 격렬임.

| 파일 | 단계 | 구간 | 내용 |
|---|---|---|---|
| `test_pred_vol_gru.npz` | Round 1 conv+GRU | 테스트 | `prob`, `day`, `minute` |
| `vol_v2_val_gbr.npz` | Round 2 GBM 회귀 | 검증 | `pred`(예측한 log(1 + RV60)), `day`, `row` |
| `vol_v2_test_train.npz` | Round 2 GBM 회귀 + 경계값 [2.0, 2.85] | 테스트 | `pred`(클래스), `day`, `row` |
| `vol_r3_val_stack_gbm.npz` | Round 3 stacking (날짜 하나씩 빼고 학습한 예측) | 검증 | `prob`, `day`, `row` |
| `vol_r3_test.npz` | Round 3 stacking | 테스트 | `prob`, `pred`(클래스), `day`, `row` |
| `vol_r4_test.npz` | Round 4 혼합 | 테스트 | `x`(혼합한 log(1 + RV60)), `pred`(클래스), `day`, `row` |
| `r5_vol_test.npz` | Round 5 혼합 | 테스트 | `x`, `pred`, `day`, `row` |
| `r7_val_vol_mkt.npz` | Round 7 시장 피처 GBM 회귀 | 검증 | `pred`(예측한 log(1 + RV60)), `day`, `row` |
| `r7_vol_test.npz` | Round 7 혼합 | 테스트 | `x`, `pred`, `day`, `row` |

- `day`는 거래일(YYYYMMDD), `row`는 그날 호가 스냅샷의 순번임.
