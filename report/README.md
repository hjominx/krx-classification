# 결과물 모음

발표 자료, 결과 표, 그래프를 한곳에 모았습니다. 표와 그래프의 숫자는 모두 `results/`의 실험 결과 파일에서 코드로 만든 것입니다(손으로 입력한 숫자 없음). 원본 데이터는 포함되어 있지 않습니다.

## 1. 발표 자료 (`presentation/`)

| 파일 | 내용 |
|---|---|
| `발표자료.pdf` | 발표 슬라이드 19장 (16:9) |
| `발표자료.html` | 같은 슬라이드를 브라우저에서 넘기며 볼 수 있는 파일. ← → 키, 클릭, 스와이프로 이동, N 키로 발표 대본 표시 |
| `발표대본.pdf` | 슬라이드별 발표 대본, 처음 나오는 용어 설명, 예상 질문과 답 14개 |
| `프로젝트_요약.pdf` | 프로젝트 전체를 2쪽으로 요약한 문서 |

## 2. 결과 표 (`tables/`)

`src/make_tables.py`가 `results/*.jsonl`에서 만든 CSV입니다. 엑셀에서 바로 열 수 있습니다(UTF-8 BOM).

| 파일 | 내용 | 데이터 구간 |
|---|---|---|
| `01_direction_5s_test.csv` | 5초 뒤 방향 분류(상승/보합/하락), 모델별 macro-F1과 정확도 | 테스트 |
| `02_horizon_val.csv` | 예측 시간 1·2·5·10·30초별 모델 성능 (예측 시간 선택에 사용) | 검증 |
| `03_horizon_test.csv` | 예측 시간별 최종 성능 | 테스트 |
| `04_gru_tuning_val.csv` | conv+GRU 하이퍼파라미터를 하나씩 바꾼 실험 13개 | 검증 |
| `05_full_training_curve.csv` | 학습 데이터 전체로 학습할 때 epoch별 학습 손실과 검증 성능 | 검증 |
| `06_trade_features_val.csv` | 체결 피처 추가(A), 피처 그룹별 기여도(B), GBM 설정(C) | 검증 |
| `07_raw_input_comparison.csv` | 원시 호가 40개만 넣었을 때 모델 비교와 GRU와의 차이(날짜 단위 bootstrap 95% 구간) | 검증·테스트 |
| `08_volatility_60s.csv` | 다음 60초 변동성 분류(잔잔/보통/격렬), 모델별 성능 | 검증·테스트 |
| `09_volatility_confusion_gru_test.csv` | 변동성 분류 conv+GRU의 혼동행렬(건수) | 테스트 |
| `10_selective_next_move.csv` | 다음 가격 변화 방향을 확신할 때만 답할 때의 목표 적중률별 결과 | 검증·테스트 |
| `11_backtest_test.csv` | 확신 기준(85%) 신호로 거래했을 때 비용 가정별 손익 | 테스트 |

### 주요 결과

| 문제 | 최고 모델 | 테스트 macro-F1 | 비교 |
|---|---|---|---|
| 5초 뒤 방향 (3클래스) | conv+GRU (튜닝) | 0.507 | 무작위 약 0.33, 항상 보합 0.25, 모멘텀 0.36 |
| 60초 변동성 (3클래스) | GBM / conv+GRU | 0.655 / 0.654 | 직전 60초 클래스 그대로 0.601 |
| 다음 변화 방향 (2클래스, 확신 상위 약 4%) | GBM | 정확도 0.865 | 항상 답하면 0.66~0.67 |

## 3. 그래프 (`../figures/`)

| 파일 | 내용 |
|---|---|
| `01_data_coverage.png` | 날짜별 수집 스냅샷 수와 학습/검증/테스트 구분 |
| `02_horizon_val.png` | 예측 시간별 검증 macro-F1 |
| `03_horizon_curve.png` | 예측 시간별 테스트 macro-F1 (모델별) |
| `04_feature_ablation_val.png` | 피처 그룹을 하나씩 더할 때 검증 성능 |
| `05_gbm_grid_val.png` | GBM 하이퍼파라미터 8개 조합의 검증 성능 |
| `06_test_per_day.png` | 테스트 날짜별 macro-F1 (모델별) |
| `07_confidence_coverage.png` | 확신도 상위 비율별 적중률 |
| `08_time_of_day.png` | 30분 단위 시간대별 테스트 성능 |
| `09_confusion_gbm_5s.png` | GBM 5초 방향 분류 혼동행렬 |
| `10_cross_venue.png` | 해외 24시간 파생상품 호가를 추가한 탐색 실험 (참고용) |
| `11_deep_val_curves.png` | conv+GRU와 DeepLOB의 epoch별 검증 성능 |
| `12_gru_tuning_val.png` | conv+GRU 하이퍼파라미터 실험 |
| `13_class_weight_tradeoff.png` | 클래스 가중치 유무에 따른 정확도와 macro-F1 |
| `14_next_move_test.png` | 다음 가격 변화 방향 분류 결과와 확신도별 정확도 |
| `15_norm_rolling_val.png` | 가격 수준 정규화와 주기적 재학습 실험 |
| `16_trade_groups_val.png` | 체결 피처 그룹별 기여도 |
| `17_trades_test.png` | 체결 피처 추가 전후 테스트 비교 (예측 시간별, 날짜별) |
| `18_vol_confusion.png` | 변동성 분류 conv+GRU 혼동행렬 (비율) |
| `19_full_training.png` | 전체 데이터 학습 시 학습 손실과 검증 성능 (과적합) |

## 4. 용어

| 용어 | 뜻 |
|---|---|
| 호가 / 체결 | 사고팔려는 주문 목록 / 실제로 이루어진 거래 |
| 중간가 | 가장 비싼 매수 호가와 가장 싼 매도 호가의 평균 |
| 틱 | 가격의 최소 변동 단위 (이 종목은 1,000원) |
| macro-F1 | 클래스별 F1 점수의 단순 평균. 드문 클래스도 같은 비중으로 평가 |
| 검증 / 테스트 | 설정을 고르는 구간 / 마지막에 한 번만 평가하는 구간 |
| bootstrap 95% 구간 | 테스트 날짜를 다시 뽑아 계산한 차이의 범위. 0을 포함하지 않으면 우연이 아니라고 봄 |

## 다시 만들기

```bash
cd src
python make_tables.py   # report/tables/*.csv
python plots.py         # figures/*.png
python plot_vol.py
python plot_full.py
```
