# 방향 예측(5초): 결과와 파일 안내

5초 뒤 중간가가 **상승 / 보합 / 하락** 중 무엇일지 맞히는 과제임. 지표는 macro-F1임. 보합이 약 60%라서 정확도로 보면 "항상 보합"이 유리하기 때문임. 무작위로 찍으면 약 0.33임.
변동성 예측(60초)은 [`VOLATILITY.md`](VOLATILITY.md)에 따로 정리했음.

| 구분 | 위치 |
|---|---|
| 코드 | `src/direction/` (과제 전용), `src/common/` (두 과제 공통: 데이터·피처, Round 2 GBM `exp_v2.py`와 딥러닝 `deep_v2.py`) |
| 결과 기록 | `results/direction/*.jsonl` (파일별 설명: [`results/direction/README.md`](results/direction/README.md)) |
| 모델 가중치 | `results/direction/*.pt`, `*.pkl` (MLPLOB·TLOB 제외) |
| 최종 예측 (정답 라벨 제외) | `results/direction/predictions/` |
| Apple GPU 실행 결과 | `results_gpu/direction/` |

## 1. 라운드별 결과 (macro-F1)

| 단계 | 방법 | 검증 | 테스트 | 자세한 문서 |
|---|---|---|---|---|
| 단순 기준 | 항상 보합 / 모멘텀 | 0.2564 / 0.3486 | 0.2498 / 0.3612 | [`RESULTS.md`](RESULTS.md) |
| Round 1 | 로지스틱 / GBM / DeepLOB / conv+GRU(튜닝) | 0.5012 / 0.5024 / 0.4895 / 0.5109 | 0.4985 / 0.4961 / 0.4782 / 0.5067 | [`RESULTS.md`](RESULTS.md) |
| Round 2 | GBM + conv+GRU 2종(`mt`, `tcn`) 평균, 보합 배수 1.1 | 0.5179 | 0.5172 | 2절 |
| MLPLOB·TLOB | TLOB 128스냅샷, 시드 3개 평균 (같은 행의 conv+GRU 0.5138) | 0.5183 | 0.5094 | [`LOB_MODELS.md`](LOB_MODELS.md) |
| **Round 3** | **모델 8개를 로지스틱 회귀로 결합(stacking)** | **0.5194** | **0.5185** | [`ROUND3.md`](ROUND3.md) |
| Round 5 | 피처 145개 `mt` / 변동성과 공동 학습(`joint`), 시드 0·1 평균 | 0.5086 / 0.5068 | 기준(0.5129) 미달로 하지 않음 | [`VOLATILITY.md`](VOLATILITY.md) 6절 |
| Round 7 | GBM + 시장 전체 정보 피처 (같은 코드의 기준 0.5120) | 0.5118 | 기준 미달로 하지 않음 | [`VOLATILITY.md`](VOLATILITY.md) 8절 |

- 무작위(약 0.33)보다는 확실히 높지만 0.52 근처에서 멈췄음.
- 가장 좋은 것은 Round 3 stacking임. Round 2보다 테스트 +0.0014, 날짜 단위 부트스트랩 95% 구간 [+0.0002, +0.0027]로 유의했음.
- 같은 테스트 10일을 여러 라운드에서 썼으므로 테스트 점수는 다소 낙관적일 수 있음.

## 2. Round 2 요약 (`src/common/exp_v2.py`, `src/common/deep_v2.py`, `src/direction/ens_v2.py`, `src/direction/final_v2.py`)
- 피처를 127개로 늘리고, GBM(학습률 0.02, 잎 15개, 잎당 최소 100행)과 딥러닝 여러 구성(conv+GRU `base`, 보조 정답 `mt`, `tcn` 등)을 검증셋에서 비교했음.
- 결합 우승은 GBM + `mt` 5시드 + `tcn` 5시드의 확률 평균에 보합 배수 1.1을 곱한 것임(검증 0.5179).
- 반반 검증(6일로 고르고 나머지 6일로 평가)에서도 같은 조합이 뽑혔고, 나머지 반쪽에서 0.5187 / 0.5169였음.
- 테스트: 0.5172(정확도 0.5621). 학습+검증 데이터로 다시 학습한 버전은 0.5095로 오히려 낮았음.
- 결과 기록: `results/direction/v2.jsonl`, `v2_deep.jsonl`, `v2_ens.jsonl`, `v2_final.jsonl`.

## 3. 실제로 쓸 수 있나
- 모델이 "5초 뒤 상승"이라고 하면 실제로 오르는 경우는 38.6%임. 아무 때나 찍을 때(19.8%)의 2배지만, 나머지는 그대로이거나 내림.
- 확신할 때만 거래해서 적중률 88%를 만들어도, 수수료·세금·스프레드를 넣은 백테스트는 모든 거래가 손실이었음(거래당 −0.28%). 60~900초로 길게 보유해도 모두 손실이었음([`RESULTS.md`](RESULTS.md) 부록 4~6).

## 4. 재현
```bash
cd src/common
python build_features.py            # 호가 피처·라벨 (두 과제 공통)
python trade_features.py            # 체결 피처
python features_v2.py               # Round 2 피처
python exp_v2.py A B C T L P        # Round 2 GBM·로지스틱 (검증)
python deep_v2.py D1                # Round 2 딥러닝, 구성별 시드 0·1
python deep_v2.py D2                # 상위 3개 구성의 시드 2~4
cd ../direction
python ens_v2.py                    # Round 2 결합 선택 (검증)
python final_v2.py primary          # Round 2 테스트
```
- Round 1은 [`run_all.sh`](run_all.sh), Round 3은 [`ROUND3.md`](ROUND3.md) 8절, MLPLOB·TLOB는 [`LOB_MODELS.md`](LOB_MODELS.md) 7절에 재현 순서가 있음.
- 모든 스크립트는 이미 끝난 작업을 건너뛰고, 결과를 `results/direction/`에 이어 씀.
