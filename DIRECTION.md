# 방향 예측(5초): 결과와 파일 안내

5초 뒤 중간가가 **상승 / 보합 / 하락** 중 무엇일지 맞히는 과제임. 지표는 macro-F1임. 보합이 약 60%라서 정확도로 보면 "항상 보합"이 유리하기 때문임. 무작위로 찍으면 약 0.33임.
변동성 예측(60초)은 [`VOLATILITY.md`](VOLATILITY.md)에 따로 정리했음.

| 구분 | 위치 |
|---|---|
| 코드 | `src/direction/` (과제 전용, Round 2 GBM `exp_v2.py`와 딥러닝 `deep_v2.py` 포함), `src/common/` (두 과제 공통: 데이터·피처, `data_v2.py`, `seq_models.py`) |
| 결과 기록 | `results/direction/*.jsonl` (파일별 설명: [`results/direction/README.md`](results/direction/README.md)) |
| 모델 가중치 | `results/direction/*.pt`, `*.pkl` (MLPLOB·TLOB 제외) |
| 최종 예측 (정답 라벨 제외) | `results/direction/predictions/` |
| Apple GPU 실행 결과 | `results_gpu/direction/` |

## 1. 라운드별 결과 (macro-F1)

| 단계 | 방법 | 검증 | 테스트 | 자세한 문서 |
|---|---|---|---|---|
| 단순 기준 | 항상 보합 / 모멘텀 | 0.2564 / 0.3486 | 0.2498 / 0.3612 | [`RESULTS.md`](RESULTS.md) |
| Round 1 | 로지스틱 / GBM(튜닝) / DeepLOB / conv+GRU(튜닝) | 0.5012 / 0.5069 / 0.4895 / 0.5109 | 0.4985 / 0.4961 / 0.4782 / 0.5067 | [`RESULTS.md`](RESULTS.md) |
| Round 2 | GBM + conv+GRU(`mt`) + TCN(`tcn`) 평균, 보합 배수 1.1 | 0.5179 | 0.5172 | 2절 |
| MLPLOB·TLOB | TLOB 128스냅샷, 시드 3개 평균 (같은 행의 conv+GRU 0.5138) | 0.5183 | 0.5094 | [`LOB_MODELS.md`](LOB_MODELS.md) |
| **Round 3** | **모델 8개를 로지스틱 회귀로 결합(stacking)** | **0.5194** | **0.5185** | [`ROUND3.md`](ROUND3.md) |
| Round 5 | 피처 145개 `mt` / 변동성과 공동 학습(`joint`), 시드 0·1 평균 | 0.5086 / 0.5068 | 기준(0.5129) 미달로 하지 않음 | [`VOLATILITY.md`](VOLATILITY.md) 6절 |
| Round 7 | GBM + 시장 전체 정보 피처 (같은 코드의 기준 0.5120) | 0.5118 | 기준 미달로 하지 않음 | [`VOLATILITY.md`](VOLATILITY.md) 8절 |
| Round 8 | conv+GRU(`mt`) 층 폭 6가지, 시드 0·1 평균 (기준 64×64 0.51290) | 0.51175–0.51285 | 기준 미달로 하지 않음 | 3절 |

- 무작위(약 0.33)보다는 확실히 높지만 0.52 근처에서 멈췄음.
- 가장 좋은 것은 Round 3 stacking임. Round 2보다 테스트 +0.0014, 날짜 단위 부트스트랩 95% 구간 [+0.0002, +0.0027]로 유의했음.
- 같은 테스트 10일을 여러 라운드에서 썼으므로 테스트 점수는 다소 낙관적일 수 있음.

## 2. Round 2 요약 (`src/direction/exp_v2.py`, `deep_v2.py`, `ens_v2.py`, `final_v2.py`)
- 피처를 127개로 늘리고, GBM(학습률 0.02, 잎 15개, 잎당 최소 100행)과 딥러닝 여러 구성(conv+GRU `base`, 보조 정답 `mt`, `tcn` 등)을 검증셋에서 비교했음.
- 결합 우승은 GBM + `mt` 5시드 + `tcn` 5시드의 확률 평균에 보합 배수 1.1을 곱한 것임(검증 0.5179).
- 반반 검증(6일로 고르고 나머지 6일로 평가)에서도 같은 조합이 뽑혔고, 나머지 반쪽에서 0.5187 / 0.5169였음.
- 테스트: 0.5172(정확도 0.5621). 학습+검증 데이터로 다시 학습한 버전은 0.5095로 오히려 낮았음.
- 결과 기록: `results/direction/v2.jsonl`, `v2_deep.jsonl`, `v2_ens.jsonl`, `v2_final.jsonl`.

## 3. Round 8: conv+GRU 층 폭 바꾸기 (`src/direction/width_r8.py`)
Round 2·3에서 쓴 conv+GRU `mt`는 합성곱 2층의 채널 수와 GRU 칸 수가 모두 64임(64×64). 이 폭을 6가지로 바꿔 봤음. 변동성 분류도 같은 방식으로 확인했음([`VOLATILITY.md`](VOLATILITY.md) 12절).

- 설계는 실행 전에 고정했고, 검증셋에서만 골랐음. 폭 말고는 Round 2 `mt`와 같음(피처 127개, 스냅샷 50개, 보조 정답 1·2·10·30초, EMA, 학습 예산, 조기 종료).
- 통과 기준: 시드 0·1 평균 검증 macro-F1이 Round 2 `mt` 시드 0·1 평균(0.5129)보다 높을 것. 먼저 바꾼 코드로 64×64 시드 0을 다시 돌려 Round 2와 같은 0.5135가 나오는 것을 확인했음.
- 통과하면 최고 폭에 시드 2~9를 더 돌려 Round 3 stacking의 `mt` 자리에 넣고, Round 3(검증 0.5194)보다 높을 때만 테스트를 한 번 보기로 했음.

| 폭 (채널 × 칸) | 가중치 수 | 시드 0 | 시드 1 | 평균 | 결과 | 1 에폭 시간 (초, 시드 0) |
|---|---|---|---|---|---|---|
| 64×64 (Round 2 `mt`) | 62,735 | 0.5135 | 0.5123 | 0.51290 | 기준 | 61 |
| 32×32 | 22,159 | 0.5118 | 0.5129 | 0.51235 | 탈락 | 30 |
| 32×64 | 35,119 | 0.5132 | 0.5103 | 0.51175 | 탈락 | 43 |
| 64×32 | 46,703 | 0.5133 | 0.5124 | 0.51285 | 탈락 | 43 |
| 64×128 | 113,231 | 0.5135 | 0.5122 | 0.51285 | 탈락 | 95 |
| 128×64 | 136,399 | 0.5128 | 0.5116 | 0.51220 | 탈락 | 94 |
| 128×128 | 199,183 | 0.5125 | 0.5117 | 0.51210 | 탈락 | 133 |

- 통과한 폭이 없어서 시드 추가, stacking 비교, 테스트는 하지 않았음. 방향 최종은 그대로 Round 3 stacking임.
- 가장 가까운 64×32와 64×128도 기준보다 0.00005 낮았음. 모든 폭의 평균이 0.51175~0.51285 사이로, 같은 폭 안에서 시드만 바꿨을 때의 차이(32×64에서 0.0029)보다 좁은 범위임.
- 크게 해도(가중치 약 3.2배) 나아지지 않았고, 작게 해도(약 0.35배) 크게 나빠지지 않았음. Round 1에서 GRU 칸 수만 32·128로 바꿔 본 결과(0.506 / 0.508, 기본값 시드 3개는 0.506–0.511, [`RESULTS.md`](RESULTS.md) 부록)와 같은 결론임.
- 1 에폭은 학습 창 20만 개와 검증 확인을 합한 시간이고, CPU 2스레드 기준임.
- 결과 기록: `results/direction/width_r8.jsonl`, 가중치 `results/direction/r8_deep_mt_c<채널>_h<칸>_s<시드>.pt`, 표 `report/tables/20_width_sweep_val.csv`, 그림 `figures/29_width_sweep.png`.

## 4. 실제로 쓸 수 있나
- 모델이 "5초 뒤 상승"이라고 하면 실제로 오르는 경우는 38.6%임. 아무 때나 찍을 때(19.8%)의 2배지만, 나머지는 그대로이거나 내림.
- 확신할 때만 거래해서 적중률 88%를 만들어도, 수수료·세금·스프레드를 넣은 백테스트는 모든 거래가 손실이었음(거래당 −0.28%). 60~900초로 길게 보유해도 모두 손실이었음([`RESULTS.md`](RESULTS.md) 부록 4~6).

## 5. 재현
```bash
cd src/common
python build_features.py            # 호가 피처·라벨 (두 과제 공통)
python trade_features.py            # 체결 피처
python features_v2.py               # Round 2 피처
cd ../direction
python exp_v2.py A B C T L P        # Round 2 GBM·로지스틱 (검증)
python deep_v2.py D1                # Round 2 딥러닝, 구성별 시드 0·1
python deep_v2.py D2                # 상위 3개 구성의 시드 2~4
python ens_v2.py                    # Round 2 결합 선택 (검증)
python final_v2.py primary          # Round 2 테스트
python width_r8.py screen           # Round 8: conv+GRU 폭 6가지, 시드 0·1 (검증)
python width_r8.py more             # Round 8: 통과한 폭의 시드 2~9 (이번에는 해당 없음)
python width_r8.py stack            # Round 8: Round 3 stacking에 넣어 비교 (이번에는 해당 없음)
```
- Round 1은 [`run_all.sh`](run_all.sh), Round 3은 [`ROUND3.md`](ROUND3.md) 8절, MLPLOB·TLOB는 [`LOB_MODELS.md`](LOB_MODELS.md) 7절에 재현 순서가 있음.
- 모든 스크립트는 이미 끝난 작업을 건너뛰고, 결과를 `results/direction/`에 이어 씀.
- 방향 코드는 변동성 코드를 불러오지 않고 `src/common/`만 씀. `exp_v2.py`와 `deep_v2.py`는 Round 2 실험 단계만 담고, 변동성 코드와 함께 쓰는 부분(피처 열·불러오기·GBM 설정, conv+GRU 등 모델과 학습 예산)은 `src/common/data_v2.py`, `src/common/seq_models.py`에서 가져옴.
