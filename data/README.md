# 데이터

원본 데이터는 증권사 API에서 직접 수집한 것으로, 이 저장소에는 포함하지 않음. 아래 위치에 두면 코드가 그대로 동작함.

| 경로 | 내용 | 만드는 방법 |
|---|---|---|
| `data/krx_raw/` | KIS H0STASP0(호가) 원본 `*.jsonl.gz` | 직접 수집 |
| `data/krx_trades_raw/` | KIS H0STCNT0(체결) 원본 `YYYY-MM-DD.jsonl.gz` | 직접 수집 |
| `data/krx_npz/KRX_YYYYMMDD.npz` | 하루 단위 호가 배열 | `preprocess/convert_krx.py` |
| `data/krx_trades/YYYYMMDD.npz` | 하루 단위 체결 배열 | `preprocess/convert_trades.py` |
| `data/feat/`, `data/feat_tr/`, `data/feat_vol/` | 피처·라벨 캐시 | `src/build_features.py`, `src/trade_features.py`, `src/vol_cls.py labels` |
| `data/hl_npz/` | (선택) Hyperliquid 호가, 교차시장·갭 실험에만 사용 | `preprocess/convert.py` |
| `data/feat_vol_mh/` | 10·30·120초 변동성 정답 (Round 5) | `src/r5.py labels` |
| `data/market/` | 공개 시장 데이터: Yahoo Finance 시간봉·일봉 JSON (Round 7) | `src/market_feat.py fetch` |
| `data/feat_mkt/` | 시장 전체 정보 피처 24개 (Round 7) | `src/market_feat.py build` |

## 원본 형식

- 한 줄이 웹소켓 메시지 하나인 JSON Lines이고, 수집기가 주기적으로 flush 하는 multi-member gzip임.
- 각 줄에는 수신 시각 `recv_ts_ns`(나노초)와 KIS 원본 문자열 `raw_json.raw_message`가 있음.
- `raw_message`는 `0|TR_ID|건수|필드^필드^...` 형식임. 한 메시지에 여러 건이 묶여 오면 필드 묶음이 건수만큼 반복됨.
- 손상된 gzip member는 `preprocess/convert.py`의 `members()`가 다음 헤더로 건너뛰며 복구함.
