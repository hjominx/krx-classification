#!/bin/bash
# Round-1 pipeline in the order the experiments were run. Every script appends to results/direction or
# results/volatility and skips finished work. Later rounds: see DIRECTION.md and VOLATILITY.md.
# Raw data must be placed as described in data/README.md.
set -e
cd "$(dirname "$0")"

# 1. raw jsonl.gz -> per-day npz
mkdir -p data/krx_npz data/krx_trades
for f in data/krx_raw/*.jsonl.gz; do python preprocess/convert_krx.py "$f" data/krx_npz preprocess; done
python preprocess/convert_trades.py data/krx_trades_raw data/krx_trades preprocess

# 2. features and labels (shared by both tasks)
cd src/common
python build_features.py
python trade_features.py

# 3. direction (5 s): model selection on VAL, then TEST once
cd ../direction
python select_models.py
python final_eval.py
python deep.py gru 5 8 200000 4
python deep.py deeplob 5 6 100000 4
python tune_gru.py tune
python tune_gru.py final

# 4. direction: improvement attempts
python ensemble.py
python norm_wf.py val
python norm_wf.py test
python norm_diag.py
python next_move.py
python next_move_sel.py

# 5. direction: trade data
python exp_trades.py A
python exp_trades.py B
python exp_trades.py C
python exp_trades.py final
python gru_tr.py tune
python next_move_tr.py

# 6. direction: cost-aware backtests
python backtest.py
python swing.py

# 7. direction: flat calibration, full-data training, raw-input comparison
python gru_flat.py
python gru_full.py
python raw_cmp.py

# 8. volatility (60 s)
cd ../volatility
python vol_cls.py labels
python vol_cls.py base
python vol_cls.py gru
python vol_cls.py test_tab
python vol_cls.py test_gru

# 9. figures and tables
cd ../direction
python plots.py
python plot_full.py
cd ../volatility
python plot_vol.py
cd ../report
python make_tables.py

# optional, needs data/hl_npz (Hyperliquid book data); run from src/direction
# python cross_venue.py
# python hl_gap.py
