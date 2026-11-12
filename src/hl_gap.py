"""Opening gap vs Hyperliquid (xyz:SKHX perp, USD, 24h). Exploratory, all days with both feeds (no fitting).
FX proxy: KRX mid / HL mid at the last KRX snapshot of the previous trading day.
At the first KRX continuous snapshot of day d (09:00): implied = HL mid (as of t - 500 ms) * FX proxy;
dev_bp = (KRX mid - implied) / implied * 1e4. Then KRX mid return over the next N minutes, and the tradable long
P&L (buy at ask of the snapshot after open, sell at bid N minutes later, commission 0.015% x2 + sell tax 0.20%).
usage: python hl_gap.py -> results/hl_gap.jsonl
"""
import json, os
import numpy as np
import common as C
from bt_util import book, COMM, TAX
HL = os.path.join(C.DATA, 'hl_npz'); LAG = 500 * 10**6
days = [d for d in C.all_days() if os.path.exists(os.path.join(HL, f'{d[4:]}_SKHX.npz'))]
def hl(d):
    z = np.load(os.path.join(HL, f'{d[4:]}_SKHX.npz')); return z['ts'].astype(np.int64) * 10**6, (z['bp'][:, 0] + z['ap'][:, 0]) / 2
def at(ts, v, t):
    j = np.searchsorted(ts, t - LAG, 'right') - 1; return v[j] if j >= 0 and t - LAG - ts[j] < 60 * 10**9 else np.nan
rows, prev = [], None
for d in days:
    z = C.load_day(d); ts = z['ts'].astype(np.int64); mid = z['mid'].astype(float); ap, aq, bp, bq = book(z)
    hts, hm = hl(d)
    if prev is not None:
        pts, pmid, phts, phm = prev
        fx = pmid / at(phts, phm, pts)
        t0 = ts[0]; imp = at(hts, hm, t0) * fx
        hl_ov = (at(hts, hm, t0) / at(phts, phm, pts) - 1) * 1e4        # HL move from prev KRX close to KRX open (bp)
        r = dict(day=d, open_hhmmss_s=round(float((t0 % (86400 * 10**9)) / 1e9 / 3600 + 9), 3), dev_bp=round(float((mid[0] - imp) / imp * 1e4), 1),
                 krx_gap_bp=round(float((mid[0] / pmid - 1) * 1e4), 1), hl_overnight_bp=round(float(hl_ov), 1))
        for N in (1, 5, 15, 30):
            x = np.searchsorted(ts, t0 + N * 60 * 10**9)
            if x >= len(ts): continue
            r[f'ret_{N}m_bp'] = round(float((mid[x] / mid[0] - 1) * 1e4), 1)
            buy, sell = ap[1, 0], bp[x, 0]
            r[f'long_net_{N}m_bp'] = round(float((sell * (1 - COMM - TAX) - buy * (1 + COMM)) / buy * 1e4), 1)
            r[f'hl_ret_{N}m_bp'] = round(float((at(hts, hm, t0 + N * 60 * 10**9) / at(hts, hm, t0) - 1) * 1e4), 1)
        rows.append(r)
    prev = (ts[-1], mid[-1], hts, hm)
with open(os.path.join(C.ROOT, 'results', 'hl_gap.jsonl'), 'w') as f:
    for r in rows: f.write(json.dumps(r) + '\n')
import pandas as pd
df = pd.DataFrame(rows); pd.set_option('display.width', 250); print(df.to_string())
for N in (1, 5, 15, 30):
    c = df[['dev_bp', f'ret_{N}m_bp']].dropna()
    print(N, 'corr(dev, KRX ret)', round(c.corr().iloc[0, 1], 3), 'n', len(c))
print('corr(krx_gap, hl_overnight)', round(df[['krx_gap_bp', 'hl_overnight_bp']].corr().iloc[0, 1], 3))
