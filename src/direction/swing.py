"""Bigger moves: can a long trade held H seconds beat the full cost? (design fixed before running)
Label (per snapshot t, every 10th snapshot): 1 if buying at the best ask of snapshot t+1 and selling at the best bid of the
first snapshot at or after t+1+H (within 5 s of it, same day, before 15:20) is profitable after commission 0.015% per side
and sell tax 0.20%; else 0. H in {60, 300, 900} s.
Features: the 77 kept book+trade features + 8 slower ones (mid return over 1/5/15/30 min, 5-min realised vol in ticks,
return since 09:00, minutes to close, 30-min high-low range position).
Model: GBM (lr 0.05, 15 leaves, l2 1, 300 trees, balanced weights) per H, trained on TRAIN.
Selection on VAL only: for each H and each top-q% confidence cut-off q in {50, 20, 10, 5, 2, 1}, simulate non-overlapping
long trades with 10,000,000 KRW (walk the book, same costs); pick the (H, q) with the largest VAL net P&L among those with
>= 20 trades. TEST is then simulated once with the chosen H and the VAL probability threshold of that q.
usage: python swing.py -> results/direction/swing.jsonl
"""
import _paths  # noqa: F401  (src/common, src/direction, src/volatility, src/report on the import path)
import json, os
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
import common as C
from select_models import OUT
from bt_util import book, fill, CAP, COMM, TAX

HS, QS, SUB, S = (60, 300, 900), (50, 20, 10, 5, 2, 1), 10, 10**9
RES = os.path.join(OUT, 'swing.jsonl'); FT = os.path.join(C.DATA, 'feat_tr')
COLS = next(json.loads(l) for l in open(os.path.join(OUT, 'trades.jsonl')) if '"kept"' in l)['cols']
TR0, VA, TE = C.split_days(); TR = [d for d in TR0 if os.path.exists(os.path.join(FT, f'{d}.npz'))]


def slow(ts, mid):
    f = []
    for m in (1, 5, 15, 30):
        j = np.searchsorted(ts, ts - m * 60 * S, 'left'); f.append((mid - mid[j]) / C.TICK)
    j5 = np.searchsorted(ts, ts - 300 * S, 'left'); dm = np.r_[0, np.diff(mid)] / C.TICK
    c2 = np.r_[0, np.cumsum(dm ** 2)]; f.append(np.sqrt(c2[1:] - c2[j5]))
    f.append((mid - mid[0]) / C.TICK)
    sec = (ts - ts[0]) / S; f.append(np.maximum(22800 - sec, 0) / 60)   # approx minutes to 15:20 from first snapshot
    j30 = np.searchsorted(ts, ts - 1800 * S, 'left')
    lo_, hi_ = np.empty(len(mid)), np.empty(len(mid))
    import collections
    dq_max, dq_min = collections.deque(), collections.deque()
    for i in range(len(mid)):
        while dq_max and mid[dq_max[-1]] <= mid[i]: dq_max.pop()
        dq_max.append(i)
        while dq_min and mid[dq_min[-1]] >= mid[i]: dq_min.pop()
        dq_min.append(i)
        while dq_max[0] < j30[i]: dq_max.popleft()
        while dq_min[0] < j30[i]: dq_min.popleft()
        hi_[i], lo_[i] = mid[dq_max[0]], mid[dq_min[0]]
    f.append((mid - lo_) / np.maximum(hi_ - lo_, C.TICK))
    return np.column_stack(f).astype(np.float32)


def day(d):
    z = C.load_day(d); T = np.load(os.path.join(FT, f'{d}.npz'))['T']
    ts, mid = z['ts'].astype(np.int64), z['mid'].astype(np.float64); n = len(ts)
    X = np.hstack([np.hstack([z['X'], T])[:, COLS], slow(ts, mid)])
    ap, aq, bp, bq = book(z)
    e = np.minimum(np.arange(n) + 1, n - 1)
    out = dict(X=X, ap=ap, aq=aq, bp=bp, bq=bq, ts=ts, cov=float(z['cov']), e=e, x={}, y={})
    for H in HS:
        tgt = ts[e] + H * S; x = np.searchsorted(ts, tgt, 'left')
        ok = (x < n) & (np.arange(n) + 1 < n)
        xx = np.minimum(x, n - 1); ok &= (ts[xx] - tgt) <= 5 * S
        net = bp[xx, 0] * (1 - COMM - TAX) - ap[e, 0] * (1 + COMM)
        out['x'][H] = np.where(ok, xx, -1); out['y'][H] = np.where(ok, (net > 0).astype(int), -1)
    return out


def sim(D, H, probs, thr):
    """non-overlapping long trades on each day, signal when prob >= thr; returns per-trade net P&L list and per-day sums"""
    pnl, perday = [], {}
    for d, z in D.items():
        p = probs[d]; idx = np.arange(0, len(z['ts']), SUB); busy = -1; s = 0.0
        for k, i in enumerate(idx):
            if p[k] < thr or i <= busy: continue
            e, x = z['e'][i], z['x'][H][i]
            if x < 0: continue
            sh = int(CAP // z['ap'][e, 0]); buy = fill(z['ap'][e], z['aq'][e], sh); sell = fill(z['bp'][x], z['bq'][x], sh)
            if buy is None or sell is None: continue
            v = sell - buy - COMM * (buy + sell) - TAX * sell; pnl.append(v); s += v; busy = x
        perday[d] = round(s)
    return pnl, perday


def log(r):
    print(json.dumps(r), flush=True)
    with open(RES, 'a') as f: f.write(json.dumps(r) + '\n')


DATA = {}
def get(days, eval_only):
    out = {}
    for d in days:
        if d not in DATA: DATA[d] = day(d)
        if eval_only and DATA[d]['cov'] < C.MIN_COV: continue
        out[d] = DATA[d]
    return out

DTR, DVA = get(TR, False), get(VA, True)
models, val = {}, []
for H in HS:
    Xs, ys = [], []
    for z in DTR.values():
        i = np.arange(0, len(z['ts']), SUB); m = z['y'][H][i] >= 0; Xs.append(z['X'][i][m]); ys.append(z['y'][H][i][m])
    X, y = np.concatenate(Xs), np.concatenate(ys)
    gb = HistGradientBoostingClassifier(learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0, max_iter=300,
                                        early_stopping=False, class_weight='balanced', random_state=0).fit(X, y)
    models[H] = gb
    pv = {d: gb.predict_proba(z['X'][::SUB])[:, 1] for d, z in DVA.items()}
    yv = np.concatenate([z['y'][H][::SUB] for z in DVA.values()]); allp = np.concatenate(list(pv.values())); mv = yv >= 0
    log(dict(stage='val_auc', H=H, train_rows=int(len(y)), train_pos_rate=round(float(y.mean()), 4),
             val_pos_rate=round(float(yv[mv].mean()), 4), val_auc=round(float(roc_auc_score(yv[mv], allp[mv])), 4)))
    for q in QS:
        thr = float(np.quantile(allp, 1 - q / 100)); pnl, pd_ = sim(DVA, H, pv, thr)
        r = dict(stage='val_sim', H=H, q=q, thr=round(thr, 5), trades=len(pnl), total_krw=round(float(np.sum(pnl))) if pnl else 0,
                 per_trade_krw=round(float(np.mean(pnl))) if pnl else None, win_rate=round(float(np.mean(np.array(pnl) > 0)), 4) if pnl else None,
                 per_day=pd_)
        log(r); val.append(r)
best = max((r for r in val if r['trades'] >= 20), key=lambda r: r['total_krw'])
log(dict(stage='choice', H=best['H'], q=best['q'], thr=best['thr'], val_total_krw=best['total_krw'], val_trades=best['trades']))
DTE = get(TE, True); H = best['H']
pt = {d: models[H].predict_proba(z['X'][::SUB])[:, 1] for d, z in DTE.items()}
pnl, pd_ = sim(DTE, H, pt, best['thr'])
log(dict(stage='test', H=H, q=best['q'], thr=best['thr'], trades=len(pnl), total_krw=round(float(np.sum(pnl))) if pnl else 0,
         per_trade_krw=round(float(np.mean(pnl))) if pnl else None, win_rate=round(float(np.mean(np.array(pnl) > 0)), 4) if pnl else None,
         return_on_capital_pct=round(float(np.sum(pnl)) / CAP * 100, 2) if pnl else 0, per_day=pd_))
