"""Selective prediction for the next-change task: the model answers only when it is confident enough.
Targets fixed before running: 0.70, 0.75, 0.80. For each, the confidence threshold is set on VAL as the lowest threshold
whose VAL accuracy is >= the target (= the largest coverage that still meets it). TEST is scored once with that threshold.
Model and settings are those of next_move.py (GBM, 71 book features, VAL-chosen settings), retrained the same way.
usage: python next_move_sel.py -> results/next_move_sel.jsonl
"""
import json, os
import numpy as np
from sklearn.metrics import accuracy_score
import common as C
from select_models import gbm, OUT

TR, VA, TE = C.split_days()
g = max((json.loads(l) for l in open(os.path.join(OUT, 'select.jsonl')) if '"gbm_grid"' in l), key=lambda r: r['mF1'])
HP = dict(learning_rate=g['learning_rate'], max_leaf_nodes=g['max_leaf_nodes'], l2_regularization=g['l2'])


def day_rows(d):                                   # identical to next_move.py
    z = C.load_day(d); mid, seg, ts = z['mid'], z['seg'], z['ts']; n = len(mid)
    chg = np.where(np.diff(mid) != 0)[0] + 1
    k = np.searchsorted(chg, np.arange(n), 'right'); ok = k < len(chg)
    j = chg[np.minimum(k, len(chg) - 1)]; ok &= seg[j] == seg
    t = np.where(ok)[0]; j = j[ok]
    return z['X'][t], (mid[j] > mid[t]).astype(int), (ts[j] - ts[t]) / 1e9, float(z['cov'])


def stack(days, sub=1, eval_only=False):
    X, Y, W, D = [], [], [], []
    for d in days:
        x, y, w, cov = day_rows(d)
        if eval_only and cov < C.MIN_COV: continue
        X.append(x[::sub]); Y.append(y[::sub]); W.append(w[::sub]); D.append(np.full(len(y[::sub]), int(d)))
    return np.concatenate(X), np.concatenate(Y), np.concatenate(W), np.concatenate(D)


Xtr, ytr, _, _ = stack(TR, sub=2)
gb = gbm(**HP).fit(Xtr, ytr); del Xtr
P = {}
for block, days in (('val', VA), ('test', TE)):
    X, y, w, dd = stack(days, eval_only=True); P[block] = (gb.predict_proba(X)[:, 1], y, w, dd)
np.savez(os.path.join(OUT, 'val_pred_next_move_sel.npz'), prob=P['val'][0], y=P['val'][1], day=P['val'][3])
pv, yv, _, _ = P['val']; cv = np.abs(pv - 0.5); ok = ((pv > 0.5).astype(int) == yv)
o = np.argsort(-cv); cum = np.cumsum(ok[o]) / np.arange(1, len(o) + 1)
for target in (0.70, 0.75, 0.80):
    idx = np.where(cum >= target)[0]
    k = idx.max() if len(idx) else None
    rec = dict(target=target)
    if k is None:
        rec['note'] = 'no threshold reaches the target on VAL'
    else:
        thr = float(cv[o][k]); rec.update(threshold=round(thr, 4), val_cov=round((k + 1) / len(o), 4), val_acc=round(float(cum[k]), 4))
        pt, yt, wt, dt = P['test']; m = np.abs(pt - 0.5) >= thr; pr = (pt > 0.5).astype(int)
        rec.update(test_cov=round(float(m.mean()), 4), test_acc=round(float((pr == yt)[m].mean()), 4), test_n=int(m.sum()),
                   test_wait_s_median=round(float(np.median(wt[m])), 2),
                   per_day={str(d): dict(cov=round(float(m[dt == d].mean()), 3), acc=round(float((pr == yt)[m & (dt == d)].mean()), 4)) for d in np.unique(dt)})
    print(json.dumps(rec), flush=True)
    with open(os.path.join(OUT, 'next_move_sel.jsonl'), 'a') as f: f.write(json.dumps(rec) + '\n')
