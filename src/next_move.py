"""Experiment 6 (a different task): direction of the NEXT mid-price change, in event time (up vs down, no 'flat').
Label at snapshot t: sign of mid[j] - mid[t] for the first j > t in the same segment with mid[j] != mid[t];
rows with no later change in the segment are dropped. This says nothing about WHEN the move comes.
Same 71 features, same day split. Models fixed in advance (no selection on TEST):
  majority, sign(imb1) rule, sign(micro_dev) rule, logistic regression (balanced), GBM (grid-selected settings).
usage: python next_move.py -> results/next_move.jsonl, results/test_pred_next_move_gbm.npz
"""
import json, os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score
import common as C
from select_models import gbm, OUT

TR, VA, TE = C.split_days()
names = list(C.load_day(TR[0])['names'])
g = max((json.loads(l) for l in open(os.path.join(OUT, 'select.jsonl')) if '"gbm_grid"' in l), key=lambda r: r['mF1'])
HP = dict(learning_rate=g['learning_rate'], max_leaf_nodes=g['max_leaf_nodes'], l2_regularization=g['l2'])


def day_rows(d):
    z = C.load_day(d); mid, seg, ts = z['mid'], z['seg'], z['ts']; n = len(mid)
    chg = np.where(np.diff(mid) != 0)[0] + 1
    k = np.searchsorted(chg, np.arange(n), 'right'); ok = k < len(chg)
    j = chg[np.minimum(k, len(chg) - 1)]; ok &= seg[j] == seg
    t = np.where(ok)[0]; j = j[ok]
    y = (mid[j] > mid[t]).astype(int)            # 1 = up, 0 = down
    wait = (ts[j] - ts[t]) / 1e9
    return z['X'][t], y, wait, float(z['cov'])


def stack(days, sub=1, eval_only=False):
    X, Y, W, D = [], [], [], []
    for d in days:
        x, y, w, cov = day_rows(d)
        if eval_only and cov < C.MIN_COV: continue
        X.append(x[::sub]); Y.append(y[::sub]); W.append(w[::sub]); D.append(np.full(len(y[::sub]), int(d)))
    return np.concatenate(X), np.concatenate(Y), np.concatenate(W), np.concatenate(D)


def sc(y, p):
    return dict(acc=round(accuracy_score(y, p), 4), mF1=round(f1_score(y, p, average='macro'), 4))


Xtr, ytr, _, _ = stack(TR, sub=2)
maj = int(np.bincount(ytr).argmax())
mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
lr = LogisticRegression(max_iter=500, class_weight='balanced').fit(((Xtr - mu) / sd)[::2], ytr[::2])
gb = gbm(**HP).fit(Xtr, ytr)
i_imb, i_mic = names.index('imb1'), names.index('micro_dev')
for block, days in (('val', VA), ('test', TE)):
    X, y, w, dd = stack(days, eval_only=True)
    rule = lambda v: np.where(v > 0, 1, np.where(v < 0, 0, maj))
    prob = gb.predict_proba(X); pg = prob.argmax(1)
    res = dict(block=block, n=int(len(y)), up_share=round(float(y.mean()), 4),
               wait_s_median=round(float(np.median(w)), 2), wait_s_p90=round(float(np.quantile(w, 0.9)), 2),
               majority=sc(y, np.full_like(y, maj)), imb1_rule=sc(y, rule(X[:, i_imb])), micro_rule=sc(y, rule(X[:, i_mic])),
               logreg=sc(y, lr.predict((X - mu) / sd)), gbm=dict(**sc(y, pg), iters=int(gb.n_iter_),
               per_day_acc={str(k): round(accuracy_score(y[dd == k], pg[dd == k]), 4) for k in np.unique(dd)}))
    conf = np.abs(prob[:, 1] - 0.5)
    res['gbm_top_conf_acc'] = {f'{int(q * 100)}%': round(float((pg == y)[conf >= np.quantile(conf, 1 - q)].mean()), 4) for q in (1.0, 0.5, 0.2, 0.1, 0.01)}
    if block == 'test': np.savez(os.path.join(OUT, 'test_pred_next_move_gbm.npz'), prob=prob, y=y, day=dd, wait=w)
    print(json.dumps(res), flush=True)
    with open(os.path.join(OUT, 'next_move.jsonl'), 'a') as f: f.write(json.dumps(res) + '\n')
