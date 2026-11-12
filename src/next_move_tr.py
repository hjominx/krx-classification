"""[next_move_tr.py] same task as next_move.py, on days with trade data, comparing book-only (71) with book + trades
(columns kept in exp_trades.py stage B). GBM settings as in next_move.py (fixed in advance).
Label: direction of the next mid-price change (up vs down), as defined in next_move.py.
usage: python next_move_tr.py -> results/next_move_tr.jsonl
"""
import json, os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score
import common as C
from select_models import gbm, OUT

TR, VA, TE = C.split_days()
FT = os.path.join(C.DATA, 'feat_tr'); TR = [d for d in TR if os.path.exists(os.path.join(FT, f'{d}.npz'))]
COLS = next(json.loads(l) for l in open(os.path.join(OUT, 'trades.jsonl')) if '"name": "kept"' in l)['cols']
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
    XT = np.hstack([z['X'], np.load(os.path.join(FT, f'{d}.npz'))['T']])
    return XT[t], y, wait, float(z['cov'])


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
sets = {'book71': list(range(71)), 'book+trades': COLS}
models = {k: gbm(**HP).fit(Xtr[:, c], ytr) for k, c in sets.items()}
i_imb = names.index('imb1')
for block, days in (('val', VA), ('test', TE)):
    X, y, w, dd = stack(days, eval_only=True)
    res = dict(block=block, n=int(len(y)), up_share=round(float(y.mean()), 4), majority=sc(y, np.full_like(y, maj)),
               imb1_rule=sc(y, np.where(X[:, i_imb] > 0, 1, np.where(X[:, i_imb] < 0, 0, maj))))
    for k, c in sets.items():
        prob = models[k].predict_proba(X[:, c]); p = prob.argmax(1); conf = np.abs(prob[:, 1] - 0.5)
        res[k] = dict(**sc(y, p), per_day_acc={str(q): round(accuracy_score(y[dd == q], p[dd == q]), 4) for q in np.unique(dd)},
                      top_conf_acc={f'{int(q * 100)}%': round(float((p == y)[conf >= np.quantile(conf, 1 - q)].mean()), 4) for q in (1.0, 0.5, 0.2, 0.1, 0.01)})
        if block == 'test': np.savez(os.path.join(OUT, f'test_pred_next_move_tr_{k}.npz'), prob=prob, y=y, day=dd)
    print(json.dumps(res), flush=True)
    with open(os.path.join(OUT, 'next_move_tr.jsonl'), 'a') as f: f.write(json.dumps(res) + '\n')
