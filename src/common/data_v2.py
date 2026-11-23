"""Shared code on the v2 features, used by both tasks: column groups and feature sets, day loading, the GBM settings
chosen on VAL (exp_trades.py stage C) and macro-F1 helpers.
Columns: 160 per snapshot = book / flow / time (build_features.py) + trades (trade_features.py) + v2 (features_v2.py).
  REF77  = book 47 + flow 15 + time 3 + trade_tw 12 (the set kept in exp_trades.py)
  REFC77 = the same with 'flowc' (causal depth scale) in place of 'flow'
  FULL   = REFC77 + hist, rvc, sprd, ofic, queue, depl, ctx, daylvl (127 columns)
Days: those with a trade file (07-09 has none): TRAIN 46, VAL 12.
Taken out of the round-2 direction script (src/direction/exp_v2.py) without changes.
"""
import _paths  # noqa: F401  (src/common, src/direction, src/volatility, src/report on the import path)
import os
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier as HGB
from sklearn.metrics import f1_score, accuracy_score
import common as C

FT, F2 = os.path.join(C.DATA, 'feat_tr'), os.path.join(C.DATA, 'feat2')
H = 5; hi = C.HORIZONS.index(H)
TR0, VA, TE = C.split_days()
has_tr = lambda d: os.path.exists(os.path.join(FT, f'{d}.npz'))
TR = [d for d in TR0 if has_tr(d)]
MS = (0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3)                       # flat-probability multipliers tried on VAL


def columns():
    z = C.load_day(TR[0]); t = np.load(os.path.join(FT, f'{TR[0]}.npz')); v = np.load(os.path.join(F2, f'{TR[0]}.npz'))
    names = list(z['names']) + list(t['names']) + list(v['names'])
    G = {g: list(ix) for g, ix in C.groups_of(list(z['names'])).items()}
    for g in set(t['groups']): G['trade_' + g] = [71 + k for k, gg in enumerate(t['groups']) if gg == g]
    for g in set(v['groups']): G[g] = [71 + len(t['names']) + k for k, gg in enumerate(v['groups']) if gg == g]
    return names, G


NAMES, G = columns()
NEW = ['hist', 'rvc', 'sprd', 'ofic', 'queue', 'depl', 'ctx', 'daylvl']
cols_of = lambda gs: sorted(sum((G[g] for g in gs), []))
REF77, REFC77 = cols_of(['book', 'flow', 'time', 'trade_tw']), cols_of(['book', 'flowc', 'time', 'trade_tw'])
FULL = cols_of(['book', 'flowc', 'time', 'trade_tw'] + NEW)


def load_day(d, need_valid=True):
    """all 160 columns of one day; rows with a valid 5 s label"""
    z = C.load_day(d); v = z['V'][:, hi]
    X = np.hstack([z['X'], np.load(os.path.join(FT, f'{d}.npz'))['T'], np.load(os.path.join(F2, f'{d}.npz'))['X2']]).astype(np.float32)
    r = np.where(v)[0] if need_valid else np.arange(len(v))
    return X[r], z['Y'][r, hi].astype(int), r, float(z['cov'])


def stack(days, sub=1, eval_only=False):
    X, Y, D, R = [], [], [], []
    for d in days:
        x, y, r, cov = load_day(d)
        if eval_only and cov < C.MIN_COV: continue
        X.append(x[::sub]); Y.append(y[::sub]); R.append(r[::sub]); D.append(np.full(len(y[::sub]), int(d)))
    return np.concatenate(X), np.concatenate(Y), np.concatenate(D), np.concatenate(R)


HP0 = dict(learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=20, l2_regularization=1.0)


def gbm(max_iter=400, **kw):
    p = dict(HP0, max_iter=max_iter, class_weight='balanced', early_stopping=False, random_state=0); p.update(kw)
    return HGB(**p)


def mf1(y, p): return float(f1_score(y, p, average='macro'))


def curve_of(m, X, y, step=50):
    """VAL macro-F1 every `step` trees; returns the curve, the best checkpoint and the probabilities at it"""
    cur, best, bp = [], (-1, 0), None
    for it, pr in enumerate(m.staged_predict_proba(X), 1):
        if it % step == 0:
            f = mf1(y, pr.argmax(1)); cur.append((it, round(f, 4)))
            if f > best[0]: best, bp = (f, it), pr.copy()
    return cur, best[1], bp


def day_stats(y, p, d):
    pd_ = [mf1(y[d == k], p[d == k]) for k in np.unique(d)]
    return dict(acc=round(float(accuracy_score(y, p)), 4), day_mean=round(float(np.mean(pd_)), 4), day_sd=round(float(np.std(pd_)), 4))


def best_m(prob, y):
    s = {m: mf1(y, (prob * [1, m, 1]).argmax(1)) for m in MS}; m = max(s, key=s.get)
    return m, round(s[m], 4), {str(k): round(v, 4) for k, v in s.items()}
