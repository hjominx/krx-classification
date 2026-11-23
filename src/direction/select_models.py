"""Model / hyper-parameter / feature selection on the VAL block (train on TRAIN only). TEST is not touched here.
usage: python select_models.py  -> results/direction/select.jsonl
"""
import _paths  # noqa: F401  (src/common, src/direction, src/volatility, src/report on the import path)
import json, os, time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score, accuracy_score
import common as C

OUT = C.RES_DIR; os.makedirs(OUT, exist_ok=True)
CACHE = {}


def day(d):
    if d not in CACHE: CACHE[d] = C.load_day(d)
    return CACHE[d]


def stack(days, h_idx, cols=None, sub=1, eval_only=False):
    X, Y, D = [], [], []
    for d in days:
        z = day(d)
        if eval_only and z['cov'] < C.MIN_COV: continue
        idx = np.where(z['V'][:, h_idx])[0][::sub]
        x = z['X'][idx]; X.append(x if cols is None else x[:, cols]); Y.append(z['Y'][idx, h_idx]); D.append(np.full(len(idx), int(d)))
    return np.concatenate(X), np.concatenate(Y).astype(int), np.concatenate(D)


def scores(y, p, d):
    per_day = [f1_score(y[d == k], p[d == k], average='macro') for k in np.unique(d)]
    return dict(mF1=round(f1_score(y, p, average='macro'), 4), acc=round(accuracy_score(y, p), 4),
                mF1_day_mean=round(float(np.mean(per_day)), 4), mF1_day_sd=round(float(np.std(per_day)), 4))


def log(rec):
    print(json.dumps(rec), flush=True)
    with open(os.path.join(OUT, 'select.jsonl'), 'a') as f: f.write(json.dumps(rec) + '\n')


def gbm(**kw):
    p = dict(max_iter=400, learning_rate=0.1, max_leaf_nodes=31, l2_regularization=0.0, class_weight='balanced',
             early_stopping=True, validation_fraction=0.1, n_iter_no_change=20, random_state=0)
    p.update(kw); return HistGradientBoostingClassifier(**p)


if __name__ == '__main__':
    TR, VA, _ = C.split_days()
    names = day(TR[0])['names']; G = C.groups_of(names)
    small = [i for i, n in enumerate(names) if n in ('imb1', 'imb5', 'micro_dev', 'ofi_l1_sum5', 'ofi_l1_sum20', 'svol_sum20')]
    # 1) horizons x models
    for hi, H in enumerate(C.HORIZONS):
        Xtr, ytr, _ = stack(TR, hi, sub=2); Xva, yva, dva = stack(VA, hi, eval_only=True)
        maj = np.bincount(ytr, minlength=3).argmax()
        log(dict(exp='horizon', H=H, model='majority', **scores(yva, np.full_like(yva, maj), dva),
                 val_dist=np.round(np.bincount(yva, minlength=3) / len(yva), 3).tolist()))
        r20 = Xva[:, list(names).index('ret20')]
        log(dict(exp='horizon', H=H, model='momentum(sign ret20)', **scores(yva, np.where(r20 > 0, 2, np.where(r20 < 0, 0, 1)), dva)))
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
        for nm, cols in (('logreg small (imb/OFI/micro)', small), ('logreg all', list(range(Xtr.shape[1])))):
            t0 = time.time()
            m = LogisticRegression(max_iter=500, class_weight='balanced').fit(((Xtr - mu) / sd)[::2][:, cols], ytr[::2])
            log(dict(exp='horizon', H=H, model=nm, **scores(yva, m.predict(((Xva - mu) / sd)[:, cols]), dva), sec=round(time.time() - t0)))
        t0 = time.time(); m = gbm().fit(Xtr, ytr)
        log(dict(exp='horizon', H=H, model='gbm all', **scores(yva, m.predict(Xva), dva), iters=int(m.n_iter_), sec=round(time.time() - t0)))
    # 2) feature-group ablation and 3) small GBM grid at the primary horizon (5 s)
    hi = C.HORIZONS.index(5)
    order = [('book', ['book']), ('book+price', ['book', 'price']), ('book+price+flow', ['book', 'price', 'flow']),
             ('all (+time)', ['book', 'price', 'flow', 'time'])]
    Xtr, ytr, _ = stack(TR, hi, sub=2); Xva, yva, dva = stack(VA, hi, eval_only=True)
    for nm, gs in order:
        cols = sorted(sum((G[g] for g in gs), []))
        m = gbm().fit(Xtr[:, cols], ytr)
        log(dict(exp='ablation', H=5, features=nm, n_features=len(cols), **scores(yva, m.predict(Xva[:, cols]), dva)))
    for lr in (0.05, 0.1):
        for leaves in (15, 63):
            for l2 in (0.0, 1.0):
                t0 = time.time(); m = gbm(learning_rate=lr, max_leaf_nodes=leaves, l2_regularization=l2).fit(Xtr, ytr)
                log(dict(exp='gbm_grid', H=5, learning_rate=lr, max_leaf_nodes=leaves, l2=l2, **scores(yva, m.predict(Xva), dva),
                         iters=int(m.n_iter_), sec=round(time.time() - t0)))
