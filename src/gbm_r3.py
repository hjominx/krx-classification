"""Round 3, 5 s direction: gradient boosting with short-term history and with recency weighting. TEST is not read here.
Design fixed before running. Same data as exp_v2.py: TRAIN days with a trade file, every 2nd row; VAL days with
coverage >= 60 %, every row with a valid 5 s label; the 127 'full' columns.
The round-2 GBM sees only the current row, while the deep models see the last 50 snapshots. Here the GBM also gets
the change of every column over the last 5 and 20 snapshots (about 1 s and 4.5 s), x_t - x_{t-k}, computed on the
whole day before sub-sampling and left missing when t-k is in an earlier segment (gap > 5 s) - strictly causal.
Configs (all: HistGradientBoosting, lr 0.02, 15 leaves, min 100 rows per leaf, l2 1.0, balanced class weights - the
best round-2 setting - up to 800 trees, VAL macro-F1 every 50 trees, best number of trees kept)
  lag     : 127 columns + 127 changes over 5 rows + 127 changes over 20 rows (381 columns)
  rec     : 127 columns, training rows weighted by exp(1.1 x position of the day in TRAIN) (last day 3x the first)
  lag_rec : both, run only if lag and rec each beat the round-2 GBM (VAL 0.5120 at m = 1)
Each run saves its VAL probabilities -> results/r3_val_gbm_<cfg>.npz and the fitted model -> results/r3_gbm_<cfg>.pkl.
usage: python gbm_r3.py lag|rec|lag_rec|all   -> results/r3_gbm.jsonl
"""
import json, os, pickle, sys, time
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier as HGB
import common as C
import exp_v2 as E

OUT = E.OUT; RES = os.path.join(OUT, 'r3_gbm.jsonl'); LAGS = (5, 20); REC = 1.1; REF = 0.5120
HP = dict(learning_rate=0.02, max_leaf_nodes=15, min_samples_leaf=100, l2_regularization=1.0, class_weight='balanced',
          early_stopping=False, random_state=0)


def log(rec):
    print(json.dumps(rec)[:400], flush=True)
    with open(RES, 'a') as f: f.write(json.dumps(rec) + '\n')


def recs(): return [json.loads(l) for l in open(RES)] if os.path.exists(RES) else []


def day_matrix(d, lag):
    """the 127 columns (+ their changes) of the rows of day d that have a valid 5 s label"""
    z = C.load_day(d); X, _, r, cov = E.load_day(d, need_valid=False)
    X = X[:, E.FULL]; v = z['V'][:, E.hi]; seg = z['seg']
    if lag:
        parts = [X]
        for k in LAGS:
            dx = np.full_like(X, np.nan); dx[k:] = X[k:] - X[:-k]; dx[k:][seg[k:] != seg[:-k]] = np.nan; dx[:k] = np.nan; parts.append(dx)
        X = np.hstack(parts)
    w = np.where(v)[0]
    return X[w], z['Y'][w, E.hi].astype(int), w, cov


def stack(days, lag, sub=1, eval_only=False):
    Xs, Ys, Ds, Rs = [], [], [], []
    for d in days:
        x, y, r, cov = day_matrix(d, lag)
        if eval_only and cov < C.MIN_COV: continue
        Xs.append(x[::sub]); Ys.append(y[::sub]); Rs.append(r[::sub]); Ds.append(np.full(len(y[::sub]), int(d)))
    return np.concatenate(Xs), np.concatenate(Ys), np.concatenate(Ds), np.concatenate(Rs)


def run(name):
    t0 = time.time(); lag = 'lag' in name; rec = 'rec' in name
    Xtr, ytr, dtr, _ = stack(E.TR, lag, sub=2); Xva, yva, dva, rva = stack(E.VA, lag, eval_only=True)
    sw = None
    if rec: dd = np.unique(dtr); sw = np.exp(REC * np.searchsorted(dd, dtr) / (len(dd) - 1))
    m = HGB(max_iter=800, **HP).fit(Xtr, ytr, sample_weight=sw); del Xtr
    cur, best = [], (-1, 0, None)
    for it, pr in enumerate(m.staged_predict_proba(Xva), 1):
        if it % 50 == 0:
            f = E.mf1(yva, pr.argmax(1)); cur.append((it, round(f, 4)))
            if f > best[0]: best = (f, it, pr.copy())
    f, it, pr = best; bm, fm, tab = E.best_m(pr, yva)
    np.savez_compressed(os.path.join(OUT, f'r3_val_gbm_{name}.npz'), prob=pr.astype(np.float32), y=yva, day=dva, row=rva)
    with open(os.path.join(OUT, f'r3_gbm_{name}.pkl'), 'wb') as fh: pickle.dump(dict(model=m, best_iter=it, lag=lag, rec=rec), fh)
    log(dict(stage='G', name=name, n_features=int(Xva.shape[1]), n_train=int(len(ytr)), best_iter=it, mF1=round(f, 4), best_m=bm,
             mF1_at_best_m=fm, m_table=tab, curve=cur, ref_r2_gbm=REF, sec=round(time.time() - t0)))


if __name__ == '__main__':
    what = sys.argv[1]; have = {r['name'] for r in recs()}
    todo = ['lag', 'rec', 'lag_rec'] if what == 'all' else [what]
    for n in todo:
        if n in have: continue
        if n == 'lag_rec':
            R = {r['name']: r.get('mF1', 0) for r in recs()}
            if not (R.get('lag', 0) > REF and R.get('rec', 0) > REF): log(dict(stage='G', name='lag_rec', skipped=True, why='lag or rec did not beat the round-2 GBM')); continue
        run(n)
