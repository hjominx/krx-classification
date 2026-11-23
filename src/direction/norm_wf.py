"""Experiment 3: price-level normalisation and rolling (walk-forward) retraining, GBM, 5 s horizon.
feature sets
  raw  : the 71 features as built (OFI / signed volume are scaled by the SAME day's median L1 depth)
  norm : causal version - log-size / log-volume features minus their mean over the previous 5 trading days,
         OFI / signed volume rescaled to the previous 5 days' median L1 depth, plus tick size in bp (1e7 / mid)
training schemes
  fixed        : train on TRAIN only (as in final_eval.py)
  wf_expanding : refit every 5 trading days on all days before the chunk
  wf_last20    : refit every 5 trading days on the 20 trading days before the chunk
Selection on VAL ('val' mode); the chosen variant is scored once on TEST ('test' mode).
usage: python norm_wf.py val|test -> results/direction/norm_wf.jsonl
"""
import _paths  # noqa: F401  (src/common, src/direction, src/volatility, src/report on the import path)
import json, os, sys, time
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
import common as C
from select_models import gbm, OUT

MODE = sys.argv[1]
H = 5; hi = C.HORIZONS.index(H); CHUNK = 5; SUB = 2
RES = os.path.join(OUT, 'norm_wf.jsonl')
DAYS = C.all_days(); TR, VA, TE = C.split_days()
names = list(C.load_day(DAYS[0])['names'])
LOGC = [i for i, n in enumerate(names) if n.startswith(('log_aq', 'log_bq', 'vol_sum')) or n == 'vol']
FLOWC = [i for i, n in enumerate(names) if n.startswith(('ofi', 'svol'))]
g = max((json.loads(l) for l in open(os.path.join(OUT, 'select.jsonl')) if '"gbm_grid"' in l), key=lambda r: r['mF1'])
HP = dict(learning_rate=g['learning_rate'], max_leaf_nodes=g['max_leaf_nodes'], l2_regularization=g['l2'])

stats = {}
for d in DAYS:
    z = C.load_day(d)
    aq0, bq0 = np.expm1(z['lob'][:, 1].astype(np.float64)), np.expm1(z['lob'][:, 3].astype(np.float64))
    stats[d] = dict(depth=float(np.median(aq0 + bq0) + 1.0), logmean=z['X'][:, LOGC].astype(np.float64).mean(0))

CACHE = {}


def rows(d, norm):
    if (d, norm) in CACHE: return CACHE[(d, norm)]
    z = C.load_day(d); v = z['V'][:, hi]; X = z['X'][v].astype(np.float32)
    if norm:
        i = DAYS.index(d); prev = DAYS[max(0, i - 5):i] or [d]
        X[:, LOGC] -= np.mean([stats[p]['logmean'] for p in prev], 0).astype(np.float32)
        X[:, FLOWC] *= np.float32(stats[d]['depth'] / np.median([stats[p]['depth'] for p in prev]))
        X = np.hstack([X, (1e7 / z['mid'][v]).astype(np.float32)[:, None]])
    out = (X, z['Y'][v, hi].astype(int), float(z['cov']))
    CACHE[(d, norm)] = out; return out


def train_set(days, norm):
    parts = [rows(d, norm) for d in days]
    return np.concatenate([p[0][::SUB] for p in parts]), np.concatenate([p[1][::SUB] for p in parts])


def evaluate(scheme, norm, block):
    ev = [d for d in block if rows(d, norm)[2] >= C.MIN_COV]
    if scheme == 'fixed': plan = [(TR, ev)]
    else:
        plan = []
        for k in range(0, len(ev), CHUNK):
            ch = ev[k:k + CHUNK]; before = [d for d in DAYS if d < ch[0]]
            plan.append((before if scheme == 'wf_expanding' else before[-20:], ch))
    y_all, p_all, d_all, fits = [], [], [], []
    for trd, ch in plan:
        t0 = time.time(); Xtr, ytr = train_set(trd, norm); m = gbm(**HP).fit(Xtr, ytr)
        fits.append(dict(train_from=trd[0], train_to=trd[-1], n_train_days=len(trd), eval_days=ch, iters=int(m.n_iter_), sec=round(time.time() - t0)))
        for d in ch:
            X, y, _ = rows(d, norm); y_all.append(y); p_all.append(m.predict(X)); d_all.append(np.full(len(y), int(d)))
        print(scheme, norm, fits[-1], flush=True)
    y, p, dd = map(np.concatenate, (y_all, p_all, d_all))
    return dict(mF1=round(f1_score(y, p, average='macro'), 4), acc=round(accuracy_score(y, p), 4), n=int(len(y)),
                per_day_mF1={str(k): round(f1_score(y[dd == k], p[dd == k], average='macro'), 4) for k in np.unique(dd)}, fits=fits)


VARIANTS = [(s, n) for n in (False, True) for s in ('fixed', 'wf_expanding', 'wf_last20')]
recs = [json.loads(l) for l in open(RES)] if os.path.exists(RES) else []
if MODE == 'val':
    done = {(r['scheme'], r['norm']) for r in recs if r['block'] == 'val'}
    for s, n in VARIANTS:
        if (s, n) in done: continue
        r = dict(block='val', scheme=s, norm=n, **evaluate(s, n, VA))
        with open(RES, 'a') as f: f.write(json.dumps(r) + '\n')
elif MODE == 'test':
    val = [r for r in recs if r['block'] == 'val']
    best = max(val, key=lambda r: r['mF1'])
    print('chosen on VAL:', best['scheme'], 'norm' if best['norm'] else 'raw', best['mF1'])
    r = dict(block='test', scheme=best['scheme'], norm=best['norm'], chosen_on_val_mF1=best['mF1'], **evaluate(best['scheme'], best['norm'], TE))
    print(json.dumps({k: v for k, v in r.items() if k != 'fits'}))
    with open(RES, 'a') as f: f.write(json.dumps(r) + '\n')
