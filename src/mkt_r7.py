"""Round 7: do market-wide features help? The 24 columns of market_feat.py (KOSPI 200, Samsung Electronics, Nikkei,
Nasdaq-100 / S&P 500 futures, USD/KRW hourly; previous US session of the semiconductor index, VIX, Micron, NVIDIA),
added to the round-2 GBM recipes of both tasks. Design fixed before running; chosen on VAL only; TEST read once.
Paired configs (same code, same rows; the only difference is the 24 extra columns)
  vol_base / vol_mkt : volatility - GBM regression of round 2 (145 allv columns, log(1 + RV60), lr 0.05, every 10th TRAIN
                       row), VAL macro-F1 with the best two thresholds every 50 trees, up to 1,000 trees
  dir_base / dir_mkt : direction - GBM classifier of round 2 (127 full columns, every 2nd TRAIN row, lr 0.02, 15 leaves,
                       min 100 rows per leaf, l2 1.0, balanced classes), VAL macro-F1 (most probable class) every 50 trees,
                       up to 800 trees
Pass rules: vol_mkt > max(vol_base, round-2 VAL 0.6417); dir_mkt > dir_base.
vol_val (if vol_mkt passed): as gbr_r6.py stage val with vol_mkt as the new GBM - candidates new GBM alone and blended
     with the round-5 conv+GRU bag (vreg_mh), incumbents round 2 and round 5 exactly as chosen; split-half check;
vol_test: once, references rounds 2-5 on the same rows (gbr_r6.py stage test).
dir_val (if dir_mkt passed): the round-3 stacking recipe (logistic regression on the members' log-probabilities,
     balanced classes, leave-one-day-out, per-class multipliers on the wide grid) with and without the dir_mkt GBM as an
     extra member, on the same rows; the stacking with it is chosen only if it beats the one without on VAL.
Note: the TEST days were already used by every earlier round.
usage: python mkt_r7.py fit vol_base,vol_mkt,dir_base,dir_mkt | vol_val | vol_test | dir_val   -> results/mkt_r7.jsonl
"""
import json, os, sys, time
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor as HGR
from sklearn.metrics import accuracy_score
import common as C
import exp_v2 as E

argv = sys.argv; sys.argv = argv[:1]
import vol_v2 as V
import vol_r4 as R4
import gbr_r6 as G6
sys.argv = argv
OUT = E.OUT; RES = os.path.join(OUT, 'mkt_r7.jsonl'); FD = os.path.join(C.DATA, 'feat_mkt'); VD = os.path.join(C.DATA, 'feat_vol')
REF_VOL = 0.6417


def log(rec):
    print(json.dumps({k: v for k, v in rec.items() if k != 'curve'})[:600], flush=True)
    with open(RES, 'a') as f: f.write(json.dumps(rec) + '\n')


def recs(stage=None):
    R = [json.loads(l) for l in open(RES)] if os.path.exists(RES) else []
    return [r for r in R if stage is None or r.get('stage') == stage]


def mkt(d): return np.load(os.path.join(FD, f'{d}.npz'))['X']


def vol_stack(days, use_mkt, sub=1, eval_only=False):
    P = [[] for _ in range(5)]
    for d in days:
        z = C.load_day(d)
        if eval_only and float(z['cov']) < C.MIN_COV: continue
        v = np.load(os.path.join(VD, f'{d}.npz'))
        X = np.hstack([z['X'], np.load(os.path.join(E.FT, f'{d}.npz'))['T'], np.load(os.path.join(E.F2, f'{d}.npz'))['X2']]).astype(np.float32)[:, list(V.ALLV)]
        if use_mkt: X = np.hstack([X, mkt(d)])
        r = np.where(v['ok'])[0][::sub]; rv = v['rv'][r].astype(np.float64)
        for lst, a in zip(P, (X[r], rv, np.digitize(rv, V.CUTS, right=True), r, np.full(len(r), int(d)))): lst.append(np.array(a))
    return [np.concatenate(p) for p in P]


def dir_stack(days, use_mkt, sub=1, eval_only=False):
    P = [[] for _ in range(4)]
    for d in days:
        x, y, r, cov = E.load_day(d)
        if eval_only and cov < C.MIN_COV: continue
        X = x[:, E.FULL]
        if use_mkt: X = np.hstack([X, mkt(d)[r]])
        for lst, a in zip(P, (X[::sub], y[::sub], r[::sub], np.full(len(y[::sub]), int(d)))): lst.append(np.array(a))
    return [np.concatenate(p) for p in P]


def fit_vol(name, use_mkt):
    t0 = time.time(); Xtr, rvt, _, _, _ = vol_stack(E.TR, use_mkt, sub=10); Xva, rvv, yva, rva, dva = vol_stack(E.VA, use_mkt, eval_only=True)
    m = HGR(learning_rate=0.05, max_iter=1000, early_stopping=False, random_state=0).fit(Xtr, np.log1p(rvt)); n = len(rvt); del Xtr
    best, curve = (-1.0, 0, None, None), []
    for it, pr in enumerate(m.staged_predict(Xva), 1):
        if it % 50 == 0:
            f, t = R4.tune(pr, yva); curve.append((it, round(f, 4)))
            if f > best[0]: best = (f, it, t, pr.astype(np.float32))
    f, it, t, x = best
    np.savez_compressed(os.path.join(OUT, f'r7_val_{name}.npz'), pred=x, y=yva, day=dva, row=rva)
    import pickle
    with open(os.path.join(OUT, f'r7_{name}.pkl'), 'wb') as fh: pickle.dump(dict(model=m, best_iter=it, use_mkt=use_mkt), fh)
    log(dict(stage='fit', name=name, task='vol', n_features=int(Xva.shape[1]), n_train=int(n), best_iter=it, mF1=round(f, 4), thr=t,
             acc=round(float(accuracy_score(yva, R4.by_thr(x, t))), 4), curve=curve, sec=round(time.time() - t0)))


def fit_dir(name, use_mkt):
    t0 = time.time(); Xtr, ytr, _, _ = dir_stack(E.TR, use_mkt, sub=2); Xva, yva, rva, dva = dir_stack(E.VA, use_mkt, eval_only=True)
    m = E.gbm(max_iter=800, learning_rate=0.02, min_samples_leaf=100).fit(Xtr, ytr); n = len(ytr); del Xtr
    cur, it, pr = E.curve_of(m, Xva, yva); bm, fm, tab = E.best_m(pr, yva); f = R4.mf1(yva, pr.argmax(1))
    np.savez_compressed(os.path.join(OUT, f'r7_val_{name}.npz'), prob=pr.astype(np.float32), y=yva, day=dva, row=rva)
    import pickle
    with open(os.path.join(OUT, f'r7_{name}.pkl'), 'wb') as fh: pickle.dump(dict(model=m, best_iter=it, use_mkt=use_mkt), fh)
    log(dict(stage='fit', name=name, task='dir', n_features=int(Xva.shape[1]), n_train=int(n), best_iter=it, mF1=round(f, 4), best_m=bm,
             mF1_at_best_m=fm, acc=round(float(accuracy_score(yva, pr.argmax(1))), 4), curve=cur, sec=round(time.time() - t0)))


def score(name): return [r for r in recs('fit') if r['name'] == name][-1]['mF1']


def passed(task):
    if task == 'vol': return score('vol_mkt') > max(score('vol_base'), REF_VOL)
    return score('dir_mkt') > score('dir_base')


if __name__ == '__main__':
    what = sys.argv[1]; have = {r['name'] for r in recs('fit')}
    if what == 'fit':
        for n in sys.argv[2].split(','):
            if n in have: continue
            (fit_vol if n.startswith('vol') else fit_dir)(n, n.endswith('mkt'))
