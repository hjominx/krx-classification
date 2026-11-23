"""Book + trade-tape experiments, 5 s horizon unless stated. Selection on VAL only; TEST in stage 'final' only.
Days: those with both book and trade data (07-09 has no trades): TRAIN 46, VAL 12, TEST 10.
  A     : base 71 book features vs base + 24 trade features (GBM, settings chosen earlier on VAL)
  B     : drop-one-group from the full set; rule fixed in advance: a group is dropped if removing it does not lower
          VAL macro-F1 by more than 0.001; the reduced set is kept only if it is not worse than the full set
  C     : GBM capacity on the kept set, up to 1500 trees; the number of trees is chosen on VAL (checkpoints of 100)
  final : best VAL model among GBM (stage C) and GRU (gru_tr.py) scored once on TEST; plus all horizons and the
          next-change task with the chosen GBM recipe (no further choices)
usage: python exp_trades.py A|B|C|final -> results/direction/trades.jsonl
"""
import _paths  # noqa: F401  (src/common, src/direction, src/volatility, src/report on the import path)
import json, os, sys, time
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
import common as C
from select_models import OUT

STAGE = sys.argv[1]
RES = os.path.join(OUT, 'trades.jsonl'); FT = os.path.join(C.DATA, 'feat_tr')
TR0, VA, TE = C.split_days()
TR = [d for d in TR0 if os.path.exists(os.path.join(FT, f'{d}.npz'))]
base_names = list(C.load_day(TR[0])['names']); tz = np.load(os.path.join(FT, f'{TR[0]}.npz'))
NAMES = base_names + list(tz['names'])
GB = C.groups_of(base_names)
GROUPS = {g: list(ix) for g, ix in GB.items()}
for g in ('tw', 'ev', 'last', 'cum'):
    GROUPS['trade_' + g] = [len(base_names) + i for i, gg in enumerate(tz['groups']) if gg == g]
ALL = sorted(sum(GROUPS.values(), []))
g0 = max((json.loads(l) for l in open(os.path.join(OUT, 'select.jsonl')) if '"gbm_grid"' in l), key=lambda r: r['mF1'])
HP0 = dict(learning_rate=g0['learning_rate'], max_leaf_nodes=g0['max_leaf_nodes'], l2_regularization=g0['l2'])
CACHE = {}


def day(d, hi):
    if (d, hi) not in CACHE:
        z = C.load_day(d); t = np.load(os.path.join(FT, f'{d}.npz'))['T']; v = z['V'][:, hi]
        CACHE[(d, hi)] = (np.hstack([z['X'], t])[v].astype(np.float32), z['Y'][v, hi].astype(int), float(z['cov']), z['minute'][v])
    return CACHE[(d, hi)]


def stack(days, hi, sub=1, eval_only=False):
    P = [(d, day(d, hi)) for d in days]
    P = [(d, p) for d, p in P if not (eval_only and p[2] < C.MIN_COV)]
    return (np.concatenate([p[0][::sub] for _, p in P]), np.concatenate([p[1][::sub] for _, p in P]),
            np.concatenate([np.full(len(p[1][::sub]), int(d)) for d, p in P]), np.concatenate([p[3][::sub] for _, p in P]))


def gbm(**kw):
    p = dict(max_iter=400, class_weight='balanced', early_stopping=True, validation_fraction=0.1, n_iter_no_change=20,
             random_state=0, **HP0); p.update(kw); return HistGradientBoostingClassifier(**p)


def sc(y, p, dd=None):
    r = dict(mF1=round(f1_score(y, p, average='macro'), 4), acc=round(accuracy_score(y, p), 4))
    if dd is not None:
        pdm = [f1_score(y[dd == k], p[dd == k], average='macro') for k in np.unique(dd)]
        r.update(day_mean=round(float(np.mean(pdm)), 4), day_sd=round(float(np.std(pdm)), 4))
    return r


def log(rec):
    print(json.dumps(rec), flush=True)
    with open(RES, 'a') as f: f.write(json.dumps(rec) + '\n')


def done(stage, name): return os.path.exists(RES) and any(json.loads(l).get('stage') == stage and json.loads(l).get('name') == name for l in open(RES))


def recs(stage): return [json.loads(l) for l in open(RES) if json.loads(l).get('stage') == stage] if os.path.exists(RES) else []


hi = C.HORIZONS.index(5)
if STAGE in ('A', 'B', 'C'):
    Xtr, ytr, _, _ = stack(TR, hi, sub=2); Xva, yva, dva, _ = stack(VA, hi, eval_only=True)
if STAGE == 'A':
    for name, cols in (('base71', sorted(sum((GROUPS[g] for g in GB), []))), ('base+trade95', ALL)):
        if done('A', name): continue
        t0 = time.time(); m = gbm().fit(Xtr[:, cols], ytr)
        log(dict(stage='A', name=name, n_features=len(cols), **sc(yva, m.predict(Xva[:, cols]), dva), iters=int(m.n_iter_), sec=round(time.time() - t0)))
elif STAGE == 'B':
    full = next(r for r in recs('A') if r['name'] == 'base+trade95')
    for g in GROUPS:
        if done('B', 'drop_' + g): continue
        cols = [c for c in ALL if c not in GROUPS[g]]; m = gbm().fit(Xtr[:, cols], ytr)
        r = sc(yva, m.predict(Xva[:, cols]), dva)
        log(dict(stage='B', name='drop_' + g, group=g, n_features=len(cols), **r, contribution=round(full['mF1'] - r['mF1'], 4), iters=int(m.n_iter_)))
    if not done('B', 'kept'):
        drop = [r['group'] for r in recs('B') if r['name'].startswith('drop_') and r['contribution'] <= 0.001]
        cols = [c for c in ALL if not any(c in GROUPS[g] for g in drop)]
        m = gbm().fit(Xtr[:, cols], ytr); r = sc(yva, m.predict(Xva[:, cols]), dva)
        use = r['mF1'] >= full['mF1'] - 0.0005
        log(dict(stage='B', name='kept', dropped_groups=drop, n_features=len(cols), **r, full_mF1=full['mF1'],
                 use_reduced=bool(use), cols=cols if use else ALL))
elif STAGE == 'C':
    cols = next(r for r in recs('B') if r['name'] == 'kept')['cols']
    grid = [('lr.05_l15_m20', dict(learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=20)),
            ('lr.05_l31_m200', dict(learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=200)),
            ('lr.05_l63_m500', dict(learning_rate=0.05, max_leaf_nodes=63, min_samples_leaf=500)),
            ('lr.1_l31_m200', dict(learning_rate=0.1, max_leaf_nodes=31, min_samples_leaf=200))]
    for name, kw in grid:
        if done('C', name): continue
        t0 = time.time(); m = gbm(max_iter=1500, early_stopping=False, l2_regularization=1.0, **kw).fit(Xtr[:, cols], ytr)
        curve = []
        for it, pr in enumerate(m.staged_predict(Xva[:, cols]), 1):
            if it % 100 == 0: curve.append((it, round(f1_score(yva, pr, average='macro'), 4)))
        best = max(curve, key=lambda c: c[1])
        log(dict(stage='C', name=name, **kw, l2=1.0, curve=curve, best_iter=best[0], mF1=best[1], sec=round(time.time() - t0)))
elif STAGE == 'final':
    cols = next(r for r in recs('B') if r['name'] == 'kept')['cols']
    c = max(recs('C'), key=lambda r: r['mF1'])
    kw = dict(learning_rate=c['learning_rate'], max_leaf_nodes=c['max_leaf_nodes'], min_samples_leaf=c['min_samples_leaf'],
              l2_regularization=1.0, max_iter=c['best_iter'], early_stopping=False)
    gru = [json.loads(l) for l in open(os.path.join(OUT, 'gru_tr.jsonl'))] if os.path.exists(os.path.join(OUT, 'gru_tr.jsonl')) else []
    gbest = max(gru, key=lambda r: r['best_val_mF1']) if gru else None
    chosen = 'gru' if gbest and gbest['best_val_mF1'] > c['mF1'] else 'gbm'
    log(dict(stage='final', name='choice', gbm=dict(name=c['name'], val_mF1=c['mF1'], trees=c['best_iter']),
             gru=dict(name=gbest['name'], val_mF1=gbest['best_val_mF1']) if gbest else None, chosen=chosen))
    for H in C.HORIZONS:                                   # GBM recipe on every horizon (5 s is the primary)
        h = C.HORIZONS.index(H)
        if done('final', f'gbm_H{H}'): continue
        Xtr, ytr, _, _ = stack(TR, h, sub=2); Xte, yte, dte, mte = stack(TE, h, eval_only=True)
        m = gbm(**kw).fit(Xtr[:, cols], ytr); prob = m.predict_proba(Xte[:, cols]); p = prob.argmax(1)
        np.savez(os.path.join(OUT, f'test_pred_trades_gbm_H{H}.npz'), prob=prob, y=yte, day=dte, minute=mte)
        maj = np.bincount(ytr, minlength=3).argmax()
        log(dict(stage='final', name=f'gbm_H{H}', H=H, **sc(yte, p, dte), majority=sc(yte, np.full_like(yte, maj)),
                 per_day_mF1={str(k): round(f1_score(yte[dte == k], p[dte == k], average='macro'), 4) for k in np.unique(dte)},
                 confusion=confusion_matrix(yte, p).tolist()))
        del Xtr
