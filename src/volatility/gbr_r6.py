"""Volatility (60 s) round 6: improving the GBM regression of round 2 (vol_v2.py stage R), the strongest single member.
Design fixed before running; chosen on VAL only; TEST is read once, in 'test'.
Reference: round-2 GBM regression - HistGradientBoostingRegressor on the 145 'allv' columns, target log(1 + RV60),
     lr 0.05, default tree size, every 10th TRAIN row; VAL macro-F1 0.6417 (best of every 100 trees, at 100 trees).
Common settings: VAL rows as vol_v2.py (VAL days with coverage >= 60 %, every valid row); every 50 trees the VAL
     macro-F1 with the best two thresholds on log(1 + RV60) (grid of vol_r4.py) is computed and the best number of trees
     kept. Defaults: lr 0.05, 31 leaves, min 20 rows per leaf, no l2, up to 1,000 trees, every 10th TRAIN row.
Configs (one change each)
  past    : + log(1 + past RV) over 10 / 60 / 300 s (the 3 'past' columns of vol_cls.py; same units as the target)
  resid   : as past, target = log(1 + RV60) - log(1 + past 60 s RV); the prediction adds the past value back (trees
            cannot extrapolate; a persistence baseline may carry over to calmer or busier periods)
  poisson : Poisson loss on RV60 in ticks; prediction log(1 + predicted RV60)
  slow    : lr 0.02, 15 leaves, min 100 rows per leaf, l2 1.0, up to 2,000 trees (the best direction GBM setting)
  sub2    : every 2nd TRAIN row instead of every 10th (5x the rows)
  lag     : + the change of every allv column over the last 5 and 20 snapshots (same segment only), 435 columns
Stage combo: all changes that beat 0.6417 in one config (poisson is left out when resid passes; when lag and sub2 both
     pass, every 4th TRAIN row is used to fit in memory).
Stage val: pool = the best round-6 GBM by VAL (single or combo), the round-2 GBM, and the round-5 conv+GRU bag
     (r5.py vreg_mh, 5 seeds). Candidates: new GBM alone; w x new GBM + (1 - w) x conv+GRU bag (w = 0.1 ... 0.9);
     incumbents exactly as chosen: round 2 (GBM, [2.0, 2.85]) and round 5 (0.7 GBM + 0.3 vreg_mh, [2.0, 2.85]).
     Winner = highest VAL macro-F1; split-half check (choice on VAL days 1-6, scored on 7-12 and the reverse, next to the
     round-2 and round-5 recipes with thresholds chosen on the same half). If an incumbent wins there is no TEST run.
Stage test: once; the saved GBM at its VAL-chosen number of trees, the conv+GRU bag from its checkpoints; references on
     the same rows: rounds 2, 3, 4 and 5 (results/volatility/vol_v2_test_train.npz, vol_r3_test.npz, vol_r4_test.npz,
     r5_vol_test.npz); day-block bootstrap (2,000 resamples).
Note: the TEST days were already used by vol_cls.py, vol_v2.py, vol_r3.py, vol_r4.py and r5_combine.py.
usage: python gbr_r6.py fit <cfg>[,<cfg>] | combo | val | test   -> results/volatility/gbr_r6.jsonl
"""
import _paths  # noqa: F401  (src/common, src/direction, src/volatility, src/report on the import path)
import json, os, pickle, sys, time
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor as HGR
from sklearn.metrics import accuracy_score, confusion_matrix
import common as C
import data_v2 as E

argv = sys.argv; sys.argv = argv[:1]
import vol_v2 as V
import vol_r4 as R4
sys.argv = argv
OUT = C.RES_VOL; os.makedirs(OUT, exist_ok=True); RES = os.path.join(OUT, 'gbr_r6.jsonl'); VD = os.path.join(C.DATA, 'feat_vol')
mf1, tune, by_thr, KEY, boot = R4.mf1, R4.tune, R4.by_thr, R4.KEY, R4.boot
REF = 0.6417; R2_THR = [2.0, 2.85]; WS = np.round(np.arange(0.1, 0.91, 0.1), 2); LAGS = (5, 20)
CFG = dict(past=dict(past=True), resid=dict(past=True, resid=True), poisson=dict(loss='poisson'),
           slow=dict(lr=0.02, leaves=15, min_leaf=100, l2=1.0, iters=2000), sub2=dict(sub=2), lag=dict(lag=True))
DEF = dict(past=False, resid=False, loss='squared_error', lr=0.05, leaves=31, min_leaf=20, l2=0.0, iters=1000, sub=10, lag=False)


def log(rec):
    print(json.dumps({k: v for k, v in rec.items() if k != 'curve'})[:600], flush=True)
    with open(RES, 'a') as f: f.write(json.dumps(rec) + '\n')


def recs(stage=None):
    R = [json.loads(l) for l in open(RES)] if os.path.exists(RES) else []
    return [r for r in R if stage is None or r.get('stage') == stage]


def day_matrix(d, cfg):
    """rows of day d with a valid 60 s volatility label: features, RV60 (ticks), log(1 + past 60 s RV), class"""
    z = C.load_day(d); v = np.load(os.path.join(VD, f'{d}.npz'))
    X = np.hstack([z['X'], np.load(os.path.join(E.FT, f'{d}.npz'))['T'], np.load(os.path.join(E.F2, f'{d}.npz'))['X2'], v['past']]).astype(np.float32)
    cols = list(V.ALLV) + (V.PAST if cfg['past'] else []); A = X[:, cols]
    if cfg['lag']:
        seg = z['seg']; parts = [A]; B = X[:, list(V.ALLV)]
        for k in LAGS:
            dx = np.full_like(B, np.nan); dx[k:] = B[k:] - B[:-k]; dx[k:][seg[k:] != seg[:-k]] = np.nan; dx[:k] = np.nan; parts.append(dx)
        A = np.hstack(parts)
    r = np.where(v['ok'])[0]; rv = v['rv'][r].astype(np.float64)
    return A[r], rv, v['past'][r, 1].astype(np.float64), np.digitize(rv, V.CUTS, right=True), r, float(z['cov'])


def stack(days, cfg, sub=1, eval_only=False):
    P = [[] for _ in range(6)]
    for d in days:
        A, rv, base, y, r, cov = day_matrix(d, cfg)
        if eval_only and cov < C.MIN_COV: continue
        for lst, a in zip(P, (A, rv, base, y, r, np.full(len(y), int(d)))): lst.append(np.array(a[::sub]))
    return [np.concatenate(p) for p in P]


def target(cfg, rv, base):
    if cfg['loss'] == 'poisson': return rv
    return np.log1p(rv) - (base if cfg['resid'] else 0.0)


def to_log(cfg, pred, base):
    """prediction on the log(1 + RV60) scale used by the thresholds"""
    if cfg['loss'] == 'poisson': return np.log1p(np.maximum(pred, 0.0))
    return pred + (base if cfg['resid'] else 0.0)


def model(cfg):
    return HGR(loss=cfg['loss'], learning_rate=cfg['lr'], max_leaf_nodes=cfg['leaves'], min_samples_leaf=cfg['min_leaf'],
               l2_regularization=cfg['l2'], max_iter=cfg['iters'], early_stopping=False, random_state=0)


def fit(name, cfg):
    t0 = time.time(); cfg = dict(DEF, **cfg)
    Xtr, rvt, bt, _, _, _ = stack(E.TR, cfg, sub=cfg['sub']); Xva, rvv, bv, yva, rva, dva = stack(E.VA, cfg, eval_only=True)
    m = model(cfg).fit(Xtr, target(cfg, rvt, bt)); n_train = len(rvt); del Xtr
    best, curve = (-1.0, 0, None, None), []
    for it, pr in enumerate(m.staged_predict(Xva), 1):
        if it % 50 == 0:
            x = to_log(cfg, pr, bv); f, t = tune(x, yva); curve.append((it, round(f, 4)))
            if f > best[0]: best = (f, it, t, x.astype(np.float32))
    f, it, t, x = best
    np.savez_compressed(os.path.join(OUT, f'r6_val_gbr_{name}.npz'), pred=x, y=yva, day=dva, row=rva)
    with open(os.path.join(OUT, f'r6_gbr_{name}.pkl'), 'wb') as fh: pickle.dump(dict(model=m, best_iter=it, cfg=cfg), fh)
    log(dict(stage='fit', name=name, cfg=cfg, n_features=int(Xva.shape[1]), n_train=int(n_train), best_iter=it, mF1=round(f, 4), thr=t,
             acc=round(float(accuracy_score(yva, by_thr(x, t))), 4), ref_r2=REF, beats_ref=bool(f > REF), curve=curve, sec=round(time.time() - t0)))


def combo_cfg():
    ok = {r['name'] for r in recs('fit') if r['name'] in CFG and r['beats_ref']}
    if 'resid' in ok: ok.discard('poisson')
    cfg = {}
    for n in ok: cfg.update(CFG[n])
    if 'lag' in ok and 'sub2' in ok: cfg['sub'] = 4
    return sorted(ok), cfg


def best_fit():
    return max(recs('fit'), key=lambda r: r['mF1'])


def bag_val():
    Z = [np.load(os.path.join(OUT, f'r5_val_vreg_mh_s{s}_vol.npz')) for s in range(5)]
    return Z


def val():
    t0 = time.time(); b = best_fit(); name = b['name']
    if b['mF1'] <= REF: log(dict(stage='val', skipped=True, why='no round-6 GBM beat the round-2 GBM on VAL', best=name)); return
    files = dict(new=[os.path.join(OUT, f'r6_val_gbr_{name}.npz')], gbr=[os.path.join(OUT, 'vol_v2_val_gbr.npz')],
                 cg=[os.path.join(OUT, f'r5_val_vreg_mh_s{s}_vol.npz') for s in range(5)])
    Z = {n: [np.load(f) for f in fs] for n, fs in files.items()}; keys = None
    for zs in Z.values():
        for z in zs: k = KEY(z['day'], z['row']); keys = k if keys is None else np.intersect1d(keys, k)
    P, y = {}, None
    for n, zs in Z.items():
        q = []
        for z in zs:
            k = KEY(z['day'], z['row']); ix = np.searchsorted(k, keys); assert (k[ix] == keys).all(); q.append(z['pred'][ix].astype(np.float64))
            if y is None: y = z['y'][ix].astype(int)
            else: assert (z['y'][ix] == y).all()
        P[n] = np.mean(q, 0)
    d = keys // 1_000_000

    def cands(rows=None):
        yy = y if rows is None else y[rows]; s = (lambda a: a) if rows is None else (lambda a: a[rows])
        out = [(mf1(yy, by_thr(s(P['gbr']), R2_THR)), 'r2', dict(a='gbr', w=1.0, thr=R2_THR)),
               (mf1(yy, by_thr(s(0.7 * P['gbr'] + 0.3 * P['cg']), R2_THR)), 'r5', dict(a='gbr', w=0.7, thr=R2_THR))]
        f, t = tune(P['new'], y, rows); out.append((f, 'new', dict(a='new', w=1.0, thr=t)))
        for w in WS:
            f, t = tune(w * P['new'] + (1 - w) * P['cg'], y, rows); out.append((f, f'new+cg_{w}', dict(a='new', w=float(w), thr=t)))
        return sorted(out, key=lambda c: -c[0])

    def apply(sp): return by_thr(sp['w'] * P[sp['a']] + (1 - sp['w']) * P['cg'], sp['thr'])
    log(dict(stage='val_info', gbm=name, gbm_val_mF1_all_rows=b['mF1'], n_rows=int(len(y))))
    C_ = cands()
    for f, nm, sp in C_: log(dict(stage='cand', name=nm, mF1=round(f, 4), **sp))
    f, nm, sp = C_[0]; p = apply(sp)
    log(dict(stage='winner', name=nm, gbm=name, mF1=round(f, 4), **sp, acc=round(float(accuracy_score(y, p)), 4),
             recall=[round(float(v), 4) for v in confusion_matrix(y, p, normalize='true').diagonal()]))
    days = np.unique(d); halves = (days[:len(days) // 2], days[len(days) // 2:])
    for a_, b_ in ((0, 1), (1, 0)):
        sel, ev = np.where(np.isin(d, halves[a_]))[0], np.isin(d, halves[b_])
        f, nm, sp = cands(sel)[0]; p = apply(sp); _, t2 = tune(P['gbr'], y, sel); _, t5 = tune(0.7 * P['gbr'] + 0.3 * P['cg'], y, sel)
        log(dict(stage='half', chosen_on=[int(x) for x in halves[a_]], chosen=nm, spec=sp, mF1_other_half=round(mf1(y[ev], p[ev]), 4),
                 r2_recipe_other_half=round(mf1(y[ev], by_thr(P['gbr'][ev], t2)), 4),
                 r5_recipe_other_half=round(mf1(y[ev], by_thr((0.7 * P['gbr'] + 0.3 * P['cg'])[ev], t5)), 4)))
    log(dict(stage='val_done', sec=round(time.time() - t0)))


def test():
    w = recs('winner')
    if not w or w[-1]['name'] in ('r2', 'r5'): log(dict(stage='test', skipped=True, why='no round-6 winner on VAL')); return
    w = w[-1]; t0 = time.time(); g = pickle.load(open(os.path.join(OUT, f"r6_gbr_{w['gbm']}.pkl"), 'rb')); cfg = g['cfg']
    Xte, rvt, bte, yte, rte, dte = stack(E.TE, cfg, eval_only=True)
    pr = next(p for it, p in enumerate(g['model'].staged_predict(Xte), 1) if it == g['best_iter']); new = to_log(cfg, pr, bte); kt = KEY(dte, rte); del Xte
    argv = sys.argv; sys.argv = argv[:1]; import r5_combine as RC; sys.argv = argv
    cg, kc = RC.test_bag('vreg_mh') if w['w'] < 1.0 else (None, kt)
    ref = {r: np.load(os.path.join(OUT, f)) for r, f in (('r2', 'vol_v2_test_train.npz'), ('r3', 'vol_r3_test.npz'), ('r4', 'vol_r4_test.npz'), ('r5', 'r5_vol_test.npz'))}
    keys = np.intersect1d(kt, kc)
    for z in ref.values(): keys = np.intersect1d(keys, KEY(z['day'], z['row']))
    ix = lambda k: np.searchsorted(k, keys); y = yte[ix(kt)]; d = keys // 1_000_000
    x = w['w'] * new[ix(kt)] + ((1 - w['w']) * cg[ix(kc)] if cg is not None else 0.0); p = by_thr(x, w['thr']); pr_ = {}
    for r, z in ref.items():
        i = ix(KEY(z['day'], z['row'])); assert (z['y'][i] == y).all(); pr_[r] = z['pred'][i]
    log(dict(stage='test', pipeline=w['name'], gbm=w['gbm'], spec={k: w[k] for k in ('a', 'w', 'thr')}, val_mF1=w['mF1'], n=int(len(y)),
             mF1=round(mf1(y, p), 4), acc=round(float(accuracy_score(y, p)), 4),
             recall=[round(float(v), 4) for v in confusion_matrix(y, p, normalize='true').diagonal()],
             new_gbm_alone_mF1=round(mf1(y, by_thr(new[ix(kt)], [c for c in recs('cand') if c['name'] == 'new'][-1]['thr'])), 4),
             **{f'{r}_mF1': round(mf1(y, q), 4) for r, q in pr_.items()}, **{f'boot_new_minus_{r}': boot(y, d, p, q) for r, q in pr_.items()},
             days_better_than_r2=int(sum(mf1(y[d == k], p[d == k]) > mf1(y[d == k], pr_['r2'][d == k]) for k in np.unique(d))),
             day_mF1={str(k): round(mf1(y[d == k], p[d == k]), 4) for k in np.unique(d)}, confusion=confusion_matrix(y, p).tolist(),
             sec=round(time.time() - t0)))
    np.savez_compressed(os.path.join(OUT, 'r6_test.npz'), x=x.astype(np.float32), pred=p, y=y, day=d, row=keys % 1_000_000)


if __name__ == '__main__':
    what = sys.argv[1]; have = {r['name'] for r in recs('fit')}
    if what == 'fit':
        for n in sys.argv[2].split(','):
            if n not in have: fit(n, CFG[n])
    elif what == 'combo':
        names, cfg = combo_cfg(); print('combo of', names, cfg, flush=True)
        if len(names) < 2: log(dict(stage='fit_combo', skipped=True, why='fewer than two changes passed', passed=names))
        elif 'combo' not in have: fit('combo', cfg)
    elif what == 'val' and not recs('winner'): val()
    elif what == 'test' and not recs('test'): test()
