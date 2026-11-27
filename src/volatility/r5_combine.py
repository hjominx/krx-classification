"""Round 5, volatility: combine the round-5 bags that passed the screen (r5.py) with the round-2 GBM regression.
Design fixed before running; chosen on VAL only; TEST is read once, in 'vol_test'.
Pool (VAL rows every member scores): round-2 GBM regression (vol_v2.py), round-4 conv+GRU regression bag (vol_r4.py,
     seeds 0-2), and the 5-seed bag of every round-5 config that passed for volatility (mean of predictions).
Candidates (macro-F1 on VAL)
  incumbents: round 2 (GBM regression, thresholds [2.0, 2.85]) and round 4 (0.8 GBM + 0.2 round-4 bag, [2.0, 2.85]),
              exactly as chosen
  per new bag B: B alone; w x GBM + (1 - w) x B for w = 0.1 ... 0.9 - thresholds chosen on VAL (vol_r4.py grid)
  if two or more bags passed: their mean A, alone and blended with the GBM the same way
Winner: highest VAL macro-F1; if an incumbent wins there is no TEST run. Split-half check: the whole choice on VAL days
        1-6, scored on days 7-12 and the reverse, next to the round-2 and round-4 recipes with thresholds re-chosen on
        the same half.
TEST: once; new bags from their saved checkpoints, GBM refitted on TRAIN with its settings (deterministic, checked
      against round 2's TEST predictions). References on the same rows: rounds 2, 3 and 4 (results/volatility/vol_v2_test_train.npz,
      vol_r3_test.npz, vol_r4_test.npz); day-block bootstrap (2,000 resamples).
Note: the TEST days were already used by vol_cls.py, vol_v2.py, vol_r3.py and vol_r4.py.
usage: python r5_combine.py vol_val | vol_test [threads]   -> results/volatility/r5_combine.jsonl
"""
import _paths  # noqa: F401  (src/common, src/direction, src/volatility, src/report on the import path)
import json, os, sys, time
import numpy as np, torch
from sklearn.metrics import accuracy_score, confusion_matrix
import common as C
import data_v2 as E

argv = sys.argv; sys.argv = argv[:1]
import seq_models as D
import vol_v2 as V
import vol_r4 as R4
import r5 as R5
sys.argv = argv
OUT = C.RES_VOL; os.makedirs(OUT, exist_ok=True); RES = os.path.join(OUT, 'r5_combine.jsonl'); mf1, tune, by_thr, KEY, boot = R4.mf1, R4.tune, R4.by_thr, R4.KEY, R4.boot
WS = np.round(np.arange(0.1, 0.91, 0.1), 2); R2_THR = [2.0, 2.85]


def log(rec):
    print(json.dumps(rec)[:600], flush=True)
    with open(RES, 'a') as f: f.write(json.dumps(rec) + '\n')


def recs(stage=None):
    R = [json.loads(l) for l in open(RES)] if os.path.exists(RES) else []
    return [r for r in R if stage is None or r.get('stage') == stage]


def new_bags():
    return [n for n, t, _, _ in R5.passing() if t == 'vol']


def val_pool():
    files = {'gbr': [os.path.join(OUT, 'vol_v2_val_gbr.npz')], 'r4': [os.path.join(OUT, f'vol_r4_val_convgru_reg_s{s}.npz') for s in (0, 1, 2)]}
    for n in new_bags(): files[n] = [os.path.join(OUT, f'r5_val_{n}_s{s}_vol.npz') for s in range(5)]
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
    return P, y, keys


def candidates(P, y, rows=None):
    sub = lambda a: a if rows is None else a[rows]; yy = sub(y); gb = P['gbr']
    out = [(mf1(yy, by_thr(sub(gb), R2_THR)), 'r2', dict(src='gbr', w=1.0, thr=R2_THR)),
           (mf1(yy, by_thr(sub(0.8 * gb + 0.2 * P['r4']), R2_THR)), 'r4', dict(src='r4', w=0.8, thr=R2_THR))]
    srcs = [n for n in P if n not in ('gbr', 'r4', 'mean_new')]
    if len(srcs) > 1:
        if 'mean_new' not in P: P = dict(P, mean_new=np.mean([P[n] for n in srcs], 0))
        srcs = srcs + ['mean_new']
    for s in srcs:
        f, t = tune(P[s], y, rows); out.append((f, f'{s}', dict(src=s, w=0.0, thr=t)))
        for w in WS:
            f, t = tune(w * gb + (1 - w) * P[s], y, rows); out.append((f, f'{s}+gbr_{w}', dict(src=s, w=float(w), thr=t)))
    return sorted(out, key=lambda c: -c[0]), P


def apply(spec, P):
    return by_thr(spec['w'] * P['gbr'] + (1 - spec['w']) * P[spec['src']], spec['thr'])


def vol_val():
    t0 = time.time(); bags = new_bags()
    if not bags: log(dict(stage='vol_val', skipped=True, why='no round-5 config passed the volatility screen')); return
    P, y, keys = val_pool(); d = keys // 1_000_000
    log(dict(stage='info', n_rows=int(len(y)), new_bags=bags))
    C_, P = candidates(P, y)
    for f, nm, sp in C_[:30]: log(dict(stage='cand', name=nm, mF1=round(f, 4), **sp))
    f, nm, sp = C_[0]; p = apply(sp, P)
    log(dict(stage='winner', name=nm, mF1=round(f, 4), **sp, acc=round(float(accuracy_score(y, p)), 4),
             recall=[round(float(x), 4) for x in confusion_matrix(y, p, normalize='true').diagonal()]))
    days = np.unique(d); halves = (days[:len(days) // 2], days[len(days) // 2:])
    for a, b in ((0, 1), (1, 0)):
        sel, ev = np.where(np.isin(d, halves[a]))[0], np.isin(d, halves[b])
        f, nm, sp = candidates(P, y, sel)[0][0]; p = apply(sp, P)
        _, t2 = tune(P['gbr'], y, sel); _, t4 = tune(0.8 * P['gbr'] + 0.2 * P['r4'], y, sel)
        log(dict(stage='half', chosen_on=[int(x) for x in halves[a]], chosen=nm, spec=sp, mF1_other_half=round(mf1(y[ev], p[ev]), 4),
                 r2_recipe_other_half=round(mf1(y[ev], by_thr(P['gbr'][ev], t2)), 4),
                 r4_recipe_other_half=round(mf1(y[ev], by_thr((0.8 * P['gbr'] + 0.2 * P['r4'])[ev], t4)), 4)))
    log(dict(stage='val_done', sec=round(time.time() - t0)))


def test_bag(name, pfx='r5'):
    """TEST predictions (log(1 + RV60)) of the 5-seed bag of a round-5 volatility config, with the keys of the rows"""
    F, Y, Vd, gseg, day, row, cov = D.load(E.TE, list(V.ALLV)); ok = np.zeros(len(F), bool); k = 0
    for dd in E.TE:
        a = np.load(os.path.join(R5.VD, f'{dd}.npz')); m = len(a['ok']); ok[k:k + m] = a['ok']; k += m
    preds, keys = [], None
    for s in range(5):
        ck = torch.load(os.path.join(OUT, f'{pfx}_{name}_s{s}_vol.pt'), weights_only=False); cfg = ck['cfg']
        st = cfg['stride']; span = st * (D.T - 1); off = np.arange(-span, 1, st)
        seg_ok = np.zeros(len(F), bool); seg_ok[span:] = gseg[span:] == gseg[:-span]; ev = np.where(seg_ok & ok & cov)[0]
        G_ = ((F - ck['mu']) / ck['sd']).astype(np.float32); np.clip(G_, -10, 10, out=G_)
        net = D.ConvGRU(F.shape[1], cfg['hidden'], cfg['dropout'], ck['nout'], c=cfg.get('conv', 64)); net.load_state_dict(ck['state']); net.to(R5.DEV)
        nd = 3 * len(D.HS) if cfg['dir'] else 0; col = nd + ck['hv'].index(R5.IV)
        o = R5.outputs(net, G_, ev, off)[:, col] * ck['zs'][R5.IV] + ck['zm'][R5.IV]; del G_
        k_ = KEY(day[ev], row[ev])
        if keys is None: keys = k_
        else: assert (keys == k_).all()
        preds.append(o)
    return np.mean(preds, 0), keys


def vol_test():
    w = recs('winner')
    if not w or w[-1]['name'] in ('r2', 'r4'): log(dict(stage='vol_test', skipped=True, why='no round-5 winner on VAL')); return
    w = w[-1]; t0 = time.time(); srcs = new_bags() if w['src'] == 'mean_new' else [w['src']]
    B = [test_bag(n) for n in srcs]; kb = B[0][1]
    for b in B[1:]: kb = np.intersect1d(kb, b[1])
    bag = np.mean([b[0][np.searchsorted(b[1], kb)] for b in B], 0)
    Xte, yte, _, dte, rte = V.stack(E.TE, eval_only=True); kt = KEY(dte, rte); Xt, yt, lt, _, _ = V.stack(E.TR, sub=10); R = V.recs('R')[-1]
    gb_all = V.gbr(R['best_iter']).fit(Xt[:, V.ALLV], lt).predict(Xte[:, V.ALLV]); del Xt, Xte
    ref = {r: np.load(os.path.join(OUT, f)) for r, f in (('r2', 'vol_v2_test_train.npz'), ('r3', 'vol_r3_test.npz'), ('r4', 'vol_r4_test.npz'))}
    keys = np.intersect1d(kb, kt)
    for z in ref.values(): keys = np.intersect1d(keys, KEY(z['day'], z['row']))
    ix = lambda k: np.searchsorted(k, keys)
    y = yte[ix(kt)]; d = keys // 1_000_000; gb = gb_all[ix(kt)]; x = w['w'] * gb + (1 - w['w']) * bag[ix(kb)]; p = by_thr(x, w['thr'])
    pr = {}
    for r, z in ref.items():
        i = ix(KEY(z['day'], z['row'])); assert (z['y'][i] == y).all(); pr[r] = z['pred'][i]
    same = float(np.mean(by_thr(gb, R2_THR) == pr['r2']))
    log(dict(stage='vol_test', pipeline=w['name'], spec={k: w[k] for k in ('src', 'w', 'thr')}, val_mF1=w['mF1'], n=int(len(y)),
             mF1=round(mf1(y, p), 4), acc=round(float(accuracy_score(y, p)), 4),
             recall=[round(float(v), 4) for v in confusion_matrix(y, p, normalize='true').diagonal()],
             refit_gbr_agrees_with_r2=round(same, 6), **{f'{r}_mF1': round(mf1(y, q), 4) for r, q in pr.items()},
             **{f'boot_new_minus_{r}': boot(y, d, p, q) for r, q in pr.items()},
             days_better_than_r2=int(sum(mf1(y[d == k], p[d == k]) > mf1(y[d == k], pr['r2'][d == k]) for k in np.unique(d))),
             day_mF1={str(k): round(mf1(y[d == k], p[d == k]), 4) for k in np.unique(d)}, confusion=confusion_matrix(y, p).tolist(),
             sec=round(time.time() - t0)))
    np.savez_compressed(os.path.join(OUT, 'r5_vol_test.npz'), x=x.astype(np.float32), pred=p, y=y, day=d, row=keys % 1_000_000)


if __name__ == '__main__':
    what = sys.argv[1]; torch.set_num_threads(int(sys.argv[2]) if len(sys.argv) > 2 else 4)
    if what == 'vol_val' and not recs('winner'): vol_val()
    elif what == 'vol_test' and not recs('vol_test'): vol_test()
