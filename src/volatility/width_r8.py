"""Round 8, 60 s volatility: layer widths of the conv+GRU regression (the r5.py 'vreg_mh' recipe, the conv+GRU member of
the round-5 and round-7 blends). Design fixed before running; chosen on VAL only; TEST is read once, in 'vol_test'.
Model : two 1D convolutions with c channels (kernel 3) -> GRU with h hidden units -> outputs; everything else as r5.py
        'vreg_mh' (145 allv columns, 50 snapshots, log(1 + RV) of the next 10/30/60/120 s, EMA, lr 1e-3, dropout 0.2,
        round-2 budget and early stop). Round 5 used c = h = 64.
Widths: (c, h) = (32, 32), (32, 64), (64, 32), (64, 128), (128, 64), (128, 128).
screen: every width with seeds 0 and 1. A width passes if its seed-0/1 mean VAL macro-F1 beats round 5's vreg_mh seeds 0/1
        (results/volatility/r5.jsonl, mean 0.6340). Check first: (64, 64) seed 0 with this code must give round 5's vreg_mh
        seed 0; if it does not, (64, 64) seed 1 is run as well and the mean of the two re-runs is the reference.
more  : the passing width with the highest mean gets seeds 2, 3, 4 (a 5-seed bag, as round 5).
vol_val: on the VAL rows every member scores, as mkt_r7.py: the round-7 market GBM blended with a conv+GRU bag,
        w x GBM + (1 - w) x bag for w = 0.1 ... 0.9 with thresholds chosen on VAL, once with the round-5 bag (the round-7
        recipe) and once with the new bag. Winner: highest VAL macro-F1; if the round-5 bag wins there is no TEST run.
        Split-half check: the choice on VAL days 1-6 scored on days 7-12 and the reverse, next to the round-7 recipe
        re-chosen on the same half.
vol_test: once; references on the same rows: rounds 2, 5 and 7 (their saved TEST predictions); day-block bootstrap
        (2,000 resamples) of the macro-F1 differences.
Files : results/volatility/r8_val_w<c>x<h>_s<seed>_vol.npz and r8_w<c>x<h>_s<seed>_vol.pt (as r5.py's).
Note: the TEST days were already used by every earlier round.
usage: python width_r8.py screen [threads] | more [threads] | summary | vol_val | vol_test [threads]
       -> results/volatility/width_r8.jsonl
"""
import _paths  # noqa: F401  (src/common, src/direction, src/volatility, src/report on the import path)
import json, os, sys, time
import numpy as np, torch
from sklearn.metrics import accuracy_score, confusion_matrix
import common as C
import data_v2 as E

argv = sys.argv; sys.argv = argv[:1]
import seq_models as D  # noqa: F401
import vol_r4 as R4
import r5 as R5
sys.argv = argv

OUT = C.RES_VOL; RES = os.path.join(OUT, 'width_r8.jsonl')
WIDTHS = [(32, 32), (32, 64), (64, 32), (64, 128), (128, 64), (128, 128)]
REF = 'vreg_mh'; WS = np.round(np.arange(0.1, 0.91, 0.1), 2)
name = lambda c, h: f'w{c}x{h}'
for c_, h_ in WIDTHS + [(64, 64)]: R5.CFG[name(c_, h_)] = dict(R5.CFG[REF], conv=c_, hidden=h_)
R5.RES, R5.PFX = RES, 'r8'


def recs(stage=None):
    R = [json.loads(l) for l in open(RES)] if os.path.exists(RES) else []
    return [r for r in R if stage is None or r.get('stage') == stage]


def runs(): return {(r['name'], r['seed']): r for r in recs('train') if r['task'] == 'vol'}


def log(rec):
    print(json.dumps(rec)[:600], flush=True)
    with open(RES, 'a') as f: f.write(json.dumps(rec) + '\n')


def reference():
    """round 5's vreg_mh seeds 0/1, or the re-runs of (64, 64) here if the check did not reproduce round 5"""
    r5 = {r['seed']: r['val_mF1'] for r in map(json.loads, open(os.path.join(OUT, 'r5.jsonl')))
          if r.get('stage') == 'train' and r['name'] == REF and r['task'] == 'vol' and r['seed'] in (0, 1)}
    R = runs(); chk = R.get((name(64, 64), 0))
    if chk is None or chk['val_mF1'] == r5[0]: return 'round 5 vreg_mh seeds 0/1', r5
    return '(64, 64) re-run here, seeds 0/1', {s: R[(name(64, 64), s)]['val_mF1'] for s in (0, 1) if (name(64, 64), s) in R}


def summary(write=True):
    src, ref = reference(); R = runs(); m0 = float(np.mean(list(ref.values()))); rows = []
    for c, h in WIDTHS:
        v = [R[(name(c, h), s)]['val_mF1'] for s in (0, 1) if (name(c, h), s) in R]
        if len(v) == 2: rows.append(dict(width=[c, h], name=name(c, h), seeds01=v, mean=round(float(np.mean(v)), 4), passes=bool(np.mean(v) > m0),
                                         n_params=sum(q.numel() for q in D.ConvGRU(len(R5.V.ALLV), h, 0.2, 4, c=c).parameters()),
                                         sec=[R[(name(c, h), s)]['sec'] for s in (0, 1)]))
    best = max((r for r in rows if r['passes']), key=lambda r: r['mean'], default=None); chk = R.get((name(64, 64), 0))
    if write: log(dict(stage='screen_summary', reference=src, reference_seeds01=list(ref.values()), reference_mean=round(m0, 4),
                       check_64x64_seed0=chk['val_mF1'] if chk else None, widths=rows, best=best))
    return best


def pool(best):
    files = dict(new=[os.path.join(OUT, 'r7_val_vol_mkt.npz')], cg=[os.path.join(OUT, f'r5_val_{REF}_s{s}_vol.npz') for s in range(5)],
                 w8=[os.path.join(OUT, f'r8_val_{best}_s{s}_vol.npz') for s in range(5)])
    Z = {n: [np.load(f) for f in fs] for n, fs in files.items()}; keys = None
    for zs in Z.values():
        for z in zs: k = R4.KEY(z['day'], z['row']); keys = k if keys is None else np.intersect1d(keys, k)
    P, y = {}, None
    for n, zs in Z.items():
        q = []
        for z in zs:
            k = R4.KEY(z['day'], z['row']); ix = np.searchsorted(k, keys); assert (k[ix] == keys).all(); q.append(z['pred'][ix].astype(np.float64))
            if y is None: y = z['y'][ix].astype(int)
            else: assert (z['y'][ix] == y).all()
        P[n] = np.mean(q, 0)
    return P, y, keys


def cands(P, y, rows=None):
    out = []
    for bag in ('cg', 'w8'):
        for w in WS:
            f, t = R4.tune(w * P['new'] + (1 - w) * P[bag], y, rows); out.append((f, f'mkt+{bag}_{w}', dict(bag=bag, w=float(w), thr=t)))
    return sorted(out, key=lambda c: -c[0])


def apply(sp, P): return R4.by_thr(sp['w'] * P['new'] + (1 - sp['w']) * P[sp['bag']], sp['thr'])


def vol_val():
    best = [r for r in recs('screen_summary')][-1]['best']
    if best is None or len(recs('more_done')) == 0: log(dict(stage='vol_winner', skipped=True, why='no width passed the screen')); return
    t0 = time.time(); P, y, keys = pool(best['name']); d = keys // 1_000_000
    C_ = cands(P, y)
    for f, nm, sp in C_: log(dict(stage='vol_cand', name=nm, mF1=round(f, 4), **sp))
    f, nm, sp = C_[0]; p = apply(sp, P)
    log(dict(stage='vol_winner', name=nm, width=best['name'], mF1=round(f, 4), **sp, acc=round(float(accuracy_score(y, p)), 4), n_rows=int(len(y))))
    days = np.unique(d); halves = (days[:len(days) // 2], days[len(days) // 2:])
    for a_, b_ in ((0, 1), (1, 0)):
        sel, ev = np.where(np.isin(d, halves[a_]))[0], np.isin(d, halves[b_])
        f, nm, sp = cands(P, y, sel)[0]; p = apply(sp, P)
        r7 = max((c for c in cands(P, y, sel) if c[2]['bag'] == 'cg'), key=lambda c: c[0]); q = apply(r7[2], P)
        log(dict(stage='vol_half', chosen_on=[int(x) for x in halves[a_]], chosen=nm, spec=sp, mF1_other_half=round(R4.mf1(y[ev], p[ev]), 4),
                 r7_recipe=r7[1], r7_recipe_other_half=round(R4.mf1(y[ev], q[ev]), 4)))
    log(dict(stage='vol_val_done', sec=round(time.time() - t0)))


def vol_test():
    w = recs('vol_winner')
    if not w or w[-1].get('skipped') or w[-1]['bag'] == 'cg': log(dict(stage='vol_test', skipped=True, why='no round-8 winner on VAL')); return
    import pickle
    argv_ = sys.argv; sys.argv = argv_[:1]; import mkt_r7 as M7; import r5_combine as RC; sys.argv = argv_
    w = w[-1]; t0 = time.time(); g = pickle.load(open(os.path.join(OUT, 'r7_vol_mkt.pkl'), 'rb'))
    Xte, rvt, yte, rte, dte = M7.vol_stack(E.TE, True, eval_only=True)
    new = next(p for it, p in enumerate(g['model'].staged_predict(Xte), 1) if it == g['best_iter']); kt = R4.KEY(dte, rte); del Xte
    bag, kb = RC.test_bag(w['width'], pfx='r8')
    ref = {r: np.load(os.path.join(OUT, f)) for r, f in (('r2', 'vol_v2_test_train.npz'), ('r5', 'r5_vol_test.npz'), ('r7', 'r7_vol_test.npz'))}
    keys = np.intersect1d(kt, kb)
    for z in ref.values(): keys = np.intersect1d(keys, R4.KEY(z['day'], z['row']))
    ix = lambda k: np.searchsorted(k, keys); y = yte[ix(kt)]; d = keys // 1_000_000
    x = w['w'] * new[ix(kt)] + (1 - w['w']) * bag[ix(kb)]; p = R4.by_thr(x, w['thr']); pr_ = {}
    for r, z in ref.items():
        i = ix(R4.KEY(z['day'], z['row'])); assert (z['y'][i] == y).all(); pr_[r] = z['pred'][i]
    log(dict(stage='vol_test', pipeline=w['name'], width=w['width'], spec={k: w[k] for k in ('bag', 'w', 'thr')}, val_mF1=w['mF1'], n=int(len(y)),
             mF1=round(R4.mf1(y, p), 4), acc=round(float(accuracy_score(y, p)), 4),
             recall=[round(float(v), 4) for v in confusion_matrix(y, p, normalize='true').diagonal()],
             **{f'{r}_mF1': round(R4.mf1(y, q), 4) for r, q in pr_.items()}, **{f'boot_new_minus_{r}': R4.boot(y, d, p, q) for r, q in pr_.items()},
             days_better_than_r7=int(sum(R4.mf1(y[d == k], p[d == k]) > R4.mf1(y[d == k], pr_['r7'][d == k]) for k in np.unique(d))),
             day_mF1={str(k): round(R4.mf1(y[d == k], p[d == k]), 4) for k in np.unique(d)}, confusion=confusion_matrix(y, p).tolist(),
             sec=round(time.time() - t0)))
    np.savez_compressed(os.path.join(OUT, 'r8_vol_test.npz'), x=x.astype(np.float32), pred=p, y=y, day=d, row=keys % 1_000_000)


if __name__ == '__main__':
    what = sys.argv[1]
    if what in ('screen', 'more', 'vol_test'): torch.set_num_threads(int(sys.argv[2]) if len(sys.argv) > 2 else 2)
    if what == 'screen':
        if (name(64, 64), 0) not in runs(): R5.run(name(64, 64), 0)
        todo = ([] if reference()[0].startswith('round 5') else [(name(64, 64), 1)]) + [(name(c, h), s) for s in (0, 1) for c, h in WIDTHS]
        for n, s in todo:
            if (n, s) not in runs(): R5.run(n, s)
        summary()
    elif what == 'summary': summary()
    elif what == 'more':
        best = summary(write=False)
        if best is None: log(dict(stage='more_done', skipped=True, why='no width passed the screen'))
        else:
            for s in (2, 3, 4):
                if (best['name'], s) not in runs(): R5.run(best['name'], s)
            log(dict(stage='more_done', width=best['name']))
    elif what == 'vol_val' and not recs('vol_winner'): vol_val()
    elif what == 'vol_test' and not recs('vol_test'): vol_test()
