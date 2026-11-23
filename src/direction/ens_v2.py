"""Round 2, 5 s direction: choice of the final pipeline on VAL. TEST is not read here. Design fixed before running.
Members   : gbm    - the better on VAL of the multiclass GBM (exp_v2.py stage P) and the two-stage GBM (stage T)
            logreg - logistic regression (stage L)
            one bag per deep config that reached stage D2: plain mean of the probabilities of its 5 seeds
Candidates: the plain (unweighted) mean of the members' probabilities for every non-empty subset of the members,
            followed by a flat-class multiplier m from exp_v2.MS chosen on VAL.
Winner    : highest VAL macro-F1 on the rows every member can score (windows of 50 snapshots).
Also reported: each seed of each bag, each bag, and a split-half check (pipeline chosen on VAL days 1-6 scored on days
7-12 and the reverse, against the single best earlier model) to show how much of the gain is selection.
usage: python ens_v2.py -> results/direction/v2_ens.jsonl
"""
import _paths  # noqa: F401  (src/common, src/direction, src/volatility, src/report on the import path)
import json, os
from itertools import combinations
import numpy as np
import exp_v2 as E

OUT = E.OUT; RES = os.path.join(OUT, 'v2_ens.jsonl'); mf1 = E.mf1
KEY = lambda z: z['day'].astype(np.int64) * 1_000_000 + z['row']


def log(rec):
    print(json.dumps(rec)[:300], flush=True)
    with open(RES, 'a') as f: f.write(json.dumps(rec) + '\n')


def deep_runs():
    R = [json.loads(l) for l in open(os.path.join(OUT, 'v2_deep.jsonl'))]; by = {}
    for r in R: by.setdefault(r['name'], {})[r['seed']] = r
    return by


def members():
    by = deep_runs(); bags = sorted(n for n, s in by.items() if len(s) >= 5)
    Z = {f'{n}_s{s}': np.load(os.path.join(OUT, f'v2_val_deep_{n}_s{s}.npz')) for n in bags for s in sorted(by[n])}
    z0 = next(iter(Z.values())); k0 = KEY(z0); y = z0['y']; d = z0['day']
    for z in Z.values(): assert (KEY(z) == k0).all()

    def on(z):
        kz = KEY(z); ix = np.searchsorted(kz, k0); assert (kz[ix] == k0).all() and (z['y'][ix] == y).all(); return z['prob'][ix]

    P = {}
    t = next((r for r in E.recs('T')), None); p = E.recs('P')[-1]
    two = t is not None and t['mF1_at_best_m'] > p['mF1_at_best_m']
    P['gbm'] = on(np.load(os.path.join(OUT, 'v2_val_gbm2stage.npz' if two else 'v2_val_gbm.npz')))
    P['logreg'] = on(np.load(os.path.join(OUT, 'v2_val_logreg.npz')))
    for n in bags: P[n] = np.mean([Z[f'{n}_s{s}']['prob'] for s in sorted(by[n])], 0)
    return P, Z, y, d, bags, ('two_stage' if two else 'multiclass')


def score(pr, y, rows=None):
    if rows is not None: pr, y = pr[rows], y[rows]
    s = {m: mf1(y, (pr * [1, m, 1]).argmax(1)) for m in E.MS}; m = max(s, key=s.get); return s[m], m, s[1.0]


if __name__ == '__main__':
    P, Z, y, d, bags, gk = members()
    log(dict(stage='info', n_rows=int(len(y)), bags=bags, gbm_kind=gk, n_days=int(len(np.unique(d)))))
    for n, z in Z.items(): log(dict(stage='single', name=n, mF1=round(mf1(y, z['prob'].argmax(1)), 4)))
    for n in P: f, m, f1 = score(P[n], y); log(dict(stage='member', name=n, mF1_m1=round(f1, 4), best_m=m, mF1=round(f, 4)))
    names = list(P); cands = {}
    for k in range(1, len(names) + 1):
        for sub in combinations(names, k):
            pr = np.mean([P[n] for n in sub], 0); f, m, f1 = score(pr, y); cands['+'.join(sub)] = (f, m, sub)
            log(dict(stage='cand', name='+'.join(sub), mF1_m1=round(f1, 4), best_m=m, mF1=round(f, 4)))
    best = max(cands, key=lambda n: cands[n][0])
    log(dict(stage='winner', name=best, members=list(cands[best][2]), m=cands[best][1], val_mF1=round(cands[best][0], 4), gbm_kind=gk))
    # split-half check
    days = np.unique(d); halves = (days[:len(days) // 2], days[len(days) // 2:])
    single = max(Z, key=lambda n: mf1(y, Z[n]['prob'].argmax(1)))            # best single run on all of VAL (favours the baseline)
    for a, b in ((0, 1), (1, 0)):
        sel, ev = np.isin(d, halves[a]), np.isin(d, halves[b]); best_h = (-1,)
        for n, (_, _, sub) in cands.items():
            pr = np.mean([P[q] for q in sub], 0); f, m, _ = score(pr, y, sel)
            if f > best_h[0]: best_h = (f, n, m, pr)
        log(dict(stage='half', chosen_on=[int(x) for x in halves[a]], chosen=best_h[1], m=best_h[2],
                 mF1_other_half=round(mf1(y[ev], (best_h[3][ev] * [1, best_h[2], 1]).argmax(1)), 4),
                 best_single_run=single, single_other_half=round(mf1(y[ev], Z[single]['prob'][ev].argmax(1)), 4),
                 gbm_other_half=round(mf1(y[ev], P['gbm'][ev].argmax(1)), 4)))
