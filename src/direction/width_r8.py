"""Round 8, 5 s direction: layer widths of the conv+GRU (the deep_v2.py 'mt' recipe, a member of the round-3 stacking).
Design fixed before running; chosen on VAL only; TEST is read once, in 'dir_test'.
Model : two 1D convolutions with c channels (kernel 3) -> GRU with h hidden units -> heads; everything else as deep_v2.py
        'mt' (127 'full' columns, 50 snapshots, EMA, auxiliary 1/2/10/30 s heads, round-2 budget and early stop).
        Rounds 1-3 used c = h = 64.
Widths: (c, h) = (32, 32), (32, 64), (64, 32), (64, 128), (128, 64), (128, 128).
screen: every width with seeds 0 and 1. A width passes if its seed-0/1 mean full-VAL macro-F1 beats round 2's mt seeds 0/1
        (results/direction/v2_deep.jsonl, mean 0.5129). Check first: (64, 64) seed 0 with this code must give round 2's mt
        seed 0; if it does not, (64, 64) seed 1 is run as well and the mean of the two re-runs is the reference.
more  : the passing width with the highest mean gets seeds 2-9 (a 10-seed bag, as round 3; all seeds with deep_v2.py's
        training loop).
stack : the round-3 winner (post_r3.py: logistic stacking of the log-probabilities of its 8 members, balanced class
        weights, leave-one-day-out over the 12 VAL days, per-class multipliers on the wide grid) re-run with the new bag in
        place of mt, next to the round-3 pool on the same rows. The new bag is used only if its stacking beats round 3's
        on VAL (0.5194). Check: multipliers chosen on VAL days 1-6 scored on days 7-12 and the reverse, for both.
dir_test: once, only if the new stacking won on VAL: every member on TEST as final_r3.py (the stacker refitted on all
        VAL rows), with the new bag in place of mt; references on the same rows: rounds 2 and 3 (their saved TEST
        predictions); day-block bootstrap (2,000 resamples) of the macro-F1 differences.
Files : results/direction/r8_val_deep_mt_c<c>_h<h>_s<seed>.npz and r8_deep_mt_c<c>_h<h>_s<seed>.pt (as deep_v2.py's),
        r8_val_stack_lr.npz (leave-one-day-out VAL probabilities of the new stacking), r8_test.npz.
Note: the TEST days were already used by every earlier round.
usage: python width_r8.py screen [threads] | more [threads] | summary | stack | dir_test [threads]
       -> results/direction/width_r8.jsonl
"""
import _paths  # noqa: F401  (src/common, src/direction, src/volatility, src/report on the import path)
import json, os, sys, time
import numpy as np
import deep_v2 as D

argv = sys.argv; sys.argv = argv[:1] + ['main']
import post_r3 as PR
sys.argv = argv

WIDTHS = [(32, 32), (32, 64), (64, 32), (64, 128), (128, 64), (128, 128)]
REF = 'mt'; RES = os.path.join(D.OUT, 'width_r8.jsonl')
name = lambda c, h: f'{REF}_c{c}_h{h}'
for c_, h_ in WIDTHS + [(64, 64)]: D.CONFIGS[name(c_, h_)] = dict(D.CONFIGS[REF], conv=c_, hidden=h_)
D.RES, D.PFX = RES, 'r8'


def recs(stage=None):
    R = [json.loads(l) for l in open(RES)] if os.path.exists(RES) else []
    return [r for r in R if stage is None or r.get('stage') == stage]


def runs(): return {(r['name'], r['seed']): r for r in recs() if 'stage' not in r}


def log(rec):
    print(json.dumps(rec)[:600], flush=True)
    with open(RES, 'a') as f: f.write(json.dumps(rec) + '\n')


def reference():
    """round 2's mt seeds 0/1, or the re-runs of (64, 64) here if the check did not reproduce round 2"""
    r2 = {r['seed']: r['val_mF1'] for r in map(json.loads, open(os.path.join(D.OUT, 'v2_deep.jsonl'))) if r['name'] == REF and r['seed'] in (0, 1)}
    R = runs(); chk = R.get((name(64, 64), 0))
    if chk is None or chk['val_mF1'] == r2[0]: return 'round 2 mt seeds 0/1', r2
    return '(64, 64) re-run here, seeds 0/1', {s: R[(name(64, 64), s)]['val_mF1'] for s in (0, 1) if (name(64, 64), s) in R}


def summary(write=True):
    src, ref = reference(); R = runs(); m0 = float(np.mean(list(ref.values())))
    rows = []
    for c, h in WIDTHS:
        v = [R[(name(c, h), s)]['val_mF1'] for s in (0, 1) if (name(c, h), s) in R]
        if len(v) == 2: rows.append(dict(width=[c, h], name=name(c, h), seeds01=v, mean=round(float(np.mean(v)), 4), passes=bool(np.mean(v) > m0),
                                         n_params=R[(name(c, h), 0)]['n_params'], sec=[R[(name(c, h), s)]['sec'] for s in (0, 1)]))
    best = max((r for r in rows if r['passes']), key=lambda r: r['mean'], default=None); chk = R.get((name(64, 64), 0))
    if write: log(dict(stage='screen_summary', reference=src, reference_seeds01=list(ref.values()), reference_mean=round(m0, 4),
                       check_64x64_seed0=chk['val_mF1'] if chk else None, widths=rows, best=best))
    return best


def stack():
    done = [r for r in recs('more_done') if not r.get('skipped')]
    if not done: log(dict(stage='stack_winner', skipped=True, why='no width passed the screen')); return
    t0 = time.time(); bag = done[-1]['width']
    w3 = [json.loads(l) for l in open(os.path.join(D.OUT, 'r3_post.jsonl')) if '"winner"' in l and '"pool": "main"' in l][-1]
    P = PR.pool_files(); names = sorted(w3['members']); assert sorted(P) == names and w3['stack_kind'] == 'lr' and not w3['ctx']
    P['new'] = [os.path.join(D.OUT, f'{D.PFX}_val_deep_{bag}_s{s}.npz') for s in range(10)]
    Pr, keys = PR.load_pool(P); y = Pr.pop('y'); d = keys // 1_000_000; out = {}
    for n in ('mt', 'new'):
        f, mm = PR.best_flat(Pr[n], y); log(dict(stage='bag', name=n if n == 'mt' else bag, n_seeds=len(P[n]), mF1_m1=round(PR.mf1(y, Pr[n].argmax(1)), 4), mF1=round(f, 4), m=mm[1]))
    for tag, sub in (('r3', Pr), ('r8', dict(Pr, mt=Pr['new']))):
        lp = np.log(np.clip(np.hstack([sub[n] for n in names]), 1e-6, 1)); pr = PR.stack_cv(lp, y, d, 'lr', w3['class_weight'])
        f, mm = PR.best_cls(pr, y, grid=PR.MFW); out[tag] = (f, mm, pr)
        log(dict(stage='stack', pool=tag, mt_bag='mt' if tag == 'r3' else bag, members=names, n_rows=int(len(y)), mF1_m1=round(PR.mf1(y, pr.argmax(1)), 4),
                 mF1=round(f, 4), mult=mm, sec=round(time.time() - t0)))
    np.savez_compressed(os.path.join(D.OUT, 'r8_val_stack_lr.npz'), prob=out['r8'][2].astype(np.float32), y=y, day=d, row=keys % 1_000_000)
    days = np.unique(d); halves = (days[:len(days) // 2], days[len(days) // 2:])
    for a_, b_ in ((0, 1), (1, 0)):
        sel, ev = np.where(np.isin(d, halves[a_]))[0], np.isin(d, halves[b_]); h = {}
        for tag in ('r3', 'r8'):
            _, mm = PR.best_cls(out[tag][2], y, sel, grid=PR.MFW); p = PR.apply(out[tag][2], mm); h[tag] = dict(mult=mm, mF1_other_half=round(PR.mf1(y[ev], p[ev]), 4))
        log(dict(stage='stack_half', chosen_on=[int(x) for x in halves[a_]], **h))
    win = 'r8' if round(out['r8'][0], 4) > w3['mF1'] else 'r3'
    log(dict(stage='stack_winner', winner=win, width=bag, mF1=round(out[win][0], 4), mult=out[win][1], r3_val_mF1=w3['mF1'],
             r3_recomputed_here=round(out['r3'][0], 4), r8_val_mF1=round(out['r8'][0], 4), members=names, class_weight=w3['class_weight'],
             sec=round(time.time() - t0)))


def dir_test():
    w = recs('stack_winner')
    if not w or w[-1].get('skipped') or w[-1]['winner'] != 'r8': log(dict(stage='dir_test', skipped=True, why='no round-8 winner on VAL')); return
    import torch
    from sklearn.metrics import accuracy_score, confusion_matrix
    import common as C
    argv_ = sys.argv; sys.argv = argv_[:1]; import final_r3 as F3; sys.argv = argv_
    torch.set_num_threads(int(sys.argv[2]) if len(sys.argv) > 2 else 2)
    w = w[-1]; bag, names = w['width'], w['members']; t0 = time.time(); E = D.E; mf1 = PR.mf1
    te_days = [d for d in E.TE if float(C.load_day(d)['cov']) >= C.MIN_COV]
    cks = [os.path.join(D.OUT, f'{D.PFX}_deep_{bag}_s{s}.pt') for s in range(10)]
    P, K = F3.test_members(names, te_days, deep=lambda n, td: F3.deep_member(n, td, cks if n == 'mt' else None))
    w3 = [json.loads(l) for l in open(os.path.join(D.OUT, 'r3_post.jsonl')) if '"winner"' in l and '"pool": "main"' in l][-1]
    ref = {r: np.load(os.path.join(D.OUT, f)) for r, f in (('r2', 'v2_test_primary.npz'), ('r3', 'r3_test.npz'))}
    mult = dict(r2=[1, 1.1, 1], r3=w3['mult']); keys = None
    for z in ref.values(): k = F3.KEY(z['day'], z['row']); keys = k if keys is None else np.intersect1d(keys, k)
    for n in names: keys = np.intersect1d(keys, K[n])
    for n in names: assert (K[n][np.searchsorted(K[n], keys)] == keys).all()
    Pm = {n: P[n][np.searchsorted(K[n], keys)] for n in names}; d = keys // 1_000_000; y, pr_ = None, {}
    for r, z in ref.items():
        kz = F3.KEY(z['day'], z['row']); i = np.searchsorted(kz, keys); assert (kz[i] == keys).all()
        if y is None: y = z['y'][i].astype(int)
        else: assert (z['y'][i] == y).all()
        pr_[r] = (z['prob'][i] * mult[r]).argmax(1)
    Pf = dict(PR.pool_files(), mt=[os.path.join(D.OUT, f'{D.PFX}_val_deep_{bag}_s{s}.npz') for s in range(10)])
    Pv, kv = PR.load_pool(Pf); yv = Pv.pop('y'); assert sorted(Pv) == names
    n_val = [r for r in recs('stack') if r['pool'] == 'r8'][-1]['n_rows']; assert len(yv) == n_val
    lv = np.log(np.clip(np.hstack([Pv[n] for n in names]), 1e-6, 1)); lt = np.log(np.clip(np.hstack([Pm[n] for n in names]), 1e-6, 1))
    mu, sd = lv.mean(0), lv.std(0) + 1e-6
    prob = PR.stack_model('lr', w['class_weight']).fit(((lv - mu) / sd)[::3], yv[::3]).predict_proba((lt - mu) / sd); p = (prob * w['mult']).argmax(1)
    log(dict(stage='dir_test', pipeline='stack_lr', width=bag, members=names, mult=w['mult'], val_mF1=w['mF1'], n=int(len(y)),
             mF1=round(mf1(y, p), 4), acc=round(float(accuracy_score(y, p)), 4), mF1_m1=round(mf1(y, prob.argmax(1)), 4),
             recall=[round(float(x), 4) for x in confusion_matrix(y, p, normalize='true').diagonal()],
             member_mF1={n: round(mf1(y, Pm[n].argmax(1)), 4) for n in names},
             **{f'{r}_mF1': round(mf1(y, q), 4) for r, q in pr_.items()}, **{f'boot_new_minus_{r}': F3.boot(y, d, p, q) for r, q in pr_.items()},
             days_better_than_r3=int(sum(mf1(y[d == k], p[d == k]) > mf1(y[d == k], pr_['r3'][d == k]) for k in np.unique(d))),
             day_mF1={str(k): round(mf1(y[d == k], p[d == k]), 4) for k in np.unique(d)}, confusion=confusion_matrix(y, p).tolist(),
             sec=round(time.time() - t0)))
    np.savez_compressed(os.path.join(D.OUT, 'r8_test.npz'), prob=prob.astype(np.float32), y=y, day=d, row=keys % 1_000_000)


if __name__ == '__main__':
    what = sys.argv[1]
    if what == 'screen':
        if (name(64, 64), 0) not in runs(): D.run(name(64, 64), 0)
        todo = ([] if reference()[0].startswith('round 2') else [(name(64, 64), 1)]) + [(name(c, h), s) for s in (0, 1) for c, h in WIDTHS]
        for n, s in todo:
            if (n, s) not in runs(): D.run(n, s)
        summary()
    elif what == 'summary': summary()
    elif what == 'more':
        best = summary(write=False)
        if best is None: log(dict(stage='more_done', skipped=True, why='no width passed the screen'))
        else:
            for s in range(2, 10):
                if (best['name'], s) not in runs(): D.run(best['name'], s)
            log(dict(stage='more_done', width=best['name']))
    elif what == 'stack' and not recs('stack_winner'): stack()
    elif what == 'dir_test' and not recs('dir_test'): dir_test()
