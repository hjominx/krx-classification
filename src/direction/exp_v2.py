"""Improvement round 2 for the 5 s direction task: tabular models on the v2 features (features_v2.py).
Everything here is chosen on VAL; TEST is only read by final_v2.py. Design fixed before running.
Days: those with a trade file (07-09 has none): TRAIN 46, VAL 12. Training rows: every 2nd valid row (as before).
GBM settings: the ones chosen earlier on VAL (exp_trades.py stage C): lr 0.05, 15 leaves, min 20 per leaf, l2 1.0,
balanced class weights; 400 trees, VAL macro-F1 read every 50 trees and the best checkpoint reported.
  A : ref77  = book 47 + flow 15 + time 3 + trade_tw 12 (the set kept in exp_trades.py)
      refc77 = the same with 'flowc' (causal depth scale) in place of 'flow'  -> size of the same-day-scale effect
      full   = refc77 + hist, rvc, sprd, ofic, queue, depl, ctx, daylvl (50 new columns)
  B : drop one new group at a time from 'full'; a group is dropped if removing it does not lower VAL macro-F1 by more
      than 0.001 (same rule as exp_trades.py); 'kept' = full minus the dropped groups, used only if not worse than full.
      The feature set for all later stages is the best on VAL among refc77, full and kept.
  C : GBM capacity on that set (5 settings, up to 800 trees, checkpoints of 50)
  T : two-stage alternative on that set: (move vs flat) x (up vs down | move), same GBM settings
  L : logistic regression on that set (standardised, balanced), and on refc77 for reference
  P : VAL probabilities of the best GBM (C or T) and of the logistic model -> results/direction/v2_val_<name>.npz (prob, y, day, row)
usage: python exp_v2.py A B C T L P   -> results/direction/v2.jsonl
The shared parts (columns, day loading, GBM settings, macro-F1 helpers) are in src/common/data_v2.py.
"""
import _paths  # noqa: F401  (src/common, src/direction, src/volatility, src/report on the import path)
import json, os, sys, time
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier as HGB
from sklearn.linear_model import LogisticRegression
import common as C
from data_v2 import (FT, F2, H, hi, TR0, VA, TE, has_tr, TR, MS, columns, NAMES, G, NEW, cols_of,  # noqa: F401  (also used as E.<name>)
                     REF77, REFC77, FULL, load_day, stack, HP0, gbm, mf1, curve_of, day_stats, best_m)

OUT = C.RES_DIR; os.makedirs(OUT, exist_ok=True)   # KRX_RESULTS: separate output folder (e.g. a GPU run)
RES = os.path.join(OUT, 'v2.jsonl')


def log(rec):
    print(json.dumps(rec)[:400], flush=True)
    with open(RES, 'a') as f: f.write(json.dumps(rec) + '\n')


def recs(stage=None):
    if not os.path.exists(RES): return []
    R = [json.loads(l) for l in open(RES)]
    return [r for r in R if stage is None or r.get('stage') == stage]


def done(stage, name): return any(r.get('name') == name for r in recs(stage))


def chosen_set():
    a = {r['name']: r for r in recs('A')}; k = next((r for r in recs('B') if r['name'] == 'kept'), None)
    cand = [('refc77', a['refc77']['mF1'], REFC77), ('full', a['full']['mF1'], FULL)]
    if k and k['use_reduced']: cand.append(('kept', k['mF1'], k['cols']))
    return max(cand, key=lambda c: c[1])


def fit_eval(stage, name, cols, Xtr, ytr, Xva, yva, dva, max_iter=400, **kw):
    t0 = time.time(); m = gbm(max_iter=max_iter, **kw).fit(Xtr[:, cols], ytr)
    cur, it, pr = curve_of(m, Xva[:, cols], yva)
    rec = dict(stage=stage, name=name, n_features=len(cols), mF1=round(mf1(yva, pr.argmax(1)), 4), best_iter=it, curve=cur,
               **day_stats(yva, pr.argmax(1), dva), sec=round(time.time() - t0), **{k: v for k, v in kw.items()})
    log(rec); return rec, pr


if __name__ == '__main__':
    stages = sys.argv[1:]
    t0 = time.time(); Xtr, ytr, _, _ = stack(TR, sub=2); Xva, yva, dva, rva = stack(VA, eval_only=True)
    print('loaded', Xtr.shape, Xva.shape, round(time.time() - t0), 's', flush=True)
    for STAGE in stages:
        if STAGE == 'A':
            for name, cols in (('ref77', REF77), ('refc77', REFC77), ('full', FULL)):
                if not done('A', name): fit_eval('A', name, cols, Xtr, ytr, Xva, yva, dva)
        elif STAGE == 'B':
            full = next(r for r in recs('A') if r['name'] == 'full')
            for g in NEW:
                if done('B', 'drop_' + g): continue
                cols = [c for c in FULL if c not in G[g]]
                r, _ = fit_eval('B', 'drop_' + g, cols, Xtr, ytr, Xva, yva, dva)
                log(dict(stage='B_contrib', name=g, contribution=round(full['mF1'] - r['mF1'], 4)))
            if not done('B', 'kept'):
                drop = [r['name'] for r in recs('B_contrib') if r['contribution'] <= 0.001]
                cols = [c for c in FULL if not any(c in G[g] for g in drop)]
                if drop and len(drop) < len(NEW):
                    t1 = time.time(); m = gbm().fit(Xtr[:, cols], ytr); cur, it, pr = curve_of(m, Xva[:, cols], yva)
                    f = round(mf1(yva, pr.argmax(1)), 4)
                    log(dict(stage='B', name='kept', dropped_groups=drop, n_features=len(cols), mF1=f, best_iter=it, curve=cur,
                             full_mF1=full['mF1'], use_reduced=bool(f >= full['mF1'] - 0.0005), cols=cols, sec=round(time.time() - t1)))
                else:
                    log(dict(stage='B', name='kept', dropped_groups=drop, n_features=len(cols), mF1=full['mF1'] if not drop else None,
                             full_mF1=full['mF1'], use_reduced=False, cols=FULL, note='nothing dropped' if not drop else 'every new group dropped'))
        elif STAGE == 'C':
            sname, sf, cols = chosen_set(); log(dict(stage='C_set', name=sname, val_mF1=sf, n_features=len(cols))) if not done('C_set', sname) else None
            grid = [('l15_m20', dict()), ('l31_m200', dict(max_leaf_nodes=31, min_samples_leaf=200)),
                    ('l7_m200', dict(max_leaf_nodes=7, min_samples_leaf=200)),
                    ('l15_m20_feat50', dict(max_features=0.5)),
                    ('lr.02_l15_m100', dict(learning_rate=0.02, min_samples_leaf=100))]
            for name, kw in grid:
                if not done('C', name): fit_eval('C', name, cols, Xtr, ytr, Xva, yva, dva, max_iter=800, **kw)
        elif STAGE == 'T':
            sname, sf, cols = chosen_set()
            if not done('T', 'two_stage'):
                t1 = time.time(); mv = (ytr != 1).astype(int)
                m1 = HGB(**HP0, max_iter=400, class_weight='balanced', early_stopping=False, random_state=0).fit(Xtr[:, cols], mv)
                k = ytr != 1
                m2 = HGB(**HP0, max_iter=400, class_weight='balanced', early_stopping=False, random_state=0).fit(Xtr[k][:, cols], (ytr[k] == 2).astype(int))
                best = (-1, None, None)
                for (i1, p1), (i2, p2) in zip(enumerate(m1.staged_predict_proba(Xva[:, cols]), 1), enumerate(m2.staged_predict_proba(Xva[:, cols]), 1)):
                    if i1 % 50: continue
                    # balanced 3-class posterior from the two balanced binary models (priors 1/2, 1/4, 1/4 -> 1/3 each)
                    pr = np.column_stack([p1[:, 1] * p2[:, 0] * 4 / 3, p1[:, 0] * 2 / 3, p1[:, 1] * p2[:, 1] * 4 / 3]); pr /= pr.sum(1, keepdims=True)
                    f = mf1(yva, pr.argmax(1))
                    if f > best[0]: best = (f, i1, pr.copy())
                m, fm, tab = best_m(best[2], yva)
                log(dict(stage='T', name='two_stage', set=sname, mF1=round(best[0], 4), best_iter=best[1], best_m=m, mF1_at_best_m=fm, m_table=tab,
                         **day_stats(yva, best[2].argmax(1), dva), sec=round(time.time() - t1)))
                np.savez_compressed(os.path.join(OUT, 'v2_val_gbm2stage.npz'), prob=best[2].astype(np.float32), y=yva, day=dva, row=rva)
        elif STAGE == 'L':
            sname, sf, cols = chosen_set()
            for name, cc in ((f'logreg_{sname}', cols), ('logreg_refc77', REFC77)):
                if done('L', name): continue
                t1 = time.time(); mu, sd = Xtr[:, cc].mean(0), Xtr[:, cc].std(0) + 1e-6
                m = LogisticRegression(max_iter=500, class_weight='balanced').fit(((Xtr[:, cc] - mu) / sd)[::2], ytr[::2])
                pr = m.predict_proba((Xva[:, cc] - mu) / sd)
                log(dict(stage='L', name=name, n_features=len(cc), mF1=round(mf1(yva, pr.argmax(1)), 4), **day_stats(yva, pr.argmax(1), dva), sec=round(time.time() - t1)))
                if name != 'logreg_refc77' or sname == 'refc77':
                    np.savez_compressed(os.path.join(OUT, 'v2_val_logreg.npz'), prob=pr.astype(np.float32), y=yva, day=dva, row=rva)
        elif STAGE == 'P':
            sname, sf, cols = chosen_set(); c = max(recs('C'), key=lambda r: r['mF1'])
            kw = {k: c[k] for k in ('learning_rate', 'max_leaf_nodes', 'min_samples_leaf', 'max_features') if k in c}
            m = gbm(max_iter=c['best_iter'], **kw).fit(Xtr[:, cols], ytr); pr = m.predict_proba(Xva[:, cols])
            mm, fm, tab = best_m(pr, yva)
            log(dict(stage='P', name='gbm', set=sname, config=c['name'], trees=c['best_iter'], mF1=round(mf1(yva, pr.argmax(1)), 4), best_m=mm, mF1_at_best_m=fm, m_table=tab))
            np.savez_compressed(os.path.join(OUT, 'v2_val_gbm.npz'), prob=pr.astype(np.float32), y=yva, day=dva, row=rva)
