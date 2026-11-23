"""Round 3 for the 60 s volatility task (calm / normal / turbulent, class cuts of vol_cls.py): stacking of the
round-2 members. Chosen on VAL only; TEST is read once, in the 'test' stage. Design fixed before running.
Members (all from vol_v2.py, trained on TRAIN, no new training)
  gbm    : GBM classifier on the 145 'allv' columns (VAL probabilities)
  logreg : logistic regression on allv (VAL probabilities)
  gru    : mean of the 3 conv+GRU seeds on allv (VAL probabilities)
  gbr    : GBM regressor on log(1 + RV60) (VAL prediction, a single number per row) - the round-2 winner, with the
           two class thresholds chosen on VAL ([2.0, 2.85], VAL macro-F1 0.6412)
Rows   : the VAL rows every member can score (those of the conv+GRU, 50-snapshot windows)
Candidates (each scored on VAL macro-F1)
  round 2   : the round-2 winner exactly as chosen (gbr, thresholds [2.0, 2.85]); gbr with thresholds re-chosen on
              these rows; each classifier and the plain mean of every subset of {gbm, logreg, gru}
  multiplier: every probability candidate gets a multiplier per class [1, a, b], a and b in 0.50-2.00 (step 0.05)
  stack     : a model on the members' outputs (log-probabilities of gbm, logreg, gru + the gbr prediction), with or
              without time of day, spread and short-term volatility; multinomial logistic regression or a small GBM;
              with balanced or no class weights (8 stackers, the settings of post_r3.py). Scores are leave-one-day-out
              over the 12 VAL days (each model is fitted on every 3rd row of the other 11 days), then the multipliers
Winner : highest VAL macro-F1. Split-half check: the whole choice (model, multipliers, thresholds) on VAL days 1-6,
         scored on days 7-12, and the reverse, next to the round-2 recipe (gbr, thresholds chosen on the same half)
TEST   : members as trained on TRAIN (the saved conv+GRU checkpoints; gbm, logreg, gbr refitted with their chosen
         settings - deterministic, as in vol_v2.py); a stacker is refitted on all VAL rows with the same settings.
         Reference: the round-2 pipeline on TEST (results/volatility/vol_v2_test_train.npz) on the same rows; day-block bootstrap
         (2,000 resamples) of the macro-F1 difference.
Note: the TEST days were already used by vol_cls.py and vol_v2.py, so they are no longer an untouched hold-out.
usage: python vol_r3.py val | test   -> results/volatility/vol_r3.jsonl
"""
import _paths  # noqa: F401  (src/common, src/direction, src/volatility, src/report on the import path)
import json, os, sys, time
from itertools import combinations
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier as HGB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import data_v2 as E

argv = sys.argv; sys.argv = argv[:1]
import vol_v2 as V
sys.argv = argv

OUT = V.OUT; RES = os.path.join(OUT, 'vol_r3.jsonl'); KEY = V.KEY


def mf1(y, p):
    """macro-F1 from a confusion matrix (same value as sklearn's, much faster)"""
    cm = np.bincount(np.asarray(y) * 3 + np.asarray(p), minlength=9).reshape(3, 3); tp = np.diag(cm).astype(float)
    return float(np.mean(2 * tp / np.maximum(cm.sum(0) + cm.sum(1), 1)))


def boot(y, d, p_new, p_ref, n=2000, seed=1):
    """as vol_v2.boot (same resamples), with the fast macro-F1"""
    days = np.unique(d); ix = {k: np.where(d == k)[0] for k in days}; rng = np.random.default_rng(seed); out = []
    for _ in range(n):
        m = np.concatenate([ix[k] for k in rng.choice(days, len(days))]); out.append(mf1(y[m], p_new[m]) - mf1(y[m], p_ref[m]))
    return dict(mean=round(float(np.mean(out)), 4), ci95=[round(float(np.quantile(out, .025)), 4), round(float(np.quantile(out, .975)), 4)],
                share_positive=round(float(np.mean(np.array(out) > 0)), 3))
GRID = np.round(np.arange(0.50, 2.001, 0.05), 3)
CTX = ('min_since_open', 'spread_ticks', 'rv_10s', 'rv_300s'); R2_THR = [2.0, 2.85]


def log(rec):
    print(json.dumps(rec)[:400], flush=True)
    with open(RES, 'a') as f: f.write(json.dumps(rec) + '\n')


def recs(stage=None):
    R = [json.loads(l) for l in open(RES)] if os.path.exists(RES) else []
    return [r for r in R if stage is None or r.get('stage') == stage]


def apply(pr, mm): return (pr * mm).argmax(1)


def thr_cls(reg, t): return (reg > t[0]).astype(int) + (reg > t[1])


def best_mult(pr, y, rows=None):
    if rows is not None: pr, y = pr[rows], y[rows]
    best = (-1, None)
    for a in GRID:
        for b in GRID:
            f = mf1(y, apply(pr, [1, a, b]))
            if f > best[0]: best = (f, [1.0, float(a), float(b)])
    return best


def best_thr(reg, y, rows=None):
    """the threshold grid of vol_v2.best_thr"""
    if rows is not None: reg, y = reg[rows], y[rows]
    best = (-1, None)
    for a in np.arange(1.5, 2.71, 0.05):
        lo = (reg > a).astype(int)
        for b in np.arange(2.6, 3.71, 0.05):
            if b <= a: continue
            f = mf1(y, lo + (reg > b))
            if f > best[0]: best = (f, [float(round(a, 2)), float(round(b, 2))])
    return best


def context(day, row):
    names = [str(n) for n in E.NAMES]; want = [names.index(n) for n in CTX]; out = np.zeros((len(row), len(want)), np.float32)
    for d in np.unique(day):
        x, _, r, _ = E.load_day(str(d), need_valid=False); m = day == d; assert (r == np.arange(len(r))).all(); out[m] = x[row[m]][:, want]
    return out


def features(P, reg, ctx=None):
    f = np.hstack([np.log(np.clip(P[n], 1e-6, 1)) for n in ('gbm', 'logreg', 'gru')] + [reg[:, None]])
    return np.hstack([f, ctx]) if ctx is not None else f


def stack_model(kind, cw):
    if kind == 'lr': return LogisticRegression(max_iter=300, class_weight=cw)
    return HGB(max_iter=150, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=500, class_weight=cw, random_state=0)


def fit_predict(kind, cw, Xf, yf, Xp):
    if kind == 'lr':
        mu, sd = Xf.mean(0), Xf.std(0) + 1e-6; return stack_model(kind, cw).fit(((Xf - mu) / sd)[::3], yf[::3]).predict_proba((Xp - mu) / sd)
    return stack_model(kind, cw).fit(Xf[::3], yf[::3]).predict_proba(Xp)


def stack_cv(X, y, d, kind, cw):
    pr = np.zeros((len(y), 3))
    for k in np.unique(d):
        tr, te = d != k, d == k; pr[te] = fit_predict(kind, cw, X[tr], y[tr], X[te])
    return pr


def val_members():
    Z = {n: np.load(os.path.join(OUT, f'vol_v2_val_{n}.npz')) for n in ('gbm', 'logreg', 'gru_s0', 'gru_s1', 'gru_s2', 'gbr')}
    k0 = KEY(Z['gru_s0']['day'], Z['gru_s0']['row']); y = Z['gru_s0']['y'].astype(int)
    for s in (1, 2): assert (KEY(Z[f'gru_s{s}']['day'], Z[f'gru_s{s}']['row']) == k0).all()

    def on(z):
        kz = KEY(z['day'], z['row']); ix = np.searchsorted(kz, k0); assert (kz[ix] == k0).all() and (z['y'][ix] == y).all(); return ix
    P = {n: Z[n]['prob'][on(Z[n])].astype(np.float64) for n in ('gbm', 'logreg')}
    P['gru'] = np.mean([Z[f'gru_s{s}']['prob'] for s in (0, 1, 2)], 0).astype(np.float64)
    return P, Z['gbr']['pred'][on(Z['gbr'])].astype(np.float64), y, Z['gru_s0']['day'], Z['gru_s0']['row']


def candidates(P, reg, S, y, rows=None):
    """every candidate chosen on `rows`: (score on rows, name, spec)"""
    C_ = []
    f, t = best_thr(reg, y, rows); C_.append((f, 'gbr', dict(method='gbr', thr=t)))
    for k in (1, 2, 3):
        for sub in combinations(('gbm', 'logreg', 'gru'), k):
            f, mm = best_mult(np.mean([P[n] for n in sub], 0), y, rows); C_.append((f, 'avg:' + '+'.join(sub), dict(method='avg', members=list(sub), mult=mm)))
    for nm, (pr, kind, ctx, cw) in S.items():
        f, mm = best_mult(pr, y, rows); C_.append((f, nm, dict(method='stack', stack_kind=kind, ctx=ctx, class_weight=cw, mult=mm)))
    return sorted(C_, key=lambda c: -c[0])


def predict(spec, P, reg, S, name):
    if spec['method'] == 'gbr': return thr_cls(reg, spec['thr'])
    if spec['method'] == 'avg': return apply(np.mean([P[n] for n in spec['members']], 0), spec['mult'])
    return apply(S[name][0], spec['mult'])


def stage_val():
    t0 = time.time(); P, reg, y, d, row = val_members()
    log(dict(stage='info', n_rows=int(len(y)), n_days=int(len(np.unique(d))), class_shares=[round(float(x), 4) for x in np.bincount(y, minlength=3) / len(y)]))
    r2 = thr_cls(reg, R2_THR); log(dict(stage='r2_winner', thr=R2_THR, mF1=round(mf1(y, r2), 4)))
    for n in ('gbm', 'logreg', 'gru'): f, mm = best_mult(P[n], y); log(dict(stage='member', name=n, mF1_m1=round(mf1(y, P[n].argmax(1)), 4), mF1=round(f, 4), mult=mm))
    ctx = context(d, row); S = {}
    for kind in ('lr', 'gbm'):
        for use_ctx in (False, True):
            for cw in ('balanced', None):
                nm = f'stack_{kind}' + ('_ctx' if use_ctx else '') + ('' if cw else '_uw')
                pr = stack_cv(features(P, reg, ctx if use_ctx else None), y, d, kind, cw); S[nm] = (pr, kind, use_ctx, cw)
                np.savez_compressed(os.path.join(OUT, f'vol_r3_val_{nm}.npz'), prob=pr.astype(np.float32), y=y, day=d, row=row)
                f, mm = best_mult(pr, y)
                log(dict(stage='stack', name=nm, ctx=list(CTX) if use_ctx else [], class_weight=cw, mF1_m1=round(mf1(y, pr.argmax(1)), 4), mF1=round(f, 4), mult=mm,
                         sec=round(time.time() - t0)))
    C_ = candidates(P, reg, S, y)
    for f, nm, spec in C_: log(dict(stage='cand', name=nm, mF1=round(f, 4), **spec))
    f, nm, spec = C_[0]; p = predict(spec, P, reg, S, nm)
    log(dict(stage='winner', name=nm, mF1=round(f, 4), **spec, acc=round(float(accuracy_score(y, p)), 4),
             recall=[round(float(x), 4) for x in confusion_matrix(y, p, normalize='true').diagonal()], r2_winner_mF1=round(mf1(y, r2), 4)))
    days = np.unique(d); halves = (days[:len(days) // 2], days[len(days) // 2:])
    for a, b in ((0, 1), (1, 0)):
        sel, ev = np.where(np.isin(d, halves[a]))[0], np.isin(d, halves[b])
        f, nm, spec = candidates(P, reg, S, y, sel)[0]; p = predict(spec, P, reg, S, nm); _, t = best_thr(reg, y, sel)
        log(dict(stage='half', chosen_on=[int(x) for x in halves[a]], chosen=nm, spec=spec, mF1_other_half=round(mf1(y[ev], p[ev]), 4),
                 r2_recipe_other_half=round(mf1(y[ev], thr_cls(reg[ev], t)), 4), r2_thr_on_half=t))
    log(dict(stage='val_done', sec=round(time.time() - t0)))


def stage_test(threads):
    t0 = time.time(); w = recs('winner')[-1]; a = {r['name']: r for r in V.recs('A')}; R = V.recs('R')[-1]
    os.environ['OMP_NUM_THREADS'] = str(threads)
    Gs = [V.gru_apply(s, E.TE) for s in (0, 1, 2)]; kg = KEY(Gs[0][2], Gs[0][3])
    for g in Gs[1:]: assert (KEY(g[2], g[3]) == kg).all()
    print('gru done', round(time.time() - t0), flush=True)
    Xte, yte, _, dte, rte = V.stack(E.TE, eval_only=True); kte = KEY(dte, rte); keep = np.isin(kte, kg); assert (kte[keep] == kg).all() and (yte[keep] == Gs[0][1]).all()
    Xt, yt, lt, _, _ = V.stack(E.TR, sub=10); X = Xte[keep][:, V.ALLV]
    P = dict(gru=np.mean([g[0] for g in Gs], 0), gbm=V.gbc(a['gbm_allv']['best_iter']).fit(Xt[:, V.ALLV], yt).predict_proba(X))
    mu, sd = Xt[:, V.ALLV].mean(0), Xt[:, V.ALLV].std(0) + 1e-6
    P['logreg'] = LogisticRegression(max_iter=500, class_weight='balanced').fit((Xt[:, V.ALLV] - mu) / sd, yt).predict_proba((X - mu) / sd)
    reg = V.gbr(R['best_iter']).fit(Xt[:, V.ALLV], lt).predict(X); del Xt, Xte
    print('members done', round(time.time() - t0), flush=True)
    y, d, row = yte[keep], dte[keep], rte[keep]; k = KEY(d, row)
    ref = np.load(os.path.join(OUT, 'vol_v2_test_train.npz')); kr = KEY(ref['day'], ref['row']); ix = np.searchsorted(kr, k)
    assert (kr[ix] == k).all() and (ref['y'][ix] == y).all(); pref = ref['pred'][ix]
    same = float(np.mean(thr_cls(reg, R2_THR) == pref))                     # the refitted gbr must reproduce round 2
    if w['method'] == 'gbr': p, prob = thr_cls(reg, w['thr']), None
    elif w['method'] == 'avg': prob = np.mean([P[n] for n in w['members']], 0); p = apply(prob, w['mult'])
    else:
        Pv, regv, yv, dv, rowv = val_members(); ctx = w['ctx']
        Xv = features(Pv, regv, context(dv, rowv) if ctx else None); Xp = features(P, reg, context(d, row) if ctx else None)
        prob = fit_predict(w['stack_kind'], w['class_weight'], Xv, yv, Xp); p = apply(prob, w['mult'])
    rec = dict(stage='test', pipeline=w['name'], method=w['method'], val_mF1=w['mF1'], n=int(len(y)),
               class_shares=[round(float(x), 4) for x in np.bincount(y, minlength=3) / len(y)],
               mF1=round(mf1(y, p), 4), acc=round(float(accuracy_score(y, p)), 4),
               recall=[round(float(x), 4) for x in confusion_matrix(y, p, normalize='true').diagonal()],
               member_mF1={n: round(mf1(y, P[n].argmax(1)), 4) for n in P}, gbr_r2_thr_mF1=round(mf1(y, thr_cls(reg, R2_THR)), 4),
               refit_gbr_agrees_with_r2=round(same, 6), r2_pipeline_mF1=round(mf1(y, pref), 4), boot_new_minus_r2=boot(y, d, p, pref),
               days_better=int(sum(mf1(y[d == j], p[d == j]) > mf1(y[d == j], pref[d == j]) for j in np.unique(d))),
               day_mF1={str(j): round(mf1(y[d == j], p[d == j]), 4) for j in np.unique(d)},
               r2_day_mF1={str(j): round(mf1(y[d == j], pref[d == j]), 4) for j in np.unique(d)},
               confusion=confusion_matrix(y, p).tolist(), sec=round(time.time() - t0))
    log(rec)
    np.savez_compressed(os.path.join(OUT, 'vol_r3_test.npz'), pred=p, y=y, day=d, row=row, **({'prob': prob.astype(np.float32)} if prob is not None else {}))


if __name__ == '__main__':
    if sys.argv[1] == 'val': stage_val()
    elif sys.argv[1] == 'test':
        if recs('test'): print('TEST already scored'); sys.exit()
        stage_test(int(sys.argv[2]) if len(sys.argv) > 2 else 4)
