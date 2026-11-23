"""Improvement round 2 for the 60 s volatility task (calm / normal / turbulent; class cuts fixed in vol_cls.py).
Chosen on VAL only; TEST is read once, in the 'test' stage. Design fixed before running.
Days with a trade file: TRAIN 46, VAL 12. Training rows: every 10th valid row (as in vol_cls.py).
Feature sets
  v0   : the 74 columns of vol_cls.py (71 base + log past RV over 10/60/300 s)                        -> reference
  allv : every causal column: book, price, time, flowc, all trade groups and the 8 v2 groups (145)
Stages
  A : GBM classifier (lr 0.05, 31 leaves, balanced; up to 600 trees, VAL macro-F1 every 50 trees) on v0 and allv
  B : allv without daylvl; allv without rvc and hist        (where the gain, if any, comes from)
  R : GBM regressor on log(1 + RV60) with allv; classes from two thresholds on the prediction chosen on VAL
  L : logistic regression on allv
  G : conv+GRU with EMA on allv, seeds 0-2 (T = 50, round-2 budget of seq_models.py)
  E : candidates on VAL: every single model, the plain average of the classifier probabilities (GBM, logistic, GRU bag
      and their subsets) with a 'normal'-class multiplier from MS, and the regression rule. Best VAL macro-F1 wins.
  test : the winner on TEST once, trained on TRAIN (primary) and re-trained on TRAIN + VAL with frozen settings
         (secondary); the v0 GBM recipe is scored on the same rows for a paired day-block bootstrap.
usage: python vol_v2.py A B R L | G | E | test   -> results/volatility/vol_v2.jsonl
"""
import _paths  # noqa: F401  (src/common, src/direction, src/volatility, src/report on the import path)
import json, os, sys, time
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier as HGB, HistGradientBoostingRegressor as HGR
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
import common as C
import data_v2 as E

OUT = C.RES_VOL; os.makedirs(OUT, exist_ok=True); RES = os.path.join(OUT, 'vol_v2.jsonl'); VD = os.path.join(C.DATA, 'feat_vol')
CUTS = [json.loads(l) for l in open(os.path.join(OUT, 'vol_cls.jsonl')) if '"labels"' in l][-1]['cuts']
cls = lambda x: np.digitize(x, CUTS, right=True)
PAST = [160, 161, 162]                                             # log past RV 10/60/300 s of vol_cls.py, appended
V0 = sorted(sum((E.G[g] for g in ('book', 'price', 'flow', 'time')), [])) + PAST
ALLV = [c for c in range(160) if c not in E.G['flow']]
MS = (0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4)
mf1 = E.mf1


def load_day(d):
    z = C.load_day(d); v = np.load(os.path.join(VD, f'{d}.npz')); r = np.where(v['ok'])[0]
    X = np.hstack([z['X'], np.load(os.path.join(E.FT, f'{d}.npz'))['T'], np.load(os.path.join(E.F2, f'{d}.npz'))['X2'], v['past']]).astype(np.float32)
    return X[r], cls(v['rv'][r]).astype(int), np.log1p(v['rv'][r]).astype(np.float32), r, float(z['cov'])


def stack(days, sub=1, eval_only=False):
    X, Y, L, D, R = [], [], [], [], []
    for d in days:
        x, y, l, r, cov = load_day(d)
        if eval_only and cov < C.MIN_COV: continue
        X.append(x[::sub]); Y.append(y[::sub]); L.append(l[::sub]); R.append(r[::sub]); D.append(np.full(len(y[::sub]), int(d)))
    return tuple(np.concatenate(a) for a in (X, Y, L, D, R))


def log(rec):
    print(json.dumps(rec)[:400], flush=True)
    with open(RES, 'a') as f: f.write(json.dumps(rec) + '\n')


def recs(stage=None):
    R = [json.loads(l) for l in open(RES)] if os.path.exists(RES) else []
    return [r for r in R if stage is None or r.get('stage') == stage]


def done(stage, name): return any(r.get('name') == name for r in recs(stage))


def gbc(max_iter=600): return HGB(learning_rate=0.05, max_iter=max_iter, class_weight='balanced', early_stopping=False, random_state=0)


def gbr(max_iter=600): return HGR(learning_rate=0.05, max_iter=max_iter, early_stopping=False, random_state=0)


def stats(y, p, d):
    pd_ = [mf1(y[d == k], p[d == k]) for k in np.unique(d)]
    return dict(acc=round(float(accuracy_score(y, p)), 4), day_mean=round(float(np.mean(pd_)), 4), day_min=round(float(np.min(pd_)), 4),
                recall=[round(float(x), 4) for x in confusion_matrix(y, p, normalize='true').diagonal()])


def best_m(prob, y):
    s = {m: mf1(y, (prob * [1, m, 1]).argmax(1)) for m in MS}; m = max(s, key=s.get); return m, round(s[m], 4)


def best_thr(pred, y):
    best = (-1, 0, 0)
    for a in np.arange(1.5, 2.71, 0.05):
        lo = pred > a
        for b in np.arange(2.6, 3.71, 0.05):
            if b <= a: continue
            f = mf1(y, lo.astype(int) + (pred > b))
            if f > best[0]: best = (f, float(round(a, 2)), float(round(b, 2)))
    return best


def save_val(name, prob, y, d, r): np.savez_compressed(os.path.join(OUT, f'vol_v2_val_{name}.npz'), prob=prob.astype(np.float32), y=y, day=d, row=r)


def _deep():
    import seq_models as D
    import torch; torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', 2))); return D, torch


def gru_data(days, mu=None, sd=None, tr_days=None):
    """allv matrix of `days`, standardised with TRAIN statistics (computed from tr_days unless mu/sd are given)"""
    D, torch = _deep(); T = D.T
    F, _, _, gseg, day, row, cov = D.load(days, list(ALLV))
    y = np.empty(len(F), np.int64); ok = np.empty(len(F), bool); k = 0
    for d in days:
        v = np.load(os.path.join(VD, f'{d}.npz')); m = len(v['ok']); y[k:k + m] = cls(v['rv']); ok[k:k + m] = v['ok']; k += m
    ok[:T - 1] = False; ok[T - 1:] &= gseg[T - 1:] == gseg[:-(T - 1)]
    if mu is None:
        tr = np.where(ok & np.isin(day, [int(d) for d in tr_days]))[0]; s_ = F[tr[::50]]; mu, sd = s_.mean(0), s_.std(0) + 1e-6
    np.subtract(F, mu, out=F); np.divide(F, sd, out=F); np.clip(F, -10, 10, out=F)
    return F, y, ok, day, row, cov, mu, sd


def gru_predict(net, F, idx):
    D, torch = _deep(); off = np.arange(-D.T + 1, 1); net.eval(); o = []
    with torch.no_grad():
        for j in range(0, len(idx), 4096): o.append(torch.softmax(net(torch.from_numpy(F[idx[j:j + 4096, None] + off])), 1).numpy())
    return np.concatenate(o)


def gru(seed, days_tr, days_ev, fixed=None):
    """conv+GRU with EMA on allv. fixed=None: early stopping on days_ev (= VAL), the checkpoint is saved and the VAL
    probabilities returned. fixed=(epochs, 'raw'|'ema'): that many epochs on days_tr, probabilities on days_ev."""
    D, torch = _deep(); nn = torch.nn; torch.manual_seed(seed); rng = np.random.default_rng(seed); off = np.arange(-D.T + 1, 1)
    F, y, ok, day, row, cov, mu, sd = gru_data(days_tr + days_ev, tr_days=days_tr)
    is_tr = np.isin(day, [int(d) for d in days_tr]); tr = np.where(ok & is_tr)[0]; ev = np.where(ok & ~is_tr & cov)[0]
    net, sh = D.ConvGRU(F.shape[1], 64, 0.2, 3), D.ConvGRU(F.shape[1], 64, 0.2, 3)
    lossf = nn.CrossEntropyLoss(weight=torch.tensor(len(tr) / (3 * np.bincount(y[tr], minlength=3)), dtype=torch.float32))
    opt = torch.optim.Adam(net.parameters(), 1e-3); ema = {a: b.detach().clone() for a, b in net.state_dict().items()}; step = 0
    best, bad, curve, state = -1, 0, [], None
    for ep in range(fixed[0] if fixed else D.EPOCHS):
        net.train(); sel = rng.choice(tr, min(D.PER, len(tr)), replace=False)
        for j in range(0, len(sel), D.BS):
            b = sel[j:j + D.BS]; opt.zero_grad(); lossf(net(torch.from_numpy(F[b[:, None] + off])), torch.from_numpy(y[b])).backward(); opt.step()
            with torch.no_grad():
                step += 1; dec = min(0.999, (1 + step) / (10 + step))
                for a, b2 in net.state_dict().items(): ema[a].mul_(dec).add_(b2, alpha=1 - dec)
        if fixed: continue
        sh.load_state_dict(ema); sub = ev[::10]
        fr, fe = mf1(y[sub], gru_predict(net, F, sub).argmax(1)), mf1(y[sub], gru_predict(sh, F, sub).argmax(1))
        curve.append(dict(epoch=ep, raw=round(fr, 4), ema=round(fe, 4))); print('vol gru', seed, curve[-1], flush=True)
        f, st, wh = max((fr, {a: b.clone() for a, b in net.state_dict().items()}, 'raw'), (fe, {a: b.clone() for a, b in ema.items()}, 'ema'), key=lambda c: c[0])
        if f > best: best, bad, state, best_ep, which = f, 0, st, ep, wh
        else:
            bad += 1
            if bad >= D.PAT: break
    if fixed: state = ema if fixed[1] == 'ema' else net.state_dict()
    else: torch.save(dict(state=state, mu=mu, sd=sd), os.path.join(OUT, f'vol_v2_gru_s{seed}.pt'))
    sh.load_state_dict(state); prob = gru_predict(sh, F, ev)
    return prob, y[ev], day[ev], row[ev], (None if fixed else dict(best_epoch=best_ep, best_weights=which, val10_mF1=round(best, 4), curve=curve))


def gru_apply(seed, days):
    """probabilities of the saved VAL-selected checkpoint on other days"""
    D, torch = _deep(); ck = torch.load(os.path.join(OUT, f'vol_v2_gru_s{seed}.pt'), weights_only=False)
    F, y, ok, day, row, cov, _, _ = gru_data(days, mu=ck['mu'], sd=ck['sd']); ev = np.where(ok & cov)[0]
    net = D.ConvGRU(F.shape[1], 64, 0.2, 3); net.load_state_dict(ck['state'])
    return gru_predict(net, F, ev), y[ev], day[ev], row[ev]


KEY = lambda day, row: day.astype(np.int64) * 1_000_000 + row


def boot(y, d, p_new, p_ref, n=2000, seed=1):
    days = np.unique(d); ix = {k: np.where(d == k)[0] for k in days}; rng = np.random.default_rng(seed); out = []
    for _ in range(n):
        m = np.concatenate([ix[k] for k in rng.choice(days, len(days))]); out.append(mf1(y[m], p_new[m]) - mf1(y[m], p_ref[m]))
    return dict(mean=round(float(np.mean(out)), 4), ci95=[round(float(np.quantile(out, .025)), 4), round(float(np.quantile(out, .975)), 4)])


if __name__ == '__main__':
    stages = sys.argv[1:]
    if set(stages) & set('ABRL'):
        Xtr, ytr, ltr, _, _ = stack(E.TR, sub=10); Xva, yva, lva, dva, rva = stack(E.VA, eval_only=True)
        print('loaded', Xtr.shape, Xva.shape, 'train shares', np.round(np.bincount(ytr) / len(ytr), 3), 'val shares', np.round(np.bincount(yva) / len(yva), 3), flush=True)

        def fit_cls(stage, name, cols):
            if done(stage, name): return
            t0 = time.time(); m = gbc().fit(Xtr[:, cols], ytr); cur, it, pr = E.curve_of(m, Xva[:, cols], yva)
            mm, fm = best_m(pr, yva)
            log(dict(stage=stage, name=name, n_features=len(cols), mF1=round(mf1(yva, pr.argmax(1)), 4), best_iter=it, curve=cur,
                     best_m=mm, mF1_at_best_m=fm, **stats(yva, pr.argmax(1), dva), sec=round(time.time() - t0)))
            if name == 'gbm_allv': save_val('gbm', pr, yva, dva, rva)
    for STAGE in stages:
        if STAGE == 'A':
            fit_cls('A', 'gbm_v0', V0); fit_cls('A', 'gbm_allv', ALLV)
        elif STAGE == 'B':
            fit_cls('B', 'allv_minus_daylvl', [c for c in ALLV if c not in E.G['daylvl']])
            fit_cls('B', 'allv_minus_rvc_hist', [c for c in ALLV if c not in E.G['rvc'] + E.G['hist']])
        elif STAGE == 'R' and not done('R', 'gbr_allv'):
            t0 = time.time(); m = gbr().fit(Xtr[:, ALLV], ltr); best = (-1,)
            for it, pr in enumerate(m.staged_predict(Xva[:, ALLV]), 1):
                if it % 100 == 0:
                    f, a, b = best_thr(pr, yva)
                    if f > best[0]: best = (f, a, b, it, pr.copy())
            p = (best[4] > best[1]).astype(int) + (best[4] > best[2])
            log(dict(stage='R', name='gbr_allv', mF1=round(best[0], 4), thr=[best[1], best[2]], best_iter=best[3], **stats(yva, p, dva),
                     fixed_cut_mF1=round(mf1(yva, cls(np.expm1(best[4]))), 4), sec=round(time.time() - t0)))
            np.savez_compressed(os.path.join(OUT, 'vol_v2_val_gbr.npz'), pred=best[4].astype(np.float32), y=yva, day=dva, row=rva)
        elif STAGE == 'L' and not done('L', 'logreg_allv'):
            t0 = time.time(); mu, sd = Xtr[:, ALLV].mean(0), Xtr[:, ALLV].std(0) + 1e-6
            m = LogisticRegression(max_iter=500, class_weight='balanced').fit((Xtr[:, ALLV] - mu) / sd, ytr); pr = m.predict_proba((Xva[:, ALLV] - mu) / sd)
            mm, fm = best_m(pr, yva)
            log(dict(stage='L', name='logreg_allv', mF1=round(mf1(yva, pr.argmax(1)), 4), best_m=mm, mF1_at_best_m=fm, **stats(yva, pr.argmax(1), dva), sec=round(time.time() - t0)))
            save_val('logreg', pr, yva, dva, rva)
        elif STAGE == 'G':
            for seed in (0, 1, 2):
                if done('G', f'gru_s{seed}'): continue
                t0 = time.time(); pr, y, d, r, info = gru(seed, E.TR, E.VA)
                log(dict(stage='G', name=f'gru_s{seed}', mF1=round(mf1(y, pr.argmax(1)), 4), **info, **stats(y, pr.argmax(1), d), sec=round(time.time() - t0)))
                save_val(f'gru_s{seed}', pr, y, d, r)
        elif STAGE == 'E':
            from itertools import combinations
            Z = {n: np.load(os.path.join(OUT, f'vol_v2_val_{n}.npz')) for n in ('gbm', 'logreg', 'gru_s0', 'gru_s1', 'gru_s2', 'gbr')}
            k0 = KEY(Z['gru_s0']['day'], Z['gru_s0']['row']); y = Z['gru_s0']['y']; d = Z['gru_s0']['day']

            def on(z):
                kz = KEY(z['day'], z['row']); ix = np.searchsorted(kz, k0); assert (kz[ix] == k0).all() and (z['y'][ix] == y).all(); return ix

            P = {n: Z[n]['prob'][on(Z[n])] for n in ('gbm', 'logreg')}; P['gru'] = np.mean([Z[f'gru_s{s}']['prob'] for s in (0, 1, 2)], 0)
            reg = Z['gbr']['pred'][on(Z['gbr'])]; thr = recs('R')[-1]['thr']
            for s in (0, 1, 2): log(dict(stage='E', name=f'gru_s{s}', members=[f'gru_s{s}'], m=1.0, mF1=round(mf1(y, Z[f'gru_s{s}']['prob'].argmax(1)), 4)))
            cands = {}
            for k in (1, 2, 3):
                for sub in combinations(('gbm', 'logreg', 'gru'), k):
                    pr = np.mean([P[n] for n in sub], 0); m, fm = best_m(pr, y); cands['+'.join(sub)] = (fm, m, pr)
                    log(dict(stage='E', name='+'.join(sub), members=list(sub), m=m, mF1=fm, mF1_m1=round(mf1(y, pr.argmax(1)), 4)))
            f_reg = round(mf1(y, (reg > thr[0]).astype(int) + (reg > thr[1])), 4); log(dict(stage='E', name='gbr', members=['gbr'], thr=thr, mF1=f_reg))
            best = max(cands, key=lambda n: cands[n][0])
            win = dict(stage='E_winner', name='gbr', members=['gbr'], thr=thr, val_mF1=f_reg) if f_reg > cands[best][0] else \
                dict(stage='E_winner', name=best, members=best.split('+'), m=cands[best][1], val_mF1=cands[best][0])
            log(win)
            # split-half check: choose on one half of the VAL days, score on the other half
            days = np.unique(d); halves = (days[:len(days) // 2], days[len(days) // 2:])
            for a, b in ((0, 1), (1, 0)):
                sel, ev = np.isin(d, halves[a]), np.isin(d, halves[b])
                pick = max(((mf1(y[sel], (pr[sel] * [1, m, 1]).argmax(1)), n, m) for n, (_, _, pr) in cands.items() for m in MS), key=lambda t: t[0])
                fr, ta, tb = best_thr(reg[sel], y[sel])
                if fr > pick[0]: name, p_ev = 'gbr', (reg[ev] > ta).astype(int) + (reg[ev] > tb)
                else: name, p_ev = pick[1], (cands[pick[1]][2][ev] * [1, pick[2], 1]).argmax(1)
                log(dict(stage='E_half', chosen_on=[int(x) for x in halves[a]], chosen=name, m=pick[2], mF1_on_other_half=round(mf1(y[ev], p_ev), 4),
                         gbm_m1_on_other_half=round(mf1(y[ev], P['gbm'][ev].argmax(1)), 4)))
        elif STAGE == 'test':
            w = recs('E_winner')[-1]; a = {r['name']: r for r in recs('A')}; g = {r['name']: r for r in recs('G')}; R = recs('R')[-1]
            Xte, yte, lte, dte, rte = stack(E.TE, eval_only=True); kte = KEY(dte, rte); ref = None
            for variant, days in (('train', E.TR), ('train+val', E.TR + E.VA)):
                if done('test', variant): continue
                t0 = time.time(); Xt, yt, lt, _, _ = stack(days, sub=10); P = {}; keep = np.ones(len(yte), bool)
                if ref is None and variant == 'train': ref = gbc(a['gbm_v0']['best_iter']).fit(Xt[:, V0], yt).predict(Xte[:, V0])
                if 'gru' in w['members']:
                    Gs = [gru_apply(s, E.TE) if variant == 'train' else gru(s, days, E.TE, fixed=(g[f'gru_s{s}']['best_epoch'] + 1, g[f'gru_s{s}']['best_weights']))[:4] for s in (0, 1, 2)]
                    kg = KEY(Gs[0][2], Gs[0][3]); keep = np.isin(kte, kg); assert (kte[keep] == kg).all() and (yte[keep] == Gs[0][1]).all()
                    P['gru'] = np.mean([q[0] for q in Gs], 0)
                if 'gbm' in w['members']: P['gbm'] = gbc(a['gbm_allv']['best_iter']).fit(Xt[:, ALLV], yt).predict_proba(Xte[keep][:, ALLV])
                if 'logreg' in w['members']:
                    mu, sd = Xt[:, ALLV].mean(0), Xt[:, ALLV].std(0) + 1e-6
                    P['logreg'] = LogisticRegression(max_iter=500, class_weight='balanced').fit((Xt[:, ALLV] - mu) / sd, yt).predict_proba((Xte[keep][:, ALLV] - mu) / sd)
                if w['members'] == ['gbr']:
                    pr = gbr(R['best_iter']).fit(Xt[:, ALLV], lt).predict(Xte[:, ALLV]); p = (pr > w['thr'][0]).astype(int) + (pr > w['thr'][1]); prob = None
                else:
                    prob = np.mean([P[n] for n in w['members']], 0); p = (prob * [1, w['m'], 1]).argmax(1)
                y, d = yte[keep], dte[keep]
                rec = dict(stage='test', name=variant, pipeline=w['name'], n=int(len(y)), mF1=round(mf1(y, p), 4), **stats(y, p, d),
                           confusion=confusion_matrix(y, p).tolist(), day_mF1={str(k): round(mf1(y[d == k], p[d == k]), 4) for k in np.unique(d)},
                           sec=round(time.time() - t0))
                if ref is not None:
                    rec.update(ref_v0_gbm_mF1=round(mf1(y, ref[keep]), 4), boot_new_minus_ref=boot(y, d, p, ref[keep]))
                log(rec); np.savez_compressed(os.path.join(OUT, f'vol_v2_test_{variant}.npz'), pred=p, y=y, day=d, row=rte[keep], **({'prob': prob.astype(np.float32)} if prob is not None else {}))
