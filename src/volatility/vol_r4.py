"""Volatility (60 s) round 4: a conv+GRU that predicts the size of the next 60 s volatility (regression), alone and
blended with the round-2 GBM regression. Aim: beat round 2 (GBM regression on log(1 + RV60) + thresholds [2.0, 2.85],
VAL 0.6412, TEST 0.6612). Design fixed before running; chosen on VAL only; TEST is read once, in the 'test' stage.
Why  : for the GBM, regression + thresholds beat classification on VAL (0.6417 vs 0.6289), but the conv+GRU has only been
       trained as a classifier (vol_v2.py stage G, VAL bag 0.6294).
Member: the conv+GRU of vol_v2.py (allv 145 columns, 50 snapshots, hidden 64, dropout 0.2, EMA, lr 1e-3, up to 12 epochs
       x 200,000 windows, batch 256, stop after 3 epochs without improvement) with one output: log(1 + RV60),
       standardised with TRAIN statistics, mean squared error. After each epoch: VAL macro-F1 on every 10th VAL window
       with the best two thresholds on that subset; best epoch kept (raw or EMA weights). Seeds 0, 1, 2; the bag is the
       mean of their predictions (as the 3-seed conv+GRU classifier of round 2).
Thresholds: low 1.00-3.50 and high 1.50-4.50 (step 0.05, high > low), in log(1 + ticks).
Candidates on VAL (the rows every member scores; macro-F1)
  gbr_r2      : round 2 exactly as chosen (GBM regression, thresholds [2.0, 2.85])
  convgru_reg : the bag, thresholds chosen on VAL
  blend_w     : w x GBM + (1 - w) x conv+GRU (both log(1 + RV60)), w = 0.1 ... 0.9, thresholds chosen on VAL
  reference   : round-3 stacking (vol_r3.py, VAL 0.6422) on the same rows - not a candidate (its TEST was already read)
Winner: highest VAL macro-F1. Split-half check: the whole choice on VAL days 1-6, scored on days 7-12, and the reverse,
        next to the round-2 recipe with its thresholds chosen on the same half. If gbr_r2 wins there is no TEST run.
TEST  : once. conv+GRU from the saved checkpoints; GBM refitted on TRAIN with its settings (deterministic; checked against
        round 2's TEST predictions). References on the same rows: round 2 (results/volatility/vol_v2_test_train.npz) and round 3
        (results/volatility/vol_r3_test.npz); day-block bootstrap (2,000 resamples) of the macro-F1 differences.
Note: the TEST days were already used by vol_cls.py, vol_v2.py and vol_r3.py, so they are not an untouched hold-out.
usage: python vol_r4.py train <seed>[,<seed>...] [threads] | val | test [threads]   -> results/volatility/vol_r4.jsonl
  env SMOKE=1 (deep_v2.py's switch): 1 epoch of 4,096 windows on 2 TRAIN days and 1 VAL day - a code check only
"""
import _paths  # noqa: F401  (src/common, src/direction, src/volatility, src/report on the import path)
import json, os, sys, time
import numpy as np, torch, torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix
import common as C
import exp_v2 as E

argv = sys.argv; sys.argv = argv[:1]
import deep_v2 as D
import vol_v2 as V
sys.argv = argv
OUT = C.RES_VOL; os.makedirs(OUT, exist_ok=True); RES = os.path.join(OUT, 'vol_r4.jsonl'); VD = os.path.join(C.DATA, 'feat_vol'); T = D.T
cls = lambda x: np.digitize(x, V.CUTS, right=True)
SMOKE = D.SMOKE; TR, VA = (E.TR[:2], E.VA[:1]) if SMOKE else (E.TR, E.VA)
LO = np.round(np.arange(1.00, 3.501, 0.05), 3); HI = np.round(np.arange(1.50, 4.501, 0.05), 3); WS = np.round(np.arange(0.1, 0.91, 0.1), 2)
R2_THR = [2.0, 2.85]
KEY = lambda day, row: np.asarray(day).astype(np.int64) * 1_000_000 + np.asarray(row)


def mf1(y, p):
    cm = np.bincount(np.asarray(y) * 3 + np.asarray(p), minlength=9).reshape(3, 3); tp = np.diag(cm).astype(float)
    return float(np.mean(2 * tp / np.maximum(cm.sum(0) + cm.sum(1), 1)))


def by_thr(x, t): return (x > t[0]).astype(int) + (x > t[1])


def tune(x, y, rows=None):
    """best macro-F1 over the threshold grid -> (score, [low, high]); exact, from per-class sorted predictions"""
    if rows is not None: x, y = x[rows], y[rows]
    xs = [np.sort(x[y == c]) for c in range(3)]; n = np.array([len(a) for a in xs])
    le = lambda t: np.array([np.searchsorted(a, t, side='right') for a in xs])       # per true class: count with x <= t
    LEL = {t: le(t) for t in LO}; LEH = {t: le(t) for t in HI}; best = (-1.0, None)
    for lo in LO:
        for hi in HI:
            if hi <= lo: continue
            a, b = LEL[lo], LEH[hi]; cm = np.stack([a, b - a, n - b], 1); tp = np.diag(cm)
            f = float(np.mean(2 * tp / np.maximum(cm.sum(0) + cm.sum(1), 1)))
            if f > best[0]: best = (f, [float(lo), float(hi)])
    return best


def log(rec):
    print(json.dumps({k: v for k, v in rec.items() if k != 'curve'})[:600], flush=True)
    with open(RES, 'a') as f: f.write(json.dumps(rec) + '\n')


def recs(stage=None):
    R = [json.loads(l) for l in open(RES)] if os.path.exists(RES) else []
    return [r for r in R if stage is None or r.get('stage') == stage]


def load(days):
    """allv matrix (raw values), class, log(1 + RV60) and valid 50-snapshot windows of `days`"""
    F, _, _, gseg, day, row, cov = D.load(days, list(V.ALLV))
    y = np.empty(len(F), np.int64); z = np.empty(len(F), np.float32); ok = np.empty(len(F), bool); k = 0
    for d in days:
        v = np.load(os.path.join(VD, f'{d}.npz')); m = len(v['ok'])
        y[k:k + m] = cls(v['rv']); z[k:k + m] = np.log1p(np.nan_to_num(v['rv'])); ok[k:k + m] = v['ok']; k += m
    assert k == len(F)
    ok[:T - 1] = False; ok[T - 1:] &= gseg[T - 1:] == gseg[:-(T - 1)]
    return F, y, z, ok, day, row, cov


def outputs(net, F, idx, zm, zs):
    off = np.arange(-T + 1, 1); net.eval(); o = []
    with torch.no_grad():
        for j in range(0, len(idx), 4096): o.append((net(torch.from_numpy(F[idx[j:j + 4096, None] + off]))[:, 0] * zs + zm).numpy())
    return np.concatenate(o)


def train(seeds):
    F, y, z, ok, day, row, cov = load(TR + VA)
    is_tr = np.isin(day, [int(d) for d in TR]); tr = np.where(ok & is_tr)[0]; va_all = np.where(ok & ~is_tr & cov)[0]; va = va_all[::10]
    s_ = F[tr[::50]]; mu, sd = s_.mean(0), s_.std(0) + 1e-6
    np.subtract(F, mu, out=F); np.divide(F, sd, out=F); np.clip(F, -10, 10, out=F)
    zm, zs = float(z[tr].mean()), float(z[tr].std()); off = np.arange(-T + 1, 1); have = {r['seed'] for r in recs('train')}
    for seed in seeds:
        if seed in have: continue
        torch.manual_seed(seed); rng = np.random.default_rng(seed); t0 = time.time()
        net, sh = D.ConvGRU(F.shape[1], 64, 0.2, 1), D.ConvGRU(F.shape[1], 64, 0.2, 1); mse = nn.MSELoss()
        opt = torch.optim.Adam(net.parameters(), 1e-3); ema = {a: b.detach().clone() for a, b in net.state_dict().items()}; step = 0
        best, bad, curve, state = -1, 0, [], None
        for ep in range(D.EPOCHS):
            net.train(); sel = rng.choice(tr, min(D.PER, len(tr)), replace=False)
            for j in range(0, len(sel), D.BS):
                b = sel[j:j + D.BS]; opt.zero_grad()
                mse(net(torch.from_numpy(F[b[:, None] + off]))[:, 0], torch.from_numpy((z[b] - zm) / zs)).backward(); opt.step()
                with torch.no_grad():
                    step += 1; dec = min(0.999, (1 + step) / (10 + step))
                    for a, b2 in net.state_dict().items(): ema[a].mul_(dec).add_(b2, alpha=1 - dec)
            sh.load_state_dict(ema)
            fr, fe = tune(outputs(net, F, va, zm, zs), y[va])[0], tune(outputs(sh, F, va, zm, zs), y[va])[0]
            curve.append(dict(epoch=ep, raw=round(fr, 4), ema=round(fe, 4), min=round((time.time() - t0) / 60, 1))); print('vol reg', seed, curve[-1], flush=True)
            f, st, wh = max((fr, {a: b.clone() for a, b in net.state_dict().items()}, 'raw'), (fe, {a: b.clone() for a, b in ema.items()}, 'ema'), key=lambda c: c[0])
            if f > best: best, bad, state, best_ep, which = f, 0, st, ep, wh
            else:
                bad += 1
                if bad >= D.PAT: break
        sh.load_state_dict(state); pv = outputs(sh, F, va_all, zm, zs); yy = y[va_all]; ft, thr = tune(pv, yy)
        np.savez_compressed(os.path.join(OUT, f'vol_r4_val_convgru_reg_s{seed}.npz'), pred=pv.astype(np.float32), y=yy, day=day[va_all], row=row[va_all])
        torch.save(dict(state=state, mu=mu, sd=sd, zm=zm, zs=zs), os.path.join(OUT, f'vol_r4_convgru_reg_s{seed}.pt'))
        rmse = float(np.sqrt(np.mean((pv - z[va_all]) ** 2)))
        log(dict(stage='train', name='convgru_reg', seed=seed, best_epoch=best_ep, best_weights=which, val10_mF1=round(best, 4), val_mF1=round(ft, 4),
                 thr=thr, val_rmse_log=round(rmse, 4), val_acc=round(float(accuracy_score(yy, by_thr(pv, thr))), 4), smoke=SMOKE,
                 sec=round(time.time() - t0), curve=curve))


def val_pool():
    Z = [np.load(os.path.join(OUT, f'vol_r4_val_convgru_reg_s{s}.npz')) for s in (0, 1, 2)]
    keys = KEY(Z[0]['day'], Z[0]['row'])
    for z in Z[1:]: assert (KEY(z['day'], z['row']) == keys).all()
    g = np.load(os.path.join(OUT, 'vol_v2_val_gbr.npz')); kg = KEY(g['day'], g['row'])
    s3 = np.load(os.path.join(OUT, 'vol_r3_val_stack_gbm.npz')); k3 = KEY(s3['day'], s3['row'])
    keys = np.intersect1d(np.intersect1d(keys, kg), k3)
    ix = lambda k: np.searchsorted(k, keys)
    iz, ig, i3 = ix(KEY(Z[0]['day'], Z[0]['row'])), ix(kg), ix(k3)
    assert (kg[ig] == keys).all() and (k3[i3] == keys).all()
    y = Z[0]['y'][iz]; assert (g['y'][ig] == y).all() and (s3['y'][i3] == y).all()
    cg = np.mean([z['pred'][iz] for z in Z], 0); gb = g['pred'][ig].astype(np.float64)
    w3 = [r for r in map(json.loads, open(os.path.join(OUT, 'vol_r3.jsonl'))) if r.get('stage') == 'winner'][-1]
    p3 = (s3['prob'][i3] * w3['mult']).argmax(1)
    return y, keys // 1_000_000, cg, gb, p3, w3


def candidates(cg, gb, y, rows=None):
    out = [(mf1(y if rows is None else y[rows], by_thr(gb if rows is None else gb[rows], R2_THR)), 'gbr_r2', dict(thr=R2_THR, w=1.0))]
    f, t = tune(cg, y, rows); out.append((f, 'convgru_reg', dict(thr=t, w=0.0)))
    for w in WS:
        f, t = tune(w * gb + (1 - w) * cg, y, rows); out.append((f, f'blend_{w}', dict(thr=t, w=float(w))))
    return sorted(out, key=lambda c: -c[0])


def val():
    t0 = time.time(); y, d, cg, gb, p3, w3 = val_pool()
    log(dict(stage='info', n_rows=int(len(y)), n_days=int(len(np.unique(d))), round3_stack_mF1=round(mf1(y, p3), 4), round3_name=w3['name']))
    C_ = candidates(cg, gb, y)
    for f, nm, sp in C_: log(dict(stage='cand', name=nm, mF1=round(f, 4), **sp))
    f, nm, sp = C_[0]; p = by_thr(sp['w'] * gb + (1 - sp['w']) * cg, sp['thr'])
    log(dict(stage='winner', name=nm, mF1=round(f, 4), **sp, acc=round(float(accuracy_score(y, p)), 4),
             recall=[round(float(x), 4) for x in confusion_matrix(y, p, normalize='true').diagonal()],
             gbr_r2_mF1=round(mf1(y, by_thr(gb, R2_THR)), 4), round3_stack_mF1=round(mf1(y, p3), 4)))
    days = np.unique(d); halves = (days[:len(days) // 2], days[len(days) // 2:])
    for a, b in ((0, 1), (1, 0)):
        sel, ev = np.where(np.isin(d, halves[a]))[0], np.isin(d, halves[b])
        f, nm, sp = candidates(cg, gb, y, sel)[0]; p = by_thr(sp['w'] * gb + (1 - sp['w']) * cg, sp['thr']); _, t2 = tune(gb, y, sel)
        log(dict(stage='half', chosen_on=[int(x) for x in halves[a]], chosen=nm, spec=sp, mF1_other_half=round(mf1(y[ev], p[ev]), 4),
                 r2_recipe_other_half=round(mf1(y[ev], by_thr(gb[ev], t2)), 4), r2_thr_on_half=t2, round3_other_half=round(mf1(y[ev], p3[ev]), 4)))
    log(dict(stage='val_done', sec=round(time.time() - t0)))


def boot(y, d, p_new, p_ref, n=2000, seed=1):
    days = np.unique(d); ix = {k: np.where(d == k)[0] for k in days}; rng = np.random.default_rng(seed); out = []
    for _ in range(n):
        m = np.concatenate([ix[k] for k in rng.choice(days, len(days))]); out.append(mf1(y[m], p_new[m]) - mf1(y[m], p_ref[m]))
    return dict(mean=round(float(np.mean(out)), 4), ci95=[round(float(np.quantile(out, .025)), 4), round(float(np.quantile(out, .975)), 4)],
                share_positive=round(float(np.mean(np.array(out) > 0)), 3))


def test():
    w = recs('winner')[-1]
    if w['name'] == 'gbr_r2': log(dict(stage='test', skipped=True, why='round 2 won on VAL')); return
    t0 = time.time(); F, y, z, ok, day, row, cov = load(E.TE); ev = np.where(ok & cov)[0]; preds = []
    for s in (0, 1, 2):
        ck = torch.load(os.path.join(OUT, f'vol_r4_convgru_reg_s{s}.pt'), weights_only=False)
        G_ = ((F - ck['mu']) / ck['sd']).astype(np.float32); np.clip(G_, -10, 10, out=G_)
        net = D.ConvGRU(F.shape[1], 64, 0.2, 1); net.load_state_dict(ck['state']); preds.append(outputs(net, G_, ev, ck['zm'], ck['zs'])); del G_
    del F; print('conv+GRU done', round(time.time() - t0), flush=True)
    kc = KEY(day[ev], row[ev]); cg = np.mean(preds, 0)
    Xte, yte, _, dte, rte = V.stack(E.TE, eval_only=True); kt = KEY(dte, rte); Xt, yt, lt, _, _ = V.stack(E.TR, sub=10); R = V.recs('R')[-1]
    gb_all = V.gbr(R['best_iter']).fit(Xt[:, V.ALLV], lt).predict(Xte[:, V.ALLV]); del Xt, Xte
    r2 = np.load(os.path.join(OUT, 'vol_v2_test_train.npz')); k2 = KEY(r2['day'], r2['row'])
    r3 = np.load(os.path.join(OUT, 'vol_r3_test.npz')); k3 = KEY(r3['day'], r3['row'])
    keys = np.intersect1d(np.intersect1d(np.intersect1d(kc, kt), k2), k3)
    ic, it, i2, i3 = (np.searchsorted(k, keys) for k in (kc, kt, k2, k3))
    yy = y[ev][ic]; dd = keys // 1_000_000
    assert (yte[it] == yy).all() and (r2['y'][i2] == yy).all() and (r3['y'][i3] == yy).all()
    gb = gb_all[it]; same = float(np.mean(by_thr(gb, R2_THR) == r2['pred'][i2]))
    x = w['w'] * gb + (1 - w['w']) * cg[ic]; p = by_thr(x, w['thr']); p2, p3 = r2['pred'][i2], r3['pred'][i3]
    log(dict(stage='test', pipeline=w['name'], w=w['w'], thr=w['thr'], val_mF1=w['mF1'], n=int(len(yy)),
             class_shares=[round(float(v), 4) for v in np.bincount(yy, minlength=3) / len(yy)],
             mF1=round(mf1(yy, p), 4), acc=round(float(accuracy_score(yy, p)), 4),
             recall=[round(float(v), 4) for v in confusion_matrix(yy, p, normalize='true').diagonal()],
             convgru_reg_alone_mF1=round(mf1(yy, by_thr(cg[ic], [c for c in recs('cand') if c['name'] == 'convgru_reg'][-1]['thr'])), 4),
             refit_gbr_agrees_with_r2=round(same, 6), r2_mF1=round(mf1(yy, p2), 4), r3_mF1=round(mf1(yy, p3), 4),
             boot_new_minus_r2=boot(yy, dd, p, p2), boot_new_minus_r3=boot(yy, dd, p, p3),
             days_better_than_r2=int(sum(mf1(yy[dd == k], p[dd == k]) > mf1(yy[dd == k], p2[dd == k]) for k in np.unique(dd))),
             day_mF1={str(k): round(mf1(yy[dd == k], p[dd == k]), 4) for k in np.unique(dd)},
             confusion=confusion_matrix(yy, p).tolist(), sec=round(time.time() - t0)))
    np.savez_compressed(os.path.join(OUT, 'vol_r4_test.npz'), x=x.astype(np.float32), pred=p, y=yy, day=dd, row=keys % 1_000_000)


if __name__ == '__main__':
    what = sys.argv[1]
    torch.set_num_threads(int(sys.argv[3]) if what == 'train' and len(sys.argv) > 3 else int(sys.argv[2]) if what == 'test' and len(sys.argv) > 2 else 2)
    if what == 'train': train([int(s) for s in sys.argv[2].split(',')])
    elif what == 'val' and not recs('winner'): val()
    elif what == 'test' and not recs('test'): test()
