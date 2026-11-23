"""Calm -> turbulent check for the 60 s volatility task: does a forecast catch the next 60 s turning turbulent when the
last 60 s were calm? Diagnostic only: nothing is refit or re-tuned; every forecast and threshold is the saved one chosen
on VAL in its round. Measures fixed before running.
State  : s(t) = class of the past-60 s RV with the label cuts of vol_cls.py (the persistence rule), only where the whole
         past 60 s lie inside the segment (rows in the first minute of a segment are left out).
Target : y(t) = class of RV60 over (t, t + 60 s] (the label). Cell calm -> turbulent: s = calm and y = turbulent.
Rows   : the rows every forecast of the split scores (TEST 10 days, VAL 12 days).
Forecasts
  persistence : s itself (never calls a change)
  r1_gru      : round-1 conv+GRU classifier, most probable class (TEST only)
  r2          : round-2 GBM regression + thresholds [2.0, 2.85]
  r3          : round-3 stacking (TEST only)
  r4          : round-4 blend (TEST only)
  r5          : round-5 blend 0.7 x GBM + 0.3 x conv+GRU, thresholds [2.0, 2.85]
  r7          : round-7 blend 0.9 x GBM with market features + 0.1 x conv+GRU, thresholds [2.05, 2.85] (final model)
Measures
  cells     : rows and share of each (s, y) cell; per cell the share of rows forecast calm / normal / turbulent
  recall    : P(forecast turbulent | y turbulent, s) for s = calm, normal, turbulent
  from calm : P(forecast turbulent | s calm), its precision P(y turbulent | forecast turbulent, s calm), the base rate
              P(y turbulent | s calm) and lift = precision / base rate
  visible   : the calm -> turbulent rows split by the past 10 s: already moving at the turbulent pace (past-10 s RV above
              the upper cut scaled to 10 s, 19.5 / 6 ticks) or not; recall in each part
  episodes  : runs of consecutive calm -> turbulent rows; number, median length, share with at least one turbulent call
  AUC       : within s = calm, turbulent vs the rest, for the continuous scores (blend log(1 + RV60), turbulent
              probability) and for the past 10 / 60 / 300 s RV as references
  CIs       : day-block bootstrap, 2,000 resamples (the same resamples for every forecast), 95%
Shares, AUCs and intervals are stored with 6 decimals so that percentages can be rounded once for the tables.
Added after the first run: AUC of each forecast minus that of the past-300 s RV (the strongest reference there), with the
same resamples (stage auc_vs_past300). Then, VAL only (TEST kept for a rule that may be tested later), stage topk: within
s = calm, ranked by the final forecast (r7), the share that turns turbulent among the top 0.1 / 0.5 / 1 / 2 / 5 / 10 / 20 %
of rows and the share of all calm -> turbulent rows they hold; and the median forecast by the class of the next 60 s.
usage: python transition.py   -> results/volatility/transition.jsonl (stages already in the file are skipped)
"""
import _paths  # noqa: F401  (src/common, src/direction, src/volatility, src/report on the import path)
import json, os, sys
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score
import common as C

OUT = C.RES_VOL; RES = os.path.join(OUT, 'transition.jsonl'); VD = os.path.join(C.DATA, 'feat_vol')
CUTS = [json.loads(l) for l in open(os.path.join(OUT, 'vol_cls.jsonl')) if '"labels"' in l][-1]['cuts']
FAST10 = CUTS[1] / 6                                   # turbulent pace over 10 s, ticks of mid path length
NB = 2000
cls = lambda x: np.digitize(x, CUTS, right=True)       # as vol_cls.py: <= c1 calm, <= c2 normal, else turbulent
by_thr = lambda x, t: (x > t[0]).astype(int) + (x > t[1])
KEY = lambda day, row: np.asarray(day).astype(np.int64) * 1_000_000 + np.asarray(row).astype(np.int64)


def log(rec):
    print(json.dumps(rec)[:700], flush=True)
    with open(RES, 'a') as f: f.write(json.dumps(rec) + '\n')


def npz(f): return np.load(os.path.join(OUT, f))


def day_state(d):
    z = C.load_day(d); ts = z['ts'].astype(np.int64); seg = z['seg']; v = np.load(os.path.join(VD, f'{d}.npz'))
    start = ts[np.searchsorted(seg, seg, side='left')]
    return dict(s=cls(v['past60']), y=np.where(v['ok'], cls(v['rv']), -1), full=ts - 60 * 10**9 >= start, ts=ts, seg=seg,
                past=v['past'].astype(np.float64), ok=v['ok'], minute=z['minute'], cov=float(z['cov']))


def test_forecasts(TE):
    F, info = {}, {}
    for name, f, thr in (('r2', 'vol_v2_test_train.npz', None), ('r3', 'vol_r3_test.npz', None), ('r4', 'vol_r4_test.npz', None),
                         ('r5', 'r5_vol_test.npz', [2.0, 2.85]), ('r7', 'r7_vol_test.npz', [2.05, 2.85])):
        z = npz(f); F[name] = dict(key=KEY(z['day'], z['row']), pred=z['pred'].astype(int), y=z['y'].astype(int))
        if 'x' in z.files: F[name]['score'] = z['x'].astype(np.float64)
        if 'prob' in z.files: F[name]['score'] = z['prob'][:, 2].astype(np.float64)
        if thr: info[f'{name}_pred_equals_thresholded_x'] = bool((by_thr(z['x'], thr) == z['pred']).all())
    # round 1 saved no row numbers: rebuild them as vol_cls.py chose its TEST windows (valid label, 50 snapshots in one segment)
    z = npz('test_pred_vol_gru.npz'); rows, days, mins = [], [], []
    for d in TE:
        st = day_state(d)
        if st['cov'] < C.MIN_COV: continue
        n = len(st['s']); ok = st['ok'] & (np.arange(n) >= 49); ok[49:] &= st['seg'][49:] == st['seg'][:n - 49]
        i = np.where(ok)[0]; rows.append(i); days.append(np.full(len(i), int(d))); mins.append(st['minute'][i])
    rows, days, mins = np.concatenate(rows), np.concatenate(days), np.concatenate(mins)
    same = len(rows) == len(z['y']) and bool((days == z['day']).all()) and bool(np.allclose(mins, z['minute']))
    info['r1_gru_rows_rebuilt'] = same
    if same: F['r1_gru'] = dict(key=KEY(days, rows), pred=z['prob'].argmax(1), y=z['y'].astype(int), score=z['prob'][:, 2].astype(np.float64))
    return F, info


def val_forecasts():
    """the VAL forecasts of rounds 2, 5 and 7 with their continuous scores (classes as exec_sim2.val_forecasts())"""
    def load(f): z = npz(f); return KEY(z['day'], z['row']), z['pred'].astype(np.float64), z['y'].astype(int)
    kg, gb, yg = load('vol_v2_val_gbr.npz'); km, mk, ym = load('r7_val_vol_mkt.npz')
    cs = [load(f'r5_val_vreg_mh_s{s}_vol.npz') for s in range(5)]; kc = cs[0][0]
    for k, _, _ in cs[1:]: assert (k == kc).all()
    for k in (kg, km, kc): assert (np.diff(k) > 0).all()
    cg = np.mean([p for _, p, _ in cs], 0)
    def on(keys, k, p): ix = np.searchsorted(k, keys); assert (k[ix] == keys).all(); return p[ix]
    k5 = np.intersect1d(kg, kc); x5 = 0.7 * on(k5, kg, gb) + 0.3 * on(k5, kc, cg)
    k7 = np.intersect1d(km, kc); x7 = 0.9 * on(k7, km, mk) + 0.1 * on(k7, kc, cg)
    F = dict(r2=dict(key=kg, score=gb, pred=by_thr(gb, [2.0, 2.85]), y=yg),
             r5=dict(key=k5, score=x5, pred=by_thr(x5, [2.0, 2.85]), y=on(k5, kg, yg)),
             r7=dict(key=k7, score=x7, pred=by_thr(x7, [2.05, 2.85]), y=on(k7, km, ym)))
    argv = sys.argv; sys.argv = argv[:1]; import exec_sim2 as X2; sys.argv = argv
    ref = X2.val_forecasts(); info = {}
    for n in F:
        k = KEY(ref[n]['day'], ref[n]['row'])
        info[f'{n}_same_as_exec_sim2'] = bool(len(k) == len(F[n]['key']) and (k == F[n]['key']).all() and (ref[n]['pred'] == F[n]['pred']).all())
    return F, info


def assemble(F):
    """rows every forecast scores, with state, label, past RV and each forecast; rows without a full past minute dropped"""
    keys = None
    for f in F.values(): keys = f['key'] if keys is None else np.intersect1d(keys, f['key'])
    day, row = keys // 1_000_000, keys % 1_000_000; A = {k: [] for k in ('day', 's', 'y', 'full', 'ts', 'seg', 'row', 'past')}
    for d in np.unique(day):
        st = day_state(str(d)); r = row[day == d]
        for k in ('s', 'y', 'full', 'ts', 'seg', 'past'): A[k].append(st[k][r])
        A['day'].append(np.full(len(r), d)); A['row'].append(r)
    A = {k: np.concatenate(v) for k, v in A.items()}; A['key'] = keys
    for n, f in F.items():
        ix = np.searchsorted(f['key'], keys) if (np.diff(f['key']) > 0).all() else None
        if ix is None: o = np.argsort(f['key']); ix = o[np.searchsorted(f['key'][o], keys)]
        assert (f['key'][ix] == keys).all()
        A['pred_' + n] = f['pred'][ix]; A['ystored_' + n] = f['y'][ix]
        if 'score' in f: A['score_' + n] = f['score'][ix]
    keep = A['full'] & (A['y'] >= 0)
    info = dict(rows=int(len(keys)), rows_without_full_past_minute=int((~A['full']).sum()), rows_kept=int(keep.sum()),
                labels_match={n: bool((A['ystored_' + n] == A['y']).all()) for n in F})
    A = {k: v[keep] for k, v in A.items()}; A['pred_persistence'] = A['s']
    return A, info


def counts(A, f, days):
    N = np.zeros((len(days), 3, 3, 3), np.int64)
    np.add.at(N, (np.searchsorted(days, A['day']), A['s'], A['y'], A['pred_' + f]), 1)
    return N


def measures(N):
    """N[s, y, forecast] pooled over days -> the measures (nan where undefined)"""
    dv = lambda a, b: float(a / b) if b else float('nan')
    ct = N[0, 2]; m = dict(recall_calm_turb=dv(ct[2], ct.sum()), calm_turb_forecast_share=[dv(x, ct.sum()) for x in ct],
                          calm_turb_at_least_normal=dv(ct[1] + ct[2], ct.sum()),
                          recall_turb_from_normal=dv(N[1, 2, 2], N[1, 2].sum()), recall_turb_from_turb=dv(N[2, 2, 2], N[2, 2].sum()),
                          recall_turb_all=dv(N[:, 2, 2].sum(), N[:, 2].sum()),
                          flag_rate_from_calm=dv(N[0, :, 2].sum(), N[0].sum()), precision_from_calm=dv(N[0, 2, 2], N[0, :, 2].sum()),
                          base_rate_from_calm=dv(N[0, 2].sum(), N[0].sum()))
    m['lift_from_calm'] = dv(m['precision_from_calm'], m['base_rate_from_calm']) if m['precision_from_calm'] == m['precision_from_calm'] else float('nan')
    return m


def auc_parts(score, pos, di, nd):
    """per positive row and day: negatives below + half the ties (day-block bootstrap of the AUC by weights)"""
    neg = [np.sort(score[(~pos) & (di == k)]) for k in range(nd)]; sp = score[pos]
    lo = np.stack([np.searchsorted(a, sp, 'left') for a in neg], 1); hi_ = np.stack([np.searchsorted(a, sp, 'right') for a in neg], 1)
    return lo + 0.5 * (hi_ - lo), di[pos], np.array([len(a) for a in neg], np.float64)


def auc_w(parts, w):
    B, pd_, nn_ = parts; wp = w[pd_]; return float((wp * (B @ w)).sum() / (wp.sum() * (w * nn_).sum()))


def ci(a):
    a = np.asarray(a, float); a = a[~np.isnan(a)]
    return [round(float(np.quantile(a, .025)), 6), round(float(np.quantile(a, .975)), 6)] if len(a) else None


def episodes(A, pred):
    """runs of consecutive calm -> turbulent rows (consecutive snapshots of one segment)"""
    c = (A['s'] == 0) & (A['y'] == 2); i = np.where(c)[0]
    if not len(i): return dict(n=0)
    br = np.where((np.diff(i) > 1) | (np.diff(A['row'][i]) != 1) | (np.diff(A['day'][i]) != 0) | (np.diff(A['seg'][i]) != 0))[0] + 1
    runs = np.split(i, br); length = [(A['ts'][r[-1]] - A['ts'][r[0]]) / 1e9 for r in runs]
    return dict(n=len(runs), median_sec=round(float(np.median(length)), 1),
                any_turbulent_call={n: round(float(np.mean([(p[r] == 2).any() for r in runs])), 6) for n, p in pred.items()})


def resamples(nd):
    rng = np.random.default_rng(1); return np.stack([np.bincount(rng.integers(0, nd, nd), minlength=nd) for _ in range(NB)]).astype(np.float64)


def recs(): return [json.loads(l) for l in open(RES)] if os.path.exists(RES) else []


def run(split, A, names):
    days = np.unique(A['day']); nd = len(days); di = np.searchsorted(days, A['day']); W = resamples(nd)
    cell = np.zeros((3, 3), np.int64); np.add.at(cell, (A['s'], A['y']), 1)
    calm = A['s'] == 0; ct = calm & (A['y'] == 2); vis = np.expm1(A['past'][:, 0]) > FAST10
    log(dict(stage='cells', split=split, n_days=int(nd), rows=int(len(A['s'])), n=cell.tolist(),
             share=np.round(cell / cell.sum(), 6).tolist(), turbulent_from_state=[round(float(cell[k, 2] / cell[k].sum()), 6) for k in range(3)],
             calm_turb_rows=int(ct.sum()), calm_turb_visible_share=round(float(vis[ct].mean()), 6),
             episodes=episodes(A, {n: A['pred_' + n] for n in names})))
    for n in names:
        N = counts(A, n, days); m = measures(N.sum(0)); B = [measures(np.tensordot(w, N, 1)) for w in W]
        rec = dict(stage='forecast', split=split, name=n, mF1=round(float(f1_score(A['y'], A['pred_' + n], average='macro')), 4),
                   cell_forecast_share=np.round(N.sum(0) / np.maximum(N.sum(0).sum(2, keepdims=True), 1), 6).tolist())
        for k, v in m.items():
            rec[k] = [round(x, 6) for x in v] if isinstance(v, list) else round(v, 6)
            if not isinstance(v, list): rec[k + '_ci95'] = ci([b[k] for b in B])
        p = A['pred_' + n]
        rec['visible'] = dict(rows=[int((ct & vis).sum()), int((ct & ~vis).sum())],
                              recall=[round(float((p[ct & vis] == 2).mean()), 6), round(float((p[ct & ~vis] == 2).mean()), 6)],
                              at_least_normal=[round(float((p[ct & vis] >= 1).mean()), 6), round(float((p[ct & ~vis] >= 1).mean()), 6)])
        log(rec)
    scores = {n: A['score_' + n] for n in names if 'score_' + n in A}
    scores.update({'past10': A['past'][:, 0], 'past60': A['past'][:, 1], 'past300': A['past'][:, 2]})
    pos = A['y'][calm] == 2; dic = di[calm]; one = np.ones(nd)
    parts = {n: auc_parts(sc[calm], pos, dic, nd) for n, sc in scores.items()}
    boot = {n: np.array([auc_w(parts[n], w) for w in W]) for n in parts}
    for n, sc in scores.items():
        a = auc_w(parts[n], one); assert abs(a - roc_auc_score(pos, sc[calm])) < 1e-9
        log(dict(stage='auc', split=split, name=n, auc_calm_turb=round(a, 6), ci95=ci(boot[n]),
                 minus_past10=round(a - auc_w(parts['past10'], one), 6), minus_past10_ci95=ci(boot[n] - boot['past10'])))


def vs_past300(split, A, names):
    days = np.unique(A['day']); nd = len(days); di = np.searchsorted(days, A['day']); W = resamples(nd); one = np.ones(nd)
    calm = A['s'] == 0; pos = A['y'][calm] == 2; ref = auc_parts(A['past'][calm, 2], pos, di[calm], nd)
    rb = np.array([auc_w(ref, w) for w in W])
    for n in names:
        if 'score_' + n not in A: continue
        pt = auc_parts(A['score_' + n][calm], pos, di[calm], nd)
        log(dict(stage='auc_vs_past300', split=split, name=n, minus_past300=round(auc_w(pt, one) - auc_w(ref, one), 6),
                 ci95=ci(np.array([auc_w(pt, w) for w in W]) - rb)))


def topk(split, A, name, qs=(0.001, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20)):
    calm = A['s'] == 0; sc = A['score_' + name][calm]; y = A['y'][calm]; pos = y == 2; o = np.argsort(-sc, kind='stable')
    top = []
    for q in qs:
        t = o[:int(round(q * len(sc)))]
        top.append(dict(top_share=q, rows=int(len(t)), turbulent=round(float(pos[t].mean()), 6), normal_or_turbulent=round(float((y[t] >= 1).mean()), 6),
                        share_of_calm_turb=round(float(pos[t].sum() / pos.sum()), 6), lowest_forecast=round(float(sc[t[-1]]), 4)))
    med = [float(np.median(sc[y == c])) for c in range(3)]
    log(dict(stage='topk', split=split, name=name, calm_rows=int(calm.sum()), base_rate=round(float(pos.mean()), 6), top=top,
             median_forecast_by_next=[round(m, 4) for m in med], median_forecast_ticks_by_next=[round(float(np.expm1(m)), 2) for m in med],
             above_2_85_by_next=[round(float((sc[y == c] > 2.85).mean()), 6) for c in range(3)]))


if __name__ == '__main__':
    _, VA, TE = C.split_days(); have = {(r['stage'], r.get('split')) for r in recs()}
    if ('info', None) not in have: log(dict(stage='info', cuts=CUTS, fast10_ticks=round(FAST10, 4), n_boot=NB))
    for split in ('test', 'val'):
        F, info = test_forecasts(TE) if split == 'test' else val_forecasts(); A, ia = assemble(F)
        names = ['persistence'] + (['r1_gru'] if 'r1_gru' in F else []) + [n for n in ('r2', 'r3', 'r4', 'r5', 'r7') if n in F]
        if ('auc', split) not in have: log(dict(stage='rows', split=split, **info, **ia)); run(split, A, names)
        if ('auc_vs_past300', split) not in have: vs_past300(split, A, names)
        if split == 'val' and ('topk', split) not in have: topk(split, A, 'r7')
