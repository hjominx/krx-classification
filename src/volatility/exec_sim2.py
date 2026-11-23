"""Execution simulation 2: an adaptive buy schedule driven by the volatility forecast, chosen on VAL, scored on TEST once.
Follows exec_sim.py, where skipping 'turbulent' slots and buying the skipped shares later in double-size orders cost
more than buying evenly (the extra depth walked outweighed the cheaper slots). Design fixed before running.
Order, fill and cost: as exec_sim.py - a child market buy every 60 s from 09:10 to 15:10 (361 slots), q = 100 / 1,000 /
        3,000 shares per slot on average, filled at the next snapshot walking the asks; cost per share in bp against the
        mid at the decision. Every strategy buys N = 361 q shares per day; from 15:11 the rest is bought (at most 2q per
        slot, all of it at 15:19).
Adaptive rule (a, b, L): at slot k the schedule is (k + 1) q shares bought so far.
        forecast turbulent: buy a x q; calm: buy (1 + b) x q; normal or no forecast: buy q;
        then keep the deviation from the schedule within +-L x q (buy less when too far ahead, more when too far behind).
        Grid: a in 0 / 0.25 / 0.5 / 0.75, b in 0 / 0.1 / 0.25 / 0.5, L in 5 / 20 / 60 (48 rules).
TWAP:  buys the schedule exactly (q per slot; after a slot without data it catches up at the next one).
Signals: the forecast of round 5 (0.7 x GBM regression + 0.3 x conv+GRU 'vreg_mh' bag, thresholds [2.0, 2.85]) - main;
        round 7 and round 2 - reference; the class of the previous 60 s (no model); the true class (upper bound).
        On VAL the forecasts are rebuilt from the saved VAL predictions with the weights and thresholds chosen in those
        rounds; on TEST they are the saved TEST predictions (r5_vol_test.npz, r7_vol_test.npz, vol_v2_test_train.npz).
Choice: for every signal and every q, the rule with the lowest mean cost over the 12 VAL days.
TEST:   the chosen rules once; differences to TWAP and to the previous-60 s rule averaged over the 10 TEST days, with a
        day-block bootstrap (2,000 resamples) 95% interval.
usage: python exec_sim2.py   -> results/volatility/exec_sim2.jsonl
"""
import _paths  # noqa: F401  (src/common, src/direction, src/volatility, src/report on the import path)
import json, os, sys, time
import numpy as np
import common as C
import data_v2 as E

argv = sys.argv; sys.argv = argv[:1]
import exec_sim as X
import vol_r4 as R4
sys.argv = argv
OUT = X.OUT; RES = os.path.join(OUT, 'exec_sim2.jsonl'); QS = X.QS
GRID = [(a, b, L) for a in (0.0, 0.25, 0.5, 0.75) for b in (0.0, 0.1, 0.25, 0.5) for L in (5, 20, 60)]
SIGS = ['r5', 'r7', 'r2', 'persistence', 'oracle']


def log(rec):
    print(json.dumps(rec)[:700], flush=True)
    with open(RES, 'a') as f: f.write(json.dumps(rec) + '\n')


def val_forecasts():
    """class forecasts on VAL rows, rebuilt with the weights and thresholds chosen in rounds 2, 5 and 7"""
    def load(f): z = np.load(os.path.join(OUT, f)); return R4.KEY(z['day'], z['row']), z['pred'].astype(np.float64)
    kg, gb = load('vol_v2_val_gbr.npz'); km, mk = load('r7_val_vol_mkt.npz')
    cs = [load(f'r5_val_vreg_mh_s{s}_vol.npz') for s in range(5)]; kc = cs[0][0]
    for k, _ in cs[1:]: assert (k == kc).all()
    cg = np.mean([p for _, p in cs], 0)
    def on(keys, k, p): ix = np.searchsorted(k, keys); assert (k[ix] == keys).all(); return p[ix]
    out = {}
    out['r2'] = (kg, R4.by_thr(gb, [2.0, 2.85]))
    k5 = np.intersect1d(kg, kc); out['r5'] = (k5, R4.by_thr(0.7 * on(k5, kg, gb) + 0.3 * on(k5, kc, cg), [2.0, 2.85]))
    k7 = np.intersect1d(km, kc); out['r7'] = (k7, R4.by_thr(0.9 * on(k7, km, mk) + 0.1 * on(k7, kc, cg), [2.05, 2.85]))
    return {n: dict(day=k // 1_000_000, row=k % 1_000_000, pred=p) for n, (k, p) in out.items()}


def test_forecasts():
    return {n: np.load(os.path.join(OUT, f)) for n, f in (('r5', 'r5_vol_test.npz'), ('r7', 'r7_vol_test.npz'), ('r2', 'vol_v2_test_train.npz'))}


def day_data(d, fc):
    z = C.load_day(d); ts = z['ts'].astype(np.int64); seg = z['seg']; mid = z['mid'].astype(np.float64); ap, aq, bp, bq = X.book(z)
    cls = X.classes(d, len(ts), fc); main, flush = X.slots(d)
    def snap(t):
        i = np.searchsorted(ts, t, side='right') - 1
        return i if 0 <= i < len(ts) - 1 and seg[i + 1] == seg[i] else None
    return dict(mid=mid, ap=ap, aq=aq, cls=cls, main=[snap(t) for t in main], flush=[snap(t) for t in flush])


def simulate(D, q, sig, rule):
    """per-share cost (bp) of one day; sig None = TWAP"""
    N = len(D['main']) * q; done, cost = 0, 0.0
    def buy(i, n):
        nonlocal done, cost
        n = int(min(max(n, 0), N - done))
        if n <= 0: return
        paid = X.fill(D['ap'][i + 1], D['aq'][i + 1], n); cost += (paid - n * D['mid'][i]) / D['mid'][i]; done += n
    for k, i in enumerate(D['main']):
        if i is None: continue
        due = (k + 1) * q
        if sig is None: want = due - done
        else:
            a, b, L = rule; c = D['cls'][sig][i]
            want = (a if c == 2 else 1 + b if c == 0 else 1.0) * q
            dev = done + want - due
            if dev > L * q: want -= dev - L * q
            elif dev < -L * q: want += -L * q - dev
        buy(i, round(want))
    for k, i in enumerate(D['flush']):
        if i is None: continue
        buy(i, N - done if k == len(D['flush']) - 1 else 2 * q)
    return cost / done * 1e4 if done else np.nan, N - done


if __name__ == '__main__':
    t0 = time.time()
    va = [d for d in E.VA if float(C.load_day(d)['cov']) >= C.MIN_COV]; te = [d for d in E.TE if float(C.load_day(d)['cov']) >= C.MIN_COV]
    fv = val_forecasts(); VA_D = {d: day_data(d, fv) for d in va}
    chosen = {}
    for q in QS:
        tw = [simulate(VA_D[d], q, None, None)[0] for d in va]
        log(dict(stage='val_twap', q=q, mean_cost_bp=round(float(np.mean(tw)), 3)))
        for s in SIGS:
            scores = []
            for rule in GRID:
                c = [simulate(VA_D[d], q, s, rule)[0] for d in va]; scores.append((float(np.mean(c)), rule))
            scores.sort(); best = scores[0]; chosen[(s, q)] = best[1]
            log(dict(stage='val_choice', q=q, signal=s, rule=dict(a=best[1][0], b=best[1][1], L=best[1][2]), val_cost_bp=round(best[0], 3),
                     val_twap_bp=round(float(np.mean(tw)), 3), val_minus_twap=round(best[0] - float(np.mean(tw)), 3),
                     top5=[(round(c_, 3), r_) for c_, r_ in scores[:5]], sec=round(time.time() - t0)))
    del VA_D
    ft = test_forecasts(); TE_D = {d: day_data(d, ft) for d in te}
    for q in QS:
        tw = np.array([simulate(TE_D[d], q, None, None)[0] for d in te]); res = {'twap': tw}; left = 0
        for s in SIGS:
            out = [simulate(TE_D[d], q, s, chosen[(s, q)]) for d in te]; res[s] = np.array([o[0] for o in out]); left += sum(o[1] for o in out)
        log(dict(stage='test', q=q, n_days=len(te), rules={s: dict(a=chosen[(s, q)][0], b=chosen[(s, q)][1], L=chosen[(s, q)][2]) for s in SIGS},
                 mean_cost_bp={s: round(float(v.mean()), 3) for s, v in res.items()},
                 minus_twap={s: X.boot(res[s] - tw) for s in SIGS},
                 minus_persistence={s: X.boot(res[s] - res['persistence']) for s in ('r5', 'r7', 'r2', 'oracle')},
                 left_unfilled=int(left), sec=round(time.time() - t0)))
