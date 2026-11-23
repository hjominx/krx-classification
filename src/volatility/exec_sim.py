"""Does the volatility forecast lower execution costs? A large buy order split over the day (TEST days only).
Design fixed before running; nothing is tuned on these results.
Order : every 60 s from 09:10:00 to 15:10:00 KST (361 slots per day) a child market buy of q shares, q in 100 / 1,000 /
        3,000 shares (about 0.15 / 1.5 / 4.4 billion KRW at 1.47 million KRW per share; the median ask quantity is about
        840 shares at the best level and 9,300 shares over 10 levels). Every strategy buys the same total per day.
Fill  : decision at the last snapshot at or before the slot time (snapshot i), market order filled at snapshot i+1
        (~0.2 s later) by walking the ask book level by level (as backtest.py). Commission and tax are the same for every
        strategy and left out.
Cost  : per share, (fill price - mid at the decision) / mid, in basis points (half the spread + walking the book + the
        move during the 0.2 s). Secondary: drift cost of waiting - (fill price - mid at the slot where the shares were
        first due) / that mid, for the same shares.
Strategies
  twap        : buy q at every slot
  model       : skip a slot when the forecast for the next 60 s is 'turbulent'; skipped shares are added to later
                non-turbulent slots, at most q extra per slot; from 15:11 every remaining share is bought regardless of
                the forecast (at most 2q per slot, all of it at 15:19). Forecast: the round-5 pipeline (r5_vol_test.npz,
                the first to reach the best TEST score); rounds 2 and 7 are shown for reference
  persistence : the same rule with the class of the previous 60 s (no model)
  oracle      : the same rule with the true class of the next 60 s (an upper bound, not achievable)
  Slots without a forecast (no valid window) count as 'not turbulent'.
Statistics: per day the share-weighted mean cost of each strategy; differences to twap and to persistence averaged over
        the 10 days, with a day-block bootstrap (2,000 resamples) 95% interval. Context: ask depth and its cost by true class.
usage: python exec_sim.py   -> results/volatility/exec_sim.jsonl
"""
import _paths  # noqa: F401  (src/common, src/direction, src/volatility, src/report on the import path)
import json, os, datetime as dt
import numpy as np
import common as C

OUT = C.RES_VOL; os.makedirs(OUT, exist_ok=True); RES = os.path.join(OUT, 'exec_sim.jsonl'); VD = os.path.join(C.DATA, 'feat_vol')
CUTS = [json.loads(l) for l in open(os.path.join(OUT, 'vol_cls.jsonl')) if '"labels"' in l][-1]['cuts']
QS = (100, 1000, 3000); KST = dt.timezone(dt.timedelta(hours=9))
FORECASTS = dict(r5='r5_vol_test.npz', r7='r7_vol_test.npz', r2='vol_v2_test_train.npz')


def log(rec):
    print(json.dumps(rec)[:700], flush=True)
    with open(RES, 'a') as f: f.write(json.dumps(rec) + '\n')


def book(z):                                         # as backtest.py; returns ask prices, ask quantities, bid prices, bid quantities
    L = z['lob'].reshape(-1, 10, 4).astype(np.float64); mid = z['mid'].astype(np.float64)[:, None]
    return mid + L[:, :, 0] * C.TICK, np.round(np.expm1(L[:, :, 1])), mid + L[:, :, 2] * C.TICK, np.round(np.expm1(L[:, :, 3]))


def fill(px, qty, n):
    """price paid for n shares walking the levels; beyond 10 levels the rest at the 10th level price + 1 tick"""
    tot, left = 0.0, n
    for p, q in zip(px, qty):
        take = min(left, q); tot += take * p; left -= take
        if left <= 0: return tot
    return tot + left * (px[-1] + C.TICK)


def slots(d):
    base = dt.datetime.strptime(d, '%Y%m%d').replace(tzinfo=KST)
    main = [base.replace(hour=9, minute=10) + dt.timedelta(minutes=k) for k in range(361)]
    flush = [base.replace(hour=15, minute=11) + dt.timedelta(minutes=k) for k in range(9)]
    return [int(t.timestamp() * 1e9) for t in main], [int(t.timestamp() * 1e9) for t in flush]


def classes(d, n, fc):
    """per snapshot: forecast class of each forecast source, previous-60 s class, true class (-1 where unknown)"""
    v = np.load(os.path.join(VD, f'{d}.npz')); out = {}
    true = np.where(v['ok'], np.digitize(v['rv'], CUTS, right=True), -1); out['oracle'] = true
    out['persistence'] = np.digitize(v['past60'], CUTS, right=True)
    for k, z in fc.items():
        c = np.full(n, -1); m = z['day'] == int(d); c[z['row'][m]] = z['pred'][m]; out[k] = c
    return out


def run_day(d, fc):
    z = C.load_day(d); ts = z['ts'].astype(np.int64); seg = z['seg']; mid = z['mid'].astype(np.float64); ap, aq, bp, bq = book(z)
    cls = classes(d, len(ts), fc); main, flush = slots(d); res = {}
    def snap(t):
        i = np.searchsorted(ts, t, side='right') - 1
        return i if 0 <= i < len(ts) - 1 and seg[i + 1] == seg[i] else None
    for q in QS:
        for strat in ['twap', 'persistence', 'oracle'] + list(fc):
            backlog, due, cost, drift, shares, skipped = [], 0, 0.0, 0.0, 0, 0   # backlog: list of [shares, mid when first due]
            for k, t in enumerate(main + flush):
                i = snap(t); forced = k >= len(main)
                if i is None:                                   # no data at this slot: the shares wait (same for every strategy)
                    if not forced: backlog.append([q, None])
                    continue
                if not forced: backlog.append([q, mid[i]])
                turb = strat != 'twap' and not forced and cls[strat][i] == 2
                if turb: skipped += 1; continue
                cap = sum(b[0] for b in backlog) if (forced and k == len(main) + len(flush) - 1) else (2 * q if strat != 'twap' or forced else q)
                if strat == 'twap' and not forced: cap = sum(b[0] for b in backlog)      # twap buys what is due (only waits on missing data)
                take, n = [], 0
                while backlog and n < cap:
                    b = backlog[0]; x = min(b[0], cap - n); take.append((x, b[1])); n += x; b[0] -= x
                    if b[0] == 0: backlog.pop(0)
                if n == 0: continue
                paid = fill(ap[i + 1], aq[i + 1], n); cost += (paid - n * mid[i]) / mid[i]; shares += n
                drift += sum(x * (paid / n - (m0 if m0 is not None else mid[i])) / (m0 if m0 is not None else mid[i]) for x, m0 in take)
            left = sum(b[0] for b in backlog)
            res[(q, strat)] = dict(cost_bp=cost / shares * 1e4 if shares else np.nan, drift_bp=drift / shares * 1e4 if shares else np.nan,
                                   shares=shares, left=left, skipped=skipped)
    # context: ask depth and the cost of one q-share buy at every snapshot, by true class
    ctx = {}
    for c in (0, 1, 2):
        m = np.where(cls['oracle'][:-1] == c)[0]
        if len(m) == 0: continue
        s = m[::50]; ctx[str(c)] = dict(top_ask_qty=float(np.median(aq[s, 0])), depth10=float(np.median(aq[s].sum(1))),
                                        cost_bp={str(q): float(np.mean([(fill(ap[i + 1], aq[i + 1], q) / q - mid[i]) / mid[i] * 1e4 for i in s])) for q in QS})
    return res, ctx


def boot(a, n=2000, seed=1):
    a = np.asarray(a); rng = np.random.default_rng(seed); bs = [a[rng.integers(0, len(a), len(a))].mean() for _ in range(n)]
    return dict(mean=round(float(a.mean()), 3), ci95=[round(float(np.quantile(bs, .025)), 3), round(float(np.quantile(bs, .975)), 3)],
                days_lower=int((a < 0).sum()))


if __name__ == '__main__':
    _, _, TE = C.split_days(); fc = {k: np.load(os.path.join(OUT, f)) for k, f in FORECASTS.items() if os.path.exists(os.path.join(OUT, f))}
    days = [d for d in TE if float(C.load_day(d)['cov']) >= C.MIN_COV]; per, ctxs = {}, {}
    for d in days:
        res, ctx = run_day(d, fc); per[d] = res; ctxs[d] = ctx
        log(dict(stage='day', day=d, **{f'{q}_{s}': {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()} for (q, s), r in res.items()}, context=ctx))
    strats = ['twap', 'persistence', 'oracle'] + list(fc)
    for q in QS:
        mean = {s: round(float(np.mean([per[d][(q, s)]['cost_bp'] for d in days])), 3) for s in strats}
        diff_twap = {s: boot([per[d][(q, s)]['cost_bp'] - per[d][(q, 'twap')]['cost_bp'] for d in days]) for s in strats if s != 'twap'}
        diff_pers = {s: boot([per[d][(q, s)]['cost_bp'] - per[d][(q, 'persistence')]['cost_bp'] for d in days]) for s in fc}
        drift = {s: round(float(np.mean([per[d][(q, s)]['drift_bp'] for d in days])), 3) for s in strats}
        skip = {s: round(float(np.mean([per[d][(q, s)]['skipped'] for d in days])), 1) for s in strats}
        log(dict(stage='summary', q=q, n_days=len(days), mean_cost_bp=mean, minus_twap=diff_twap, minus_persistence=diff_pers,
                 mean_drift_bp=drift, mean_slots_skipped=skip, left_unfilled=int(sum(per[d][(q, s)]['left'] for d in days for s in strats))))
