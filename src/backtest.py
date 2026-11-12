"""Backtest of the next-change signal at the 85% rule (threshold 0.3492 on |p-0.5|, fixed on VAL; TEST days only).
Everything below was fixed before running; nothing is tuned on these results.
  capital   : 10,000,000 KRW (example). One position at a time, all capital, whole shares; signals while in a position
              are ignored.
  entry     : signal at snapshot t -> market order filled at snapshot t+1 (one-snapshot latency, ~0.2 s), walking the
              book level by level (buy takes asks, sell takes bids)
  exit      : first mid change after the signal (snapshot j) -> market order filled at snapshot j+1, walking the book;
              if the segment ends first, exit at the last snapshot of the segment
  costs     : commission 0.015% per side (example online rate), sell-side tax 0.20% (assumption for 2026 KOSPI:
              transaction tax 0.05% + rural special tax 0.15%), extra slippage 0 ticks beyond walking the book
  sides     : long-only (retail cannot short easily) is the main case; long-short shown for reference
  scenarios : (a) mid-to-mid, no costs  (b) + crossing the spread  (c) + commission and tax (realistic)
              (d) optimistic upper bound: passive orders always filled at our own quote (buy at bid, sell at ask) + costs
usage: python backtest.py -> results/backtest.jsonl
"""
import json, os
import numpy as np
import common as C
from select_models import OUT

THR = next(json.loads(l)['threshold'] for l in open(os.path.join(OUT, 'next_move_sel.jsonl')) if json.loads(l).get('target') == 0.85)
CAP, COMM, TAX = 10_000_000, 0.00015, 0.0020
_, _, TE = C.split_days()
P = np.load(os.path.join(OUT, 'test_pred_next_move_gbm.npz'))


def book(z):
    L = z['lob'].reshape(-1, 10, 4).astype(np.float64); mid = z['mid'].astype(np.float64)[:, None]
    return mid + L[:, :, 0] * C.TICK, np.round(np.expm1(L[:, :, 1])), mid + L[:, :, 2] * C.TICK, np.round(np.expm1(L[:, :, 3]))


def fill(px, qty, n):                                # walk the book for n shares; returns total cost/proceeds or None
    tot, left = 0.0, n
    for p, q in zip(px, qty):
        take = min(left, q); tot += take * p; left -= take
        if left <= 0: return tot
    return None


rows, k0 = [], 0
for d in TE:
    z = C.load_day(d)
    if float(z['cov']) < C.MIN_COV: continue
    mid, seg = z['mid'], z['seg']; n = len(mid)
    chg = np.where(np.diff(mid) != 0)[0] + 1
    k = np.searchsorted(chg, np.arange(n), 'right'); ok = k < len(chg)
    j = chg[np.minimum(k, len(chg) - 1)]; ok &= seg[j] == seg
    t = np.where(ok)[0]; j = j[ok]
    m = len(t); p = P['prob'][k0:k0 + m, 1]; y = P['y'][k0:k0 + m]; assert (P['day'][k0:k0 + m] == int(d)).all()
    assert ((mid[j] > mid[t]).astype(int) == y).all(); k0 += m
    ap, aq, bp, bq = book(z)
    last_in_seg = np.r_[np.where(np.diff(seg) != 0)[0], n - 1]
    busy_until = -1
    for i in np.where(np.abs(p - 0.5) >= THR)[0]:
        ti, ji = t[i], j[i]
        if ti <= busy_until or ti + 1 >= n or seg[ti + 1] != seg[ti]: continue
        side = 1 if p[i] > 0.5 else -1
        e = ti + 1; x = ji + 1 if ji + 1 < n and seg[ji + 1] == seg[ji] else last_in_seg[np.searchsorted(last_in_seg, ji)]
        sh = int(CAP // ap[e, 0])
        if side > 0: cin, cout = fill(ap[e], aq[e], sh), fill(bp[x], bq[x], sh)     # buy, then sell
        else: cin, cout = fill(bp[e], bq[e], sh), fill(ap[x], aq[x], sh)            # short-sell, then buy back
        if cin is None or cout is None: continue
        busy_until = x
        buy, sell = (cin, cout) if side > 0 else (cout, cin)
        fees = COMM * (buy + sell) + TAX * sell
        pas_buy, pas_sell = (bp[e, 0] * sh, ap[x, 0] * sh) if side > 0 else (bp[x, 0] * sh, ap[e, 0] * sh)
        rows.append(dict(day=d, side=side, sh=sh, correct=int((y[i] == 1) == (side > 0)),
                         hold_s=float((z['ts'][x] - z['ts'][e]) / 1e9), notional=buy,
                         a=side * (mid[x] - mid[e]) * sh, b=sell - buy, c=sell - buy - fees,
                         d=pas_sell - pas_buy - (COMM * (pas_buy + pas_sell) + TAX * pas_sell),
                         fee=fees, spread_ticks=float((ap[e, 0] - bp[e, 0]) / C.TICK)))
assert k0 == len(P['y'])


def summ(rs):
    if not rs: return dict(trades=0)
    a = lambda key: np.array([r[key] for r in rs])
    out = dict(trades=len(rs), trades_per_day=round(len(rs) / len(set(r['day'] for r in rs)), 1),
               signal_correct=round(float(a('correct').mean()), 4), median_hold_s=round(float(np.median(a('hold_s'))), 2),
               mean_notional=round(float(a('notional').mean())), mean_spread_ticks=round(float(a('spread_ticks').mean()), 3),
               mean_fee_per_trade=round(float(a('fee').mean())))
    for key, name in (('a', 'mid_no_cost'), ('b', 'spread_only'), ('c', 'realistic'), ('d', 'passive_upper_bound')):
        v = a(key)
        out[name] = dict(total_krw=round(float(v.sum())), per_trade_krw=round(float(v.mean())), win_rate=round(float((v > 0).mean()), 4),
                         return_on_capital_pct=round(float(v.sum() / CAP * 100), 2))
    days = sorted(set(r['day'] for r in rs))
    out['per_day_realistic_krw'] = {d: round(float(sum(r['c'] for r in rs if r['day'] == d))) for d in days}
    out['per_day_spread_only_krw'] = {d: round(float(sum(r['b'] for r in rs if r['day'] == d))) for d in days}
    return out


res = dict(threshold=THR, capital=CAP, commission=COMM, sell_tax=TAX, test_days=sorted(set(r['day'] for r in rows)),
           long_only=summ([r for r in rows if r['side'] > 0]), long_short=summ(rows))
print(json.dumps(res, indent=1))
with open(os.path.join(OUT, 'backtest.jsonl'), 'a') as f: f.write(json.dumps(res) + '\n')
