"""Trade-tape features (KIS H0STCNT0) aligned to every continuous-session book snapshot of krx/feat/<day>.npz.
Causal: at snapshot time t only trades with receive time <= t are used (both streams carry the same local receive clock;
the quote inside each trade message matches the book as-of the trade ~90% of the time and the side/price match peaks at
0-30 ms offset, so no extra lag is needed). Only continuous-session trades (09:00-15:20, market code 20, side 1/5).
Groups (for selection later):
  tw  : time-window flow  - buy / sell volume (log), signed-volume ratio, trade count, for 1 s, 5 s, 30 s
  ev  : event-window flow - signed-volume ratio of the last 5 / 20 / 100 trades
  last: last trade        - side, seconds since it, its price vs mid (ticks), max trade size in the last 5 s,
                            30 s VWAP vs mid (ticks)
  cum : cumulative fields - day buy/sell imbalance from the exchange's cumulative counters, 60 s change of that net
                            volume (robust to missed messages), trade strength (cumulative, /100) and its 60 s change
usage: python trade_features.py -> data/feat_tr/<day>.npz (T, names, groups)
"""
import os, sys
import numpy as np
from multiprocessing import Pool
import common as C

TR_DIR = os.path.join(C.DATA, 'krx_trades'); OUT = os.path.join(C.DATA, 'feat_tr'); os.makedirs(OUT, exist_ok=True)
S = 10**9


def build(day):
    f = os.path.join(TR_DIR, f'{day}.npz')
    if not os.path.exists(f): return day, 'no trades'
    z = C.load_day(day); t = z['ts'].astype(np.int64); mid = z['mid']
    q = np.load(f)
    m = (q['hhmmss'] >= 90000) & (q['hhmmss'] < 152000) & (q['mkop'] == 20) & np.isin(q['side'], [1, 5])
    if m.sum() < 1000: return day, f'too few trades ({m.sum()})'
    o = np.argsort(q['ts'][m], kind='stable')
    ts = q['ts'][m][o]; qty = q['qty'][m][o]; px = q['price'][m][o]; sd = np.where(q['side'][m][o] == 1, 1.0, -1.0)
    cb, cs, st = q['cum_buy'][m][o], q['cum_sell'][m][o], q['strength'][m][o] / 100.0
    bv, sv = np.where(sd > 0, qty, 0.0), np.where(sd < 0, qty, 0.0)
    cbv, csv, cn, cpv = (np.concatenate([[0.0], np.cumsum(a)]) for a in (bv, sv, np.ones_like(qty), px * qty))
    cdv = np.concatenate([[0.0], np.cumsum(sd * qty)]); cqv = np.concatenate([[0.0], np.cumsum(qty)])
    hi = np.searchsorted(ts, t, 'right')                      # trades [0, hi) are known at t
    F, names, groups = [], [], []

    def add(g, n, v): F.append(np.asarray(v, np.float32)); names.append(n); groups.append(g)

    for w in (1, 5, 30):
        lo = np.searchsorted(ts, t - w * S, 'right')
        b, s, n = cbv[hi] - cbv[lo], csv[hi] - csv[lo], cn[hi] - cn[lo]
        add('tw', f'tr_buyv_{w}s', np.log1p(b)); add('tw', f'tr_sellv_{w}s', np.log1p(s))
        add('tw', f'tr_imb_{w}s', (b - s) / (b + s + 1.0)); add('tw', f'tr_n_{w}s', np.log1p(n))
    for k in (5, 20, 100):
        lo = np.maximum(hi - k, 0)
        add('ev', f'tr_imb_last{k}', (cdv[hi] - cdv[lo]) / (cqv[hi] - cqv[lo] + 1.0))
    j = np.maximum(hi - 1, 0); have = hi > 0
    add('last', 'tr_last_side', np.where(have, sd[j], 0.0))
    add('last', 'tr_dt_last', np.log1p(np.clip(np.where(have, (t - ts[j]) / S, 60.0), 0, 60)))
    add('last', 'tr_last_px_dev', np.where(have, (px[j] - mid) / C.TICK, 0.0))
    lo5 = np.searchsorted(ts, t - 5 * S, 'right')
    mx = np.zeros(len(t))
    for i in np.where(hi > lo5)[0]: mx[i] = qty[lo5[i]:hi[i]].max()
    add('last', 'tr_maxq_5s', np.log1p(mx))
    lo30 = np.searchsorted(ts, t - 30 * S, 'right'); vq = cqv[hi] - cqv[lo30]
    add('last', 'tr_vwap30_dev', np.where(vq > 0, ((cpv[hi] - cpv[lo30]) / np.maximum(vq, 1) - mid) / C.TICK, 0.0))
    cbj, csj, stj = np.where(have, cb[j], 0.0), np.where(have, cs[j], 0.0), np.where(have, st[j], 1.0)
    add('cum', 'tr_day_imb', (cbj - csj) / (cbj + csj + 1.0))
    j60 = np.maximum(np.searchsorted(ts, t - 60 * S, 'right') - 1, 0); h60 = np.searchsorted(ts, t - 60 * S, 'right') > 0
    dnet = (cbj - csj) - np.where(h60, cb[j60] - cs[j60], 0.0); dtot = (cbj + csj) - np.where(h60, cb[j60] + cs[j60], 0.0)
    add('cum', 'tr_net_60s', dnet / (dtot + 1.0))
    add('cum', 'tr_strength', stj); add('cum', 'tr_strength_d60', stj - np.where(h60, st[j60], stj))
    T = np.column_stack(F)
    assert np.isfinite(T).all()
    np.savez_compressed(os.path.join(OUT, f'{day}.npz'), T=T, names=np.array(names), groups=np.array(groups))
    return day, f'ok {T.shape} trades {len(ts)}'


if __name__ == '__main__':
    with Pool(4) as p:
        for d, msg in p.imap(build, C.all_days()): print(d, msg, flush=True)
