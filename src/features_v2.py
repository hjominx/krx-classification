"""Feature set v2: extra causal features, aligned row by row with data/feat/<day>.npz.
Design fixed before any model was fitted on them. Every value at snapshot t uses only snapshots / trades received at or
before t; day-level scales come from the PREVIOUS trading day.
Groups
  flowc  (15) the 'flow' group of common.py with a causal depth scale: the previous trading day's median L1 depth
              (common.py divides by the same day's median, which is only known at the close)
  hist   (13) mid-change history: seconds since the last mid change, sign and size of the last change, sign of the one
              before it, number of mid changes in the last 1/5/30/120 s, mid return over the last 1/5/30/120/600 s
  rvc    (5)  realised variation (sum of |d mid| in ticks) over the last 5/10/60/300/1800 s
  sprd   (3)  share of the last 5 s / 30 s with spread > 1 tick, seconds since the spread last changed
  ofic   (8)  clock-time OFI: level 1 and levels 1-5, summed over the last 1/5/30/120 s, causal depth scale
  queue  (10) best-queue dynamics: change of the log size of the best ask / bid queue over 1 s and 5 s, log size ratio
              of level 1 to level 2, level-1 size against its 300 s mean, change of log total ask / bid size over 5 s
  depl   (6)  depletion pressure: log buy volume of the last 1/5/30 s minus log best-ask size, and sells against the bid
              (needs the trade tape; zero on days without a trade file)
  ctx    (3)  mid minus its 300 s / 1800 s mean (ticks), position inside the day's running range
  daylvl (2)  volatility level: today's |d mid| per observed minute so far, the previous trading day's value
usage: python features_v2.py [check]  -> data/feat2/<day>.npz (X2, names, groups); 'check' runs the look-ahead test
"""
import os, sys
import numpy as np
from multiprocessing import Pool
import common as C

OUT = os.path.join(C.DATA, 'feat2'); FT = os.path.join(C.DATA, 'feat_tr'); S = 10**9


def day_stats(day):
    D, _ = C.load_continuous(day); ts = D['ts']; mid = (D['ap'][:, 0] + D['bp'][:, 0]) / 2
    gap = np.diff(ts) > C.GAP_NS; dm = np.abs(np.diff(mid)) / C.TICK; dt = np.diff(ts) / S
    return day, float(np.median(D['aq'][:, 0].astype(np.float64) + D['bq'][:, 0]) + 1.0), \
        float(dm[~gap].sum() / max(dt[~gap].sum() / 60.0, 1.0))


def build(arg, n_rows=None):
    day, depth, rv_prev = arg
    D, _ = C.load_continuous(day)
    if n_rows: D = {k: v[:n_rows] for k, v in D.items()}
    ts, ap, bp = D['ts'].astype(np.int64), D['ap'], D['bp']
    aq, bq = D['aq'].astype(np.float64), D['bq'].astype(np.float64)
    n = len(ts); i = np.arange(n); mid = (ap[:, 0] + bp[:, 0]) / 2
    seg = np.concatenate([[0], np.cumsum(np.diff(ts) > C.GAP_NS)])
    first = np.concatenate([[True], np.diff(seg) != 0])
    seg_start = np.maximum.accumulate(np.where(first, i, 0))
    dt = np.concatenate([[0.0], np.diff(ts) / S]); dt[first] = 0
    dmid = np.concatenate([[0.0], np.diff(mid)]) / C.TICK; dmid[first] = 0
    F, names, groups = [], [], []

    def add(g, nm, v):
        F.append(np.asarray(v, np.float64)); names.append(nm); groups.append(g)

    LO = {}

    def lo_of(w):
        if w not in LO: LO[w] = np.maximum(np.searchsorted(ts, ts - w * S, 'right'), seg_start)
        return LO[w]

    def wsum(x, w):                       # sum of per-row increments over rows received in (t - w, t], same segment
        c = np.concatenate([np.zeros((1,) + x.shape[1:]), np.cumsum(x, 0)]); return c[i + 1] - c[lo_of(w)]

    def wmean(v, w):                      # row average of a level over the same rows
        c = np.concatenate([[0.0], np.cumsum(v)]); lo = lo_of(w); return (c[i + 1] - c[lo]) / (i - lo + 1)

    def ago(w):                           # last row received at or before t - w (segment start if none)
        return np.maximum(lo_of(w) - 1, seg_start)

    def last_event(ev):                   # index of the last row with ev True at or before each row, same segment
        j = np.maximum.accumulate(np.where(ev, i, -1)); return j, j >= seg_start

    # ---- flowc: common.py 'flow' with the previous day's depth as scale
    pap, paq, pbp, pbq = (np.roll(a, 1, 0) for a in (ap, aq, bp, bq))
    e_b = (bp >= pbp) * bq - (bp <= pbp) * pbq
    e_a = (ap <= pap) * aq - (ap >= pap) * paq
    ofi_raw = e_b - e_a; ofi_raw[first] = 0; ofi = ofi_raw / depth
    k_lo = lambda k: np.maximum(i - k + 1, seg_start)

    def ksum(x, k):
        c = np.concatenate([[0.0], np.cumsum(x)]); return c[i + 1] - c[k_lo(k)]

    add('flowc', 'c_ofi_l1', ofi[:, 0]); add('flowc', 'c_ofi_l1_5', ofi[:, :5].sum(1))
    for k in (5, 20, 100):
        add('flowc', f'c_ofi_l1_sum{k}', ksum(ofi[:, 0], k)); add('flowc', f'c_ofi_l10_sum{k}', ksum(ofi.sum(1), k))
    dv = np.concatenate([[0.0], np.diff(D['cumvol'])]); dv[first] = 0; dv = np.clip(dv, 0, None)
    ask_hit = (ap[:, 0] > pap[:, 0]) | ((ap[:, 0] == pap[:, 0]) & (aq[:, 0] < paq[:, 0]))
    bid_hit = (bp[:, 0] < pbp[:, 0]) | ((bp[:, 0] == pbp[:, 0]) & (bq[:, 0] < pbq[:, 0]))
    sgn = ask_hit.astype(float) - bid_hit.astype(float); sgn[first] = 0
    add('flowc', 'c_vol', np.log1p(dv))
    for k in (5, 20, 100):
        add('flowc', f'c_vol_sum{k}', np.log1p(ksum(dv, k))); add('flowc', f'c_svol_sum{k}', ksum(dv * sgn, k) / depth)

    # ---- hist
    lc, ok = last_event(dmid != 0)
    add('hist', 'h_t_since_chg', np.log1p(np.clip(np.where(ok, ts - ts[np.maximum(lc, 0)], ts - ts[seg_start]) / S, 0, 600)))
    add('hist', 'h_last_sign', np.where(ok, np.sign(dmid[np.maximum(lc, 0)]), 0.0))
    add('hist', 'h_last_abs', np.where(ok, np.abs(dmid[np.maximum(lc, 0)]), 0.0))
    pc = np.where(lc >= 1, lc[np.maximum(lc - 1, 0)], -1); ok2 = ok & (pc >= seg_start)
    add('hist', 'h_prev_sign', np.where(ok2, np.sign(dmid[np.maximum(pc, 0)]), 0.0))
    for w in (1, 5, 30, 120): add('hist', f'h_nchg_{w}s', np.log1p(wsum((dmid != 0).astype(float), w)))
    for w in (1, 5, 30, 120, 600): add('hist', f'h_ret_{w}s', wsum(dmid, w))

    # ---- rvc
    for w in (5, 10, 60, 300, 1800): add('rvc', f'rv_{w}s', np.log1p(wsum(np.abs(dmid), w)))

    # ---- sprd
    sp = (ap[:, 0] - bp[:, 0]) / C.TICK; wide = (sp > 1.5).astype(float)
    held = np.concatenate([[0.0], wide[:-1]]) * dt        # state held between the previous row and this one
    for w in (5, 30):
        el = wsum(dt, w); add('sprd', f's_wide_{w}s', np.where(el > 0, wsum(held, w) / np.maximum(el, 1e-9), wide))
    dsp = np.concatenate([[0.0], np.diff(sp)]); dsp[first] = 0
    sc, oks = last_event(dsp != 0)
    add('sprd', 's_t_since_chg', np.log1p(np.clip(np.where(oks, ts - ts[np.maximum(sc, 0)], ts - ts[seg_start]) / S, 0, 600)))

    # ---- ofic
    for w in (1, 5, 30, 120):
        add('ofic', f'o_l1_{w}s', wsum(ofi_raw[:, 0], w) / depth); add('ofic', f'o_l5_{w}s', wsum(ofi_raw[:, :5].sum(1), w) / depth)

    # ---- queue
    la, lb = np.log1p(aq[:, 0]), np.log1p(bq[:, 0])
    for w in (1, 5):
        j = ago(w); add('queue', f'q_dla_{w}s', la - la[j]); add('queue', f'q_dlb_{w}s', lb - lb[j])
    add('queue', 'q_a1_a2', la - np.log1p(aq[:, 1])); add('queue', 'q_b1_b2', lb - np.log1p(bq[:, 1]))
    add('queue', 'q_a1_rel300', la - np.log1p(wmean(aq[:, 0], 300))); add('queue', 'q_b1_rel300', lb - np.log1p(wmean(bq[:, 0], 300)))
    lta, ltb = np.log1p(D['tot'][:, 0].astype(np.float64)), np.log1p(D['tot'][:, 1].astype(np.float64)); j = ago(5)
    add('queue', 'q_dtota_5s', lta - lta[j]); add('queue', 'q_dtotb_5s', ltb - ltb[j])

    # ---- depl (trade tape)
    f = os.path.join(FT, f'{day}.npz')
    if os.path.exists(f) and not n_rows:
        t = np.load(f); T = t['T']; tn = list(t['names'])
        for w in (1, 5, 30):
            add('depl', f'd_ask_{w}s', T[:, tn.index(f'tr_buyv_{w}s')] - la); add('depl', f'd_bid_{w}s', T[:, tn.index(f'tr_sellv_{w}s')] - lb)
    else:
        for w in (1, 5, 30): add('depl', f'd_ask_{w}s', np.zeros(n)); add('depl', f'd_bid_{w}s', np.zeros(n))

    # ---- ctx
    for w in (300, 1800): add('ctx', f'x_dev_{w}s', (mid - wmean(mid, w)) / C.TICK)
    hi, lo = np.maximum.accumulate(mid), np.minimum.accumulate(mid)
    add('ctx', 'x_range_pos', (mid - lo) / (hi - lo + C.TICK))

    # ---- daylvl
    add('daylvl', 'v_today', np.log1p(np.cumsum(np.abs(dmid)) / np.maximum(np.cumsum(dt) / 60.0, 1.0)))
    add('daylvl', 'v_prevday', np.full(n, np.log1p(rv_prev)))

    X2 = np.column_stack(F).astype(np.float32)
    assert np.isfinite(X2).all(), day
    if n_rows: return X2
    z = C.load_day(day); assert len(z['ts']) == n and (z['ts'] == ts).all(), day
    np.savez_compressed(os.path.join(OUT, f'{day}.npz'), X2=X2, names=np.array(names), groups=np.array(groups))
    return day, X2.shape


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True); days = C.all_days()
    with Pool(4) as p: st = {d: (a, b) for d, a, b in p.map(day_stats, days)}
    args = [(d, *st[days[max(k - 1, 0)]]) for k, d in enumerate(days)]      # first day uses its own stats (train only)
    if len(sys.argv) > 1 and sys.argv[1] == 'check':
        # look-ahead test: features of the first half of a day must not change when the second half is removed
        for a in (args[30], args[-3]):
            n = len(C.load_continuous(a[0])[0]['ts']); h = n // 2
            full = np.load(os.path.join(OUT, f'{a[0]}.npz')); part = build(a, n_rows=h)
            dep = [k for k, g in enumerate(full['groups']) if g != 'depl']
            print(a[0], 'rows', h, 'max abs difference', float(np.abs(full['X2'][:h][:, dep] - part[:, dep]).max()))
        sys.exit()
    with Pool(4) as p:
        for r in p.imap(build, args): print(*r, flush=True)
