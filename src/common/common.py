"""Shared data/feature/label code for KRX 000660 (SK hynix) short-horizon mid-price direction.

Design fixed before looking at any test result:
  * continuous session only: hour class '0', 09:00 <= t < 15:20 KST (opening/closing auctions and after-hours dropped)
  * a silence > 5 s breaks the sequence (collector reconnects); no window or label crosses a break
  * label: sign of mid(t + H s) - mid(t) in ticks (1 tick = 1,000 KRW for this price range); 0 = down, 1 = flat, 2 = up
  * split by day: TRAIN 07-09..10-02, VAL 10-06..10-26 (selection), TEST 10-27..11-09 (final, looked at once)
  * days with < 60 % minute coverage of the session are used for training only
"""
import _paths  # noqa: F401  (src/common, src/direction, src/volatility, src/report on the import path)
import glob, os
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repository root (this file is in src/common)
DATA = os.path.join(ROOT, 'data')
RESULTS = os.environ.get('KRX_RESULTS') or os.path.join(ROOT, 'results')   # KRX_RESULTS: another results root (e.g. a GPU run)
RES_DIR, RES_VOL = os.path.join(RESULTS, 'direction'), os.path.join(RESULTS, 'volatility')   # results of the two tasks
NPZ = os.path.join(DATA, 'krx_npz')
FEAT = os.path.join(DATA, 'feat')
TICK = 1000.0
GAP_NS = 5 * 10**9
HORIZONS = (1, 2, 5, 10, 30)
MIN_COV = 0.60
LEVELS = 10


def all_days():
    return sorted(os.path.basename(f)[4:12] for f in glob.glob(os.path.join(NPZ, 'KRX_*.npz')))


def split_days():
    days = all_days()
    tr = [d for d in days if d <= '20261002']
    va = [d for d in days if '20261006' <= d <= '20261026']
    te = [d for d in days if d >= '20261027']
    return tr, va, te


def load_continuous(day):
    z = np.load(os.path.join(NPZ, f'KRX_{day}.npz'))
    ts = z['ts']; m = ((ts // 10**9 + 9 * 3600) % 86400) / 60.0
    keep = (z['cls'] == '0') & (m >= 540) & (m < 920)
    keep &= (z['ap'][:, 0] > 0) & (z['bp'][:, 0] > 0) & (z['ap'][:, 0] > z['bp'][:, 0])
    out = {k: z[k][keep] for k in ('ts', 'ap', 'bp', 'aq', 'bq', 'tot', 'cumvol')}
    out['minute'] = m[keep]
    cov = len(np.unique(out['minute'].astype(int))) / 380.0
    return out, cov


def _wsum(x, k, seg_start_idx):
    """sum of the last k values (inclusive) without crossing a segment start."""
    n = len(x); c = np.concatenate([[0.0], np.cumsum(x, 0)])
    i = np.arange(n); j = np.maximum(i - k + 1, seg_start_idx)
    return c[i + 1] - c[j]


FEATURE_GROUPS = {}


def build_features(day):
    D, cov = load_continuous(day)
    ts, ap, bp = D['ts'], D['ap'], D['bp']
    aq, bq = D['aq'].astype(np.float64), D['bq'].astype(np.float64)
    n = len(ts)
    mid = (ap[:, 0] + bp[:, 0]) / 2
    seg = np.concatenate([[0], np.cumsum(np.diff(ts) > GAP_NS)])
    first = np.concatenate([[True], np.diff(seg) != 0])
    seg_start = np.maximum.accumulate(np.where(first, np.arange(n), 0))

    F, names = [], []
    def add(group, name, v):
        F.append(np.asarray(v, np.float64).reshape(n, -1)); k = F[-1].shape[1]
        names.extend([f'{name}{i}' if k > 1 else name for i in range(k)])
        FEATURE_GROUPS.setdefault(group, []).extend(names[-k:])

    # --- book shape (10 levels)
    add('book', 'ask_dist', (ap - mid[:, None]) / TICK)
    add('book', 'bid_dist', (mid[:, None] - bp) / TICK)
    add('book', 'log_aq', np.log1p(aq)); add('book', 'log_bq', np.log1p(bq))
    for L in (1, 3, 5, 10):
        add('book', f'imb{L}', (bq[:, :L].sum(1) - aq[:, :L].sum(1)) / (bq[:, :L].sum(1) + aq[:, :L].sum(1) + 1e-9))
    ta, tb = D['tot'][:, 0].astype(float), D['tot'][:, 1].astype(float)
    add('book', 'imb_total', (tb - ta) / (tb + ta + 1e-9))
    add('book', 'spread_ticks', (ap[:, 0] - bp[:, 0]) / TICK)
    micro = (bp[:, 0] * aq[:, 0] + ap[:, 0] * bq[:, 0]) / (aq[:, 0] + bq[:, 0] + 1e-9)
    add('book', 'micro_dev', (micro - mid) / TICK)

    # --- past price path
    dmid = np.concatenate([[0.0], np.diff(mid)]) / TICK; dmid[first] = 0
    for k in (1, 5, 20, 100):
        add('price', f'ret{k}', _wsum(dmid, k, seg_start))
    add('price', 'rv100', np.sqrt(_wsum(dmid ** 2, 100, seg_start)))
    dt = np.concatenate([[0], np.diff(ts)]) / 1e9; dt[first] = 0
    add('price', 'log_dt', np.log1p(np.clip(dt, 0, 5)))

    # --- order flow: multi-level OFI (Cont et al.), traded volume and its estimated sign
    pap, paq, pbp, pbq = (np.roll(a, 1, 0) for a in (ap, aq, bp, bq))
    e_b = (bp >= pbp) * bq - (bp <= pbp) * pbq
    e_a = (ap <= pap) * aq - (ap >= pap) * paq
    ofi = (e_b - e_a); ofi[first] = 0
    depth = np.median(aq[:, 0] + bq[:, 0]) + 1.0
    ofi = ofi / depth
    add('flow', 'ofi_l1', ofi[:, 0]); add('flow', 'ofi_l1_5', ofi[:, :5].sum(1))
    for k in (5, 20, 100):
        add('flow', f'ofi_l1_sum{k}', _wsum(ofi[:, 0], k, seg_start))
        add('flow', f'ofi_l10_sum{k}', _wsum(ofi.sum(1), k, seg_start))
    dv = np.concatenate([[0.0], np.diff(D['cumvol'])]); dv[first] = 0; dv = np.clip(dv, 0, None)
    ask_hit = (ap[:, 0] > pap[:, 0]) | ((ap[:, 0] == pap[:, 0]) & (aq[:, 0] < paq[:, 0]))
    bid_hit = (bp[:, 0] < pbp[:, 0]) | ((bp[:, 0] == pbp[:, 0]) & (bq[:, 0] < pbq[:, 0]))
    sgn = ask_hit.astype(float) - bid_hit.astype(float); sgn[first] = 0
    vnorm = depth
    add('flow', 'vol', np.log1p(dv))
    for k in (5, 20, 100):
        add('flow', f'vol_sum{k}', np.log1p(_wsum(dv, k, seg_start)))
        add('flow', f'svol_sum{k}', _wsum(dv * sgn, k, seg_start) / vnorm)

    # --- time of day
    add('time', 'min_since_open', (D['minute'] - 540) / 380)
    add('time', 'open30', (D['minute'] < 570).astype(float))
    add('time', 'close30', (D['minute'] >= 890).astype(float))

    X = np.concatenate(F, 1).astype(np.float32)
    # labels for every horizon; valid = target inside the same segment
    seg_end = np.zeros(seg.max() + 1, np.int64); np.maximum.at(seg_end, seg, ts)
    Y, V = [], []
    for H in HORIZONS:
        e = np.searchsorted(ts, ts + H * 10**9, 'right') - 1
        d = (mid[e] - mid) / TICK
        Y.append(np.where(d > 0.25, 2, np.where(d < -0.25, 0, 1)).astype(np.int8))
        V.append(ts + H * 10**9 <= seg_end[seg])
    # raw 10-level book for the deep models: prices in ticks from mid, log sizes (DeepLOB layout a_p,a_v,b_p,b_v per level)
    lob = np.stack([(ap - mid[:, None]) / TICK, np.log1p(aq), (bp - mid[:, None]) / TICK, np.log1p(bq)], 2).reshape(n, 40)
    return dict(X=X, names=np.array(names), Y=np.stack(Y, 1), V=np.stack(V, 1), seg=seg, ts=ts, mid=mid,
                minute=D['minute'].astype(np.float32), lob=lob.astype(np.float32), cov=cov)


def load_day(day):
    return dict(np.load(os.path.join(FEAT, f'{day}.npz')))


def groups_of(names):
    g = {}
    for i, nm in enumerate(names):
        for grp, lst in FEATURE_GROUPS_STATIC.items():
            if any(nm == p or (nm.startswith(p) and nm[len(p):].isdigit()) for p in lst):
                g.setdefault(grp, []).append(i); break
    return g


FEATURE_GROUPS_STATIC = {
    'book': ['ask_dist', 'bid_dist', 'log_aq', 'log_bq', 'imb1', 'imb3', 'imb5', 'imb10', 'imb_total', 'spread_ticks', 'micro_dev'],
    'price': ['ret1', 'ret5', 'ret20', 'ret100', 'rv100', 'log_dt'],
    'flow': ['ofi_l1', 'ofi_l1_5', 'ofi_l1_sum5', 'ofi_l10_sum5', 'ofi_l1_sum20', 'ofi_l10_sum20', 'ofi_l1_sum100',
             'ofi_l10_sum100', 'vol', 'vol_sum5', 'svol_sum5', 'vol_sum20', 'svol_sum20', 'vol_sum100', 'svol_sum100'],
    'time': ['min_since_open', 'open30', 'close30'],
}
