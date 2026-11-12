"""KIS H0STCNT0 (real-time trades, 000660) jsonl.gz -> krx_trades/YYYYMMDD.npz, one row per trade.
Every record inside raw_message is used (a message can carry several trades; the stored top-level fields only hold the
last one). Field positions 0-45 are the same in all files (later files append one field at the end).
Duplicates are removed by cumulative volume (field 13), which increases strictly with every trade of the day.
usage: python3 convert_trades.py <src_dir> <out_dir> <helper_dir>  -> also prints one quality line per day
"""
import json, os, sys, collections
import numpy as np
from multiprocessing import Pool

SRC, OUT, HELP = sys.argv[1], sys.argv[2], sys.argv[3]
sys.path.insert(0, HELP)
from convert import members


def one(fn):
    day = fn[:10].replace('-', '')
    b = open(os.path.join(SRC, fn), 'rb').read()
    text = b''.join(c for c, _, _ in members(b)).decode('utf-8', 'replace'); del b
    rows, bad, nmsg, multi, seen, dups = [], 0, 0, 0, set(), 0
    for l in text.split('\n'):
        if not l.strip(): continue
        try:
            r = json.loads(l); rm = json.loads(r['raw_json'])['raw_message']
            _, tr, cnt, body = rm.split('|', 3); n = int(cnt); fs = body.split('^')
            if tr != 'H0STCNT0' or len(fs) % n: bad += 1; continue
            w = len(fs) // n; ts = int(r['recv_ts_ns'])
        except Exception:
            bad += 1; continue
        nmsg += 1; multi += n > 1
        for k in range(n):
            f = fs[w * k: w * (k + 1)]
            if f[13] in seen: dups += 1; continue
            seen.add(f[13])
            try:
                rows.append((ts, int(f[1]), float(f[2]), float(f[12]), float(f[13]), float(f[19]), float(f[20]), float(f[18]),
                             int(f[21] or 0), float(f[10]), float(f[11]), int(f[34] or 0), int(f[15]), int(f[16])))
            except Exception:
                bad += 1
    a = np.array(rows, dtype=np.float64)
    a = a[np.argsort(a[:, 0], kind='stable')]                     # receive order (duplicates dropped while reading)                     # back to receive order
    # missed trades: cumulative volume jumps by more than the trade's own size
    cs = np.sort(a[:, 4]); qs = a[np.argsort(a[:, 4]), 3]
    gap = np.diff(cs) - qs[1:]; missed_vol = float(gap[gap > 0].sum()); missed_n = int((gap > 0).sum())
    np.savez_compressed(os.path.join(OUT, f'{day}.npz'), ts=a[:, 0].astype(np.int64), hhmmss=a[:, 1].astype(np.int32),
                        price=a[:, 2], qty=a[:, 3], cumq=a[:, 4], cum_sell=a[:, 5], cum_buy=a[:, 6], strength=a[:, 7],
                        side=a[:, 8].astype(np.int8), ask1=a[:, 9], bid1=a[:, 10], mkop=a[:, 11].astype(np.int16),
                        n_sell=a[:, 12].astype(np.int64), n_buy=a[:, 13].astype(np.int64))
    cont = (a[:, 1] >= 90000) & (a[:, 1] < 152000)
    return dict(day=day, msgs=nmsg, multi_msgs=multi, trades=len(a), dup_removed=dups, bad=bad,
                cont_trades=int(cont.sum()), missed_jumps=missed_n, missed_vol_share=round(missed_vol / max(a[:, 4].max(), 1), 5),
                side=dict(collections.Counter(a[cont, 8].astype(int).tolist())), mkop=dict(collections.Counter(a[:, 11].astype(int).tolist())),
                first=int(a[:, 1].min()), last=int(a[:, 1].max()))


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    only = set(sys.argv[4].split(',')) if len(sys.argv) > 4 and sys.argv[4] else None
    fs = sorted(f for f in os.listdir(SRC) if f.endswith('.jsonl.gz') and (only is None or f[:10].replace('-', '') in only))
    with Pool(int(sys.argv[5]) if len(sys.argv) > 5 else 4) as p:
        for q in p.imap(one, fs): print(json.dumps(q), flush=True)
