"""KIS H0STASP0 (KRX real-time 10-level order book) jsonl.gz -> npz, one file per KST trading date.

Fields taken from raw_json (KIS H0STASP0 layout):
  [1] business hour HHMMSS, [2] hour class code (0 = continuous, A = closing auction, B = opening auction, ...)
  [3:13] ask prices 1-10, [13:23] bid prices 1-10, [23:33] ask sizes, [33:43] bid sizes,
  [43] total ask size, [44] total bid size, [53] cumulative volume
Time base is recv_ts_ns (exchange time in the message has 1 s resolution).
usage: python convert_krx.py <src.jsonl.gz> <out_dir> <helper_dir>
"""
import json, sys, os
import numpy as np
sys.path.insert(0, sys.argv[3])
from convert import members

src, out_dir = sys.argv[1], sys.argv[2]
raw = open(src, 'rb').read()
skipped = trunc = 0
parts = []
for chunk, skipped, trunc in members(raw):
    parts.append(chunk)
text = b''.join(parts).decode('utf-8', 'replace')

ts, cls, hhmmss, P, Q, tot, vol = [], [], [], [], [], [], []
bad = dup = 0
seen_prev = None
for line in text.split('\n'):
    if not line.strip():
        continue
    try:
        x = json.loads(line); f = json.loads(x['raw_json'])['fields']
        if len(f) < 54:
            bad += 1; continue
        rec = (f[1:2] + f[2:3] + f[3:45] + f[53:54])
        if rec == seen_prev:          # exact repeat of previous message
            dup += 1; continue
        seen_prev = rec
        ts.append(int(x['recv_ts_ns'])); cls.append(f[2]); hhmmss.append(int(f[1]))
        P.append([float(v) for v in f[3:23]]); Q.append([float(v) for v in f[23:43]])
        tot.append([float(f[43]), float(f[44])]); vol.append(float(f[53]))
    except (ValueError, KeyError, TypeError, json.JSONDecodeError):
        bad += 1
ts = np.array(ts, np.int64); o = np.argsort(ts, kind='stable')
P = np.array(P).reshape(-1, 20)[o]; Q = np.array(Q, np.float32).reshape(-1, 20)[o]
date = ''
if len(ts):
    d = (ts[o] // 10**9 + 9 * 3600) // 86400
    main = np.bincount(d - d.min()).argmax() + d.min()
    import datetime as dt
    date = (dt.date(1970, 1, 1) + dt.timedelta(days=int(main))).strftime('%Y%m%d')
    np.savez(os.path.join(out_dir, f'KRX_{date}.npz'), ts=ts[o], cls=np.array(cls)[o], hhmmss=np.array(hhmmss)[o],
             ap=P[:, :10], bp=P[:, 10:], aq=Q[:, :10], bq=Q[:, 10:],
             tot=np.array(tot, np.float32)[o], cumvol=np.array(vol)[o])
print(f'{src} -> {date or "EMPTY"}: {len(ts)} msgs, bad {bad}, repeats dropped {dup}, corrupt members {skipped}, truncated {trunc}')
