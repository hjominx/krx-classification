"""jsonl(.gz) of Hyperliquid l2Book messages -> compact npz (ts, bp, bq, ap, aq).

The collector writes a multi-member gzip (one member per flush). A corrupt member makes gzip.open stop early,
so members are decoded one by one and a corrupt one is skipped by resyncing to the next gzip header.
"""
import json, sys, zlib
import numpy as np

HDR = b'\x1f\x8b\x08'
CH = 1 << 16


def members(b):
    mv = memoryview(b); pos, n = 0, len(b); skipped = 0; truncated = False
    while pos < n:
        d = zlib.decompressobj(31); out = []; p = pos
        try:
            while not d.eof and p < n:
                out.append(d.decompress(mv[p:p + CH])); p = min(p + CH, n)
        except zlib.error:
            skipped += 1
            nxt = b.find(HDR, pos + 1)
            if nxt == -1: break
            pos = nxt; continue
        if not d.eof:                       # file ends inside a member (copied while being written)
            truncated = True; yield b''.join(out), skipped, truncated; return
        pos = p - len(d.unused_data)
        yield b''.join(out), skipped, truncated



if __name__ == '__main__':
    src, dst = sys.argv[1], sys.argv[2]
    ts, bp, bq, ap, aq = [], [], [], [], []
    bad = 0; skipped = 0; truncated = False
    if src.endswith('.gz'):
        raw = open(src, 'rb').read()
        text_parts = []
        for chunk, skipped, truncated in members(raw):
            text_parts.append(chunk)
        text = b''.join(text_parts).decode('utf-8', 'replace')
    else:
        text = open(src).read()
    for line in text.split('\n'):
        if not line.strip():
            continue
        try:
            x = json.loads(line)
        except json.JSONDecodeError:
            bad += 1; continue
        if x.get('event_type') != 'l2Book':
            continue
        b = x['bids'][:5]; a = x['asks'][:5]
        if len(b) < 5 or len(a) < 5 or b[0][0] <= 0 or a[0][0] <= 0:
            bad += 1; continue
        ts.append(x['exchange_ts'])
        bp.append([p for p, _ in b]); bq.append([q for _, q in b])
        ap.append([p for p, _ in a]); aq.append([q for _, q in a])
    ts = np.array(ts, np.int64); o = np.argsort(ts, kind='stable')
    np.savez(dst, ts=ts[o], bp=np.array(bp).reshape(-1, 5)[o], bq=np.array(bq, np.float32).reshape(-1, 5)[o],
             ap=np.array(ap).reshape(-1, 5)[o], aq=np.array(aq, np.float32).reshape(-1, 5)[o])
    print(f'{src}: {len(ts)} books, bad lines {bad}, corrupt members skipped {skipped}, truncated tail {truncated}')
