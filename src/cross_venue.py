"""Cross-venue experiment: does the Hyperliquid xyz:SKHX perp (24/7, USD) help predict the KRX 000660 mid?
Only days with both KRX and HL data (07-09 .. 08-28). Same GBM, same rows, with vs without HL features.
Leakage guard: an HL book counts as known at KRX time t only if its exchange time <= t - LAG (HL stores exchange ms,
KRX stores local receive ns), LAG = 500 ms.
Split by day: train 07-09..08-07, test 08-10..08-28 (fixed before running).
usage: python cross_venue.py -> results/cross.jsonl
"""
import json, os
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
import common as C
from select_models import gbm, OUT

HL = os.path.join(C.DATA, 'hl_npz')
LAG_NS = 500 * 10**6


def hl_feats(day, ts):
    f = os.path.join(HL, f'{day[4:]}_SKHX.npz')
    if not os.path.exists(f): return None
    z = np.load(f); hts = z['ts'].astype(np.int64) * 10**6
    hmid = (z['bp'][:, 0] + z['ap'][:, 0]) / 2
    hb, ha = z['bq'][:, 0].astype(float), z['aq'][:, 0].astype(float)
    j = np.searchsorted(hts, ts - LAG_NS, 'right') - 1
    ok = j >= 0; j = np.maximum(j, 0)
    stale = (ts - LAG_NS - hts[j]) / 1e9
    def ret(sec):
        jj = np.maximum(np.searchsorted(hts, ts - LAG_NS - sec * 10**9, 'right') - 1, 0)
        return (hmid[j] - hmid[jj]) / hmid[jj] * 1e4
    imb = (hb[j] - ha[j]) / (hb[j] + ha[j] + 1e-9)
    F = np.column_stack([ret(1), ret(5), ret(30), imb, np.log1p(np.clip(stale, 0, 60))])
    ok &= stale < 5          # HL feed alive
    return F, ok


if __name__ == '__main__':
    names = list(C.load_day('20260709')['names'])
    days = [d for d in C.all_days() if d <= '20260828']
    tr = [d for d in days if d <= '20260807']; te = [d for d in days if d >= '20260810']
    for H in (2, 5, 10):
        hi = C.HORIZONS.index(H)
        parts = {}
        for name, ds in (('train', tr), ('test', te)):
            X, XH, Y, used = [], [], [], []
            for d in ds:
                z = C.load_day(d)
                if name == 'test' and z['cov'] < C.MIN_COV: continue
                r = hl_feats(d, z['ts'])
                if r is None: continue
                Fh, okh = r
                idx = np.where(z['V'][:, hi] & okh)[0][:: (2 if name == 'train' else 1)]
                if len(idx) == 0: continue
                X.append(z['X'][idx]); XH.append(np.column_stack([Fh[idx]])); Y.append(z['Y'][idx, hi]); used.append(d)
            parts[name] = (np.concatenate(X), np.concatenate(XH), np.concatenate(Y).astype(int), used)
        Xtr, Htr, ytr, dtr = parts['train']; Xte, Hte, yte, dte = parts['test']
        res = dict(H=H, train_days=dtr, test_days=dte, n_train=len(ytr), n_test=len(yte))
        for nm, A, B in (('krx_only', Xtr, Xte), ('krx+hl', np.hstack([Xtr, Htr]), np.hstack([Xte, Hte])), ('hl_only', Htr, Hte)):
            m = gbm().fit(A, ytr); p = m.predict(B)
            res[nm] = dict(mF1=round(f1_score(yte, p, average='macro'), 4), acc=round(accuracy_score(yte, p), 4))
        maj = np.bincount(ytr, minlength=3).argmax()
        res['majority_acc'] = round(accuracy_score(yte, np.full_like(yte, maj)), 4)
        print(json.dumps(res), flush=True)
        with open(os.path.join(OUT, 'cross.jsonl'), 'a') as f: f.write(json.dumps(res) + '\n')
