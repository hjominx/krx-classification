"""Diagnostic for experiment 3 (VAL only, after the TEST choice was made): which part of the 'norm' feature set hurts?
fixed TRAIN->VAL GBM, same settings; variants: norm without tick_bp column, only the log-size shift, only the OFI rescale.
Also prints the 5 s label mix of each training window. -> results/direction/norm_diag.jsonl"""
import _paths  # noqa: F401  (src/common, src/direction, src/volatility, src/report on the import path)
import json, os, sys
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
sys.argv = ['norm_wf.py', 'none']
exec(open('norm_wf.py').read().split("VARIANTS = ")[0])

def build(d, parts):
    z = C.load_day(d); v = z['V'][:, hi]; X = z['X'][v].astype(np.float32)
    i = DAYS.index(d); prev = DAYS[max(0, i - 5):i] or [d]
    if 'log' in parts: X[:, LOGC] -= np.mean([stats[p]['logmean'] for p in prev], 0).astype(np.float32)
    if 'flow' in parts: X[:, FLOWC] *= np.float32(stats[d]['depth'] / np.median([stats[p]['depth'] for p in prev]))
    if 'tick' in parts: X = np.hstack([X, (1e7 / z['mid'][v]).astype(np.float32)[:, None]])
    return X, z['Y'][v, hi].astype(int)

ev = [d for d in VA if C.load_day(d)['cov'] >= C.MIN_COV]
for name, parts in (('norm_no_tick', ('log', 'flow')), ('log_shift_only', ('log',)), ('ofi_rescale_only', ('flow',)), ('raw_plus_tick', ('tick',))):
    tr = [build(d, parts) for d in TR]; Xtr = np.concatenate([a[::SUB] for a, _ in tr]); ytr = np.concatenate([b[::SUB] for _, b in tr])
    m = gbm(**HP).fit(Xtr, ytr)
    va = [build(d, parts) for d in ev]; X = np.concatenate([a for a, _ in va]); y = np.concatenate([b for _, b in va]); p = m.predict(X)
    r = dict(variant=name, val_mF1=round(f1_score(y, p, average='macro'), 4), val_acc=round(accuracy_score(y, p), 4))
    print(r, flush=True); open(os.path.join(OUT, 'norm_diag.jsonl'), 'a').write(json.dumps(r) + '\n')
mix = lambda ds: np.round(np.bincount(np.concatenate([rows(d, False)[1] for d in ds]), minlength=3) / sum(len(rows(d, False)[1]) for d in ds), 3).tolist()
r = dict(variant='label_mix', TRAIN=mix(TR), last20_before_val=mix([d for d in DAYS if d < VA[0]][-20:]), VAL=mix(ev))
print(r); open(os.path.join(OUT, 'norm_diag.jsonl'), 'a').write(json.dumps(r) + '\n')
