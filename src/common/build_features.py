"""Build and cache per-day features/labels for every KRX day. usage: python build_features.py"""
import _paths  # noqa: F401  (src/common, src/direction, src/volatility, src/report on the import path)
import os, sys, json
from multiprocessing import Pool
import numpy as np
import common as C


def one(day):
    out = os.path.join(C.FEAT, f'{day}.npz')
    if not os.path.exists(out):
        r = C.build_features(day)
        np.savez(out, **r)
    else:
        r = np.load(out)
    return day, int(len(r['ts'])), float(r['cov']), int(r['X'].shape[1])


if __name__ == '__main__':
    os.makedirs(C.FEAT, exist_ok=True)
    with Pool(4) as p:
        rows = p.map(one, C.all_days())
    tr, va, te = C.split_days()
    for d, n, cov, nf in rows:
        part = 'train' if d in tr else 'val' if d in va else 'test'
        print(json.dumps(dict(day=d, part=part, events=n, coverage=round(cov, 3), n_features=nf,
                              eval_ok=cov >= C.MIN_COV)))
