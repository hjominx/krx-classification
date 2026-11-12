"""Volatility-regime classification (new task, design fixed before running any model):
  target   : RV60(t) = sum of |mid changes| in ticks over (t, t+60 s], same segment only (else invalid)
  classes  : 0 = calm, 1 = normal, 2 = turbulent, cut at the TRAIN terciles of RV60
  inputs   : the 71 existing features + log1p past RV over 10 / 60 / 300 s (same segment, causal)
  split    : the same TRAIN / VAL / TEST days; VAL/TEST days need coverage >= MIN_COV; metric macro-F1
  models   : persistence (class of past-60 s RV with the same cuts), time-of-day only (GBM on minute),
             logistic, GBM (HistGB, balanced), conv+GRU (tuned base T=50 h64, 8 x 200k, patience 2, balanced)
  final    : model with the best VAL macro-F1 is the chosen model; TEST is scored once for every model (reported all)
usage: python vol_cls.py labels | base | gru | test   -> results/vol_cls.jsonl
"""
import json, os, sys, time
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, recall_score
import common as C

STAGE = sys.argv[1]
OUT = os.path.join(C.ROOT, 'results'); RES = os.path.join(OUT, 'vol_cls.jsonl'); VD = os.path.join(C.DATA, 'feat_vol')
TR, VA, TE = C.split_days()
log = lambda r: (print(json.dumps(r), flush=True), open(RES, 'a').write(json.dumps(r) + '\n'))


def rv(ts, mid, seg, w_ns, future):
    a = np.concatenate([[0.0], np.cumsum(np.abs(np.diff(mid)) / C.TICK)])  # a[i] = path length 0..i
    n = len(ts); i = np.arange(n)
    if future:
        j = np.searchsorted(ts, ts + w_ns, side='right') - 1
        ok = (j < n) & (ts[np.minimum(j + 1, n - 1)] >= ts + w_ns) & (j + 1 < n)  # window fully observed
        j = np.minimum(j, n - 1); ok &= seg[j] == seg
        return a[j] - a[i], ok
    j = np.searchsorted(ts, ts - w_ns, side='left'); j = np.maximum(j, np.searchsorted(seg, seg, side='left'))
    return a[i] - a[j], None


if STAGE == 'labels':
    os.makedirs(VD, exist_ok=True)
    for d in TR + VA + TE:
        z = C.load_day(d); ts, mid, seg = z['ts'], z['mid'], z['seg']
        f, ok = rv(ts, mid, seg, 60 * 10**9, True)
        P = np.stack([np.log1p(rv(ts, mid, seg, s * 10**9, False)[0]) for s in (10, 60, 300)], 1).astype(np.float32)
        np.savez(os.path.join(VD, f'{d}.npz'), rv=f.astype(np.float32), ok=ok, past=P, past60=rv(ts, mid, seg, 60 * 10**9, False)[0])
    r = np.concatenate([np.load(os.path.join(VD, f'{d}.npz'))['rv'][np.load(os.path.join(VD, f'{d}.npz'))['ok']] for d in TR])
    cuts = np.quantile(r, [1 / 3, 2 / 3]).tolist()
    log(dict(stage='labels', cuts=cuts, n_train=int(len(r))))
    sys.exit()

cuts = [json.loads(l) for l in open(RES) if '"labels"' in l][-1]['cuts']
cls = lambda x: np.digitize(x, cuts, right=True)  # <= c1 -> 0, <= c2 -> 1, else 2


def day(d):
    z = C.load_day(d); v = np.load(os.path.join(VD, f'{d}.npz'))
    return dict(F=np.concatenate([z['X'].astype(np.float32), v['past']], 1), y=cls(v['rv']).astype(np.int64), V=v['ok'],
                seg=z['seg'], cov=float(z['cov']), minute=z['minute'], pers=cls(v['past60']))


def scores(y, p):
    return dict(mF1=round(f1_score(y, p, average='macro'), 4), acc=round(accuracy_score(y, p), 4),
                recall=[round(x, 4) for x in recall_score(y, p, average=None)])


def flat(days, step):
    out = {}
    for d in days:
        x = day(d)
        if d not in TR and x['cov'] < C.MIN_COV: continue
        i = np.where(x['V'])[0][::step]
        out[d] = dict(F=x['F'][i], y=x['y'][i], m=x['minute'][i, None], pers=x['pers'][i])
    cat = lambda k: np.concatenate([out[d][k] for d in out])
    return {k: cat(k) for k in ('F', 'y', 'm', 'pers')}, np.concatenate([[int(d)] * len(out[d]['y']) for d in out])


def tabular(evals):
    from sklearn.ensemble import HistGradientBoostingClassifier as HGB
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    tr, _ = flat(TR, 10)
    print('train', len(tr['y']), 'class shares', np.bincount(tr['y']) / len(tr['y']), flush=True)
    models = dict(time_of_day=(HGB(max_iter=200, class_weight='balanced', random_state=0), 'm'),
                  logistic=(make_pipeline(StandardScaler(), LogisticRegression(max_iter=300, class_weight='balanced')), 'F'),
                  gbm=(HGB(max_iter=300, learning_rate=0.05, class_weight='balanced', random_state=0), 'F'))
    for k, (m, key) in models.items(): m.fit(tr[key], tr['y'])
    for name, days in evals:
        ev, dd = flat(days, 1)
        rec = dict(stage=name, n=int(len(ev['y'])), class_shares=[round(x, 4) for x in np.bincount(ev['y'], minlength=3) / len(ev['y'])],
                   persistence=scores(ev['y'], ev['pers']))
        for k, (m, key) in models.items(): rec[k] = scores(ev['y'], m.predict(ev[key]))
        log(rec)


if STAGE == 'base':
    tabular([('base_val', VA)])
elif STAGE == 'test_tab':
    tabular([('test_tab', TE)])
elif STAGE in ('gru', 'test_gru'):
    import torch, torch.nn as nn
    torch.set_num_threads(4); torch.manual_seed(0); rng = np.random.default_rng(0)
    T, EPOCHS, PER, PAT, BS = 50, 8, 200_000, 2, 256
    days = TR + VA + (TE if STAGE == 'test_gru' else [])
    data = {d: day(d) for d in days}
    samp = np.concatenate([data[d]['F'][np.where(data[d]['V'])[0][::50]] for d in TR]); mu, sd = samp.mean(0), samp.std(0) + 1e-6
    for d in data: data[d]['F'] = (data[d]['F'] - mu) / sd
    NF = samp.shape[1]; off = np.arange(-T + 1, 1)

    def vidx(d):
        x = data[d]; n = len(x['y']); ok = x['V'] & (np.arange(n) >= T - 1)
        ok[T - 1:] &= x['seg'][T - 1:] == x['seg'][:n - T + 1]; return np.where(ok)[0]

    class ConvGRU(nn.Module):
        def __init__(s):
            super().__init__()
            s.conv = nn.Sequential(nn.Conv1d(NF, 64, 3, padding=1), nn.GELU(), nn.Conv1d(64, 64, 3, padding=1), nn.GELU())
            s.gru = nn.GRU(64, 64, batch_first=True); s.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(64, 3))
        def forward(s, x): return s.head(s.gru(s.conv(x.transpose(1, 2)).transpose(1, 2))[0][:, -1])

    def batch(P): return torch.from_numpy(np.stack([data[d]['F'][i + off] for d, i in P])), torch.from_numpy(np.array([data[d]['y'][i] for d, i in P]))
    def predict(P):
        net.eval(); o = []
        with torch.no_grad():
            for k in range(0, len(P), 4096): o.append(torch.softmax(net(batch(P[k:k + 4096])[0]), 1).numpy())
        return np.concatenate(o)

    net = ConvGRU(); ck = os.path.join(OUT, 'vol_cls_gru.pt')
    idx = {d: vidx(d) for d in data}
    if STAGE == 'gru':
        tr = [(d, i) for d in TR for i in idx[d]]
        va = [(d, i) for d in VA if data[d]['cov'] >= C.MIN_COV for i in idx[d][::10]]
        yv = np.array([data[d]['y'][i] for d, i in va]); ytr = np.concatenate([data[d]['y'][idx[d]] for d in TR])
        lossf = nn.CrossEntropyLoss(weight=torch.tensor(len(ytr) / (3 * np.bincount(ytr, minlength=3)), dtype=torch.float32))
        opt = torch.optim.Adam(net.parameters(), 1e-3); best, bad, curve, t0 = -1, 0, [], time.time()
        for ep in range(EPOCHS):
            net.train(); sel = rng.choice(len(tr), PER, replace=False)
            for k in range(0, PER, BS):
                X, y = batch([tr[j] for j in sel[k:k + BS]]); opt.zero_grad(); lossf(net(X), y).backward(); opt.step()
            pv = predict(va).argmax(1); f = f1_score(yv, pv, average='macro')
            curve.append(dict(epoch=ep, val_mF1=round(f, 4), val_acc=round(accuracy_score(yv, pv), 4), min=round((time.time() - t0) / 60, 1)))
            print(curve[-1], flush=True)
            if f > best: best, bad = f, 0; torch.save(net.state_dict(), ck)
            else:
                bad += 1
                if bad >= PAT: break
        log(dict(stage='gru_val', best_val_mF1=round(best, 4), curve=curve))
    else:
        net.load_state_dict(torch.load(ck))
        P = [(d, i) for d in TE if data[d]['cov'] >= C.MIN_COV for i in idx[d]]
        prob = predict(P); y = np.array([data[d]['y'][i] for d, i in P]); dd = np.array([int(d) for d, _ in P]); p = prob.argmax(1)
        mn = np.array([data[d]['minute'][i] for d, i in P])
        np.savez(os.path.join(OUT, 'test_pred_vol_gru.npz'), prob=prob, y=y, day=dd, minute=mn)
        log(dict(stage='test_gru', n=int(len(y)), **scores(y, p), confusion=confusion_matrix(y, p).tolist(),
                 day_mF1={str(k): round(f1_score(y[dd == k], p[dd == k], average='macro'), 4) for k in np.unique(dd)}))
