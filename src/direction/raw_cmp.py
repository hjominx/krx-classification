"""Does deep learning beat simple models when given the RAW book instead of hand-made features? (H = 5 s, 3 classes)
Design fixed before running:
  raw input : the 40 book columns only (10 ask / 10 bid distances to mid in ticks, log sizes) - no imbalance, OFI, flow, time
  samples   : identical for every model: valid 5 s labels with a full 50-snapshot window in the same segment
  models    : logistic_raw, gbm_raw (current snapshot), gbm_raw_lags (raw at lags 0,1,2,5,10,20,49 = 280 inputs,
              same 50-snapshot history the GRU sees), gru_raw (conv+GRU tuned base: T=50, h64, dropout .2, balanced,
              8 x 200k, patience 2; checkpoint = best VAL epoch)
  tabular models train on every 10th TRAIN sample; all use balanced class weights
  report    : VAL and TEST macro-F1 for every model (TEST looked at once, nothing chosen on it);
              day-block bootstrap (2,000 resamples of the 10 TEST days) of mF1(gru_raw) - mF1(each other model)
usage: python raw_cmp.py -> results/direction/raw_cmp.jsonl, results/direction/test_pred_raw_cmp.npz
"""
import _paths  # noqa: F401  (src/common, src/direction, src/volatility, src/report on the import path)
import json, os, time
import numpy as np, torch, torch.nn as nn
from sklearn.metrics import f1_score, accuracy_score
from sklearn.ensemble import HistGradientBoostingClassifier as HGB
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import common as C

torch.set_num_threads(4); torch.manual_seed(0); rng = np.random.default_rng(0)
OUT = C.RES_DIR; RES = os.path.join(OUT, 'raw_cmp.jsonl')
log = lambda r: (print(json.dumps(r), flush=True), open(RES, 'a').write(json.dumps(r) + '\n'))
H = 5; hi = C.HORIZONS.index(H); T = 50; LAGS = (0, 1, 2, 5, 10, 20, 49); off = np.arange(-T + 1, 1)
TR, VA, TE = C.split_days()
data = {}
for d in TR + VA + TE:
    z = C.load_day(d)
    if d not in TR and float(z['cov']) < C.MIN_COV: continue
    data[d] = dict(F=z['X'][:, :40].astype(np.float32), y=z['Y'][:, hi].astype(np.int64), V=z['V'][:, hi], seg=z['seg'])
samp = np.concatenate([data[d]['F'][::50] for d in TR]); mu, sd = samp.mean(0), samp.std(0) + 1e-6
for d in data: data[d]['F'] = (data[d]['F'] - mu) / sd


def vidx(d):
    x = data[d]; n = len(x['y']); ok = x['V'] & (np.arange(n) >= T - 1)
    ok[T - 1:] &= x['seg'][T - 1:] == x['seg'][:n - T + 1]; return np.where(ok)[0]
idx = {d: vidx(d) for d in data}


def tab(days, step):
    F0, FL, y, dd = [], [], [], []
    for d in days:
        i = idx[d][::step]; F = data[d]['F']
        F0.append(F[i]); FL.append(np.concatenate([F[i - l] for l in LAGS], 1)); y.append(data[d]['y'][i]); dd.append(np.full(len(i), int(d)))
    return np.concatenate(F0), np.concatenate(FL), np.concatenate(y), np.concatenate(dd)


t0 = time.time()
F0, FL, ytr, _ = tab(TR, 10)
tabm = {'logistic_raw': (make_pipeline(StandardScaler(), LogisticRegression(max_iter=300, class_weight='balanced')), 0),
        'gbm_raw': (HGB(max_iter=300, learning_rate=0.05, class_weight='balanced', random_state=0), 0),
        'gbm_raw_lags': (HGB(max_iter=300, learning_rate=0.05, class_weight='balanced', random_state=0), 1)}
for k, (m, lag) in tabm.items(): m.fit(FL if lag else F0, ytr); print(k, 'fit', round(time.time() - t0), flush=True)
del F0, FL


class ConvGRU(nn.Module):
    def __init__(s):
        super().__init__()
        s.conv = nn.Sequential(nn.Conv1d(40, 64, 3, padding=1), nn.GELU(), nn.Conv1d(64, 64, 3, padding=1), nn.GELU())
        s.gru = nn.GRU(64, 64, batch_first=True); s.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(64, 3))
    def forward(s, x): return s.head(s.gru(s.conv(x.transpose(1, 2)).transpose(1, 2))[0][:, -1])


def batch(P): return torch.from_numpy(np.stack([data[d]['F'][i + off] for d, i in P])), torch.from_numpy(np.array([data[d]['y'][i] for d, i in P]))
def predict(P):
    net.eval(); o = []
    with torch.no_grad():
        for k in range(0, len(P), 4096): o.append(torch.softmax(net(batch(P[k:k + 4096])[0]), 1).numpy())
    return np.concatenate(o)


net = ConvGRU(); tr = [(d, i) for d in TR for i in idx[d]]; va = [(d, i) for d in VA for i in idx[d][::10]]
yv = np.array([data[d]['y'][i] for d, i in va]); yall = np.concatenate([data[d]['y'][idx[d]] for d in TR])
lossf = nn.CrossEntropyLoss(weight=torch.tensor(len(yall) / (3 * np.bincount(yall, minlength=3)), dtype=torch.float32))
opt = torch.optim.Adam(net.parameters(), 1e-3); best, bad, curve, ck = -1, 0, [], os.path.join(OUT, 'raw_cmp_gru.pt')
for ep in range(8):
    net.train(); sel = rng.choice(len(tr), 200_000, replace=False)
    for k in range(0, 200_000, 256):
        X, y = batch([tr[j] for j in sel[k:k + 256]]); opt.zero_grad(); lossf(net(X), y).backward(); opt.step()
    f = f1_score(yv, predict(va).argmax(1), average='macro'); curve.append(dict(epoch=ep, val_mF1=round(f, 4))); print(curve[-1], flush=True)
    if f > best: best, bad = f, 0; torch.save(net.state_dict(), ck)
    else:
        bad += 1
        if bad >= 2: break
net.load_state_dict(torch.load(ck))
log(dict(stage='gru_curve', curve=curve, n_params=sum(p.numel() for p in net.parameters())))

preds = {}
for split, days in (('val', VA), ('test', TE)):
    F0, FL, y, dd = tab(days, 1)
    P = {k: m.predict(FL if lag else F0) for k, (m, lag) in tabm.items()}
    P['gru_raw'] = predict([(d, i) for d in days for i in idx[d]]).argmax(1)
    rec = dict(stage=split, n=int(len(y)), **{k: dict(mF1=round(f1_score(y, p, average='macro'), 4), acc=round(accuracy_score(y, p), 4)) for k, p in P.items()})
    if split == 'test':
        days_u = np.unique(dd); mask = {k: dd == k for k in days_u}; br = np.random.default_rng(1); diffs = {k: [] for k in tabm}
        for _ in range(2000):
            s = br.choice(days_u, len(days_u)); m = np.concatenate([np.where(mask[k])[0] for k in s])
            g = f1_score(y[m], P['gru_raw'][m], average='macro')
            for k in tabm: diffs[k].append(g - f1_score(y[m], P[k][m], average='macro'))
        rec['boot_gru_minus'] = {k: dict(mean=round(float(np.mean(v)), 4), ci95=[round(float(np.quantile(v, .025)), 4), round(float(np.quantile(v, .975)), 4)]) for k, v in diffs.items()}
        np.savez(os.path.join(OUT, 'test_pred_raw_cmp.npz'), y=y, day=dd, **P)
    log(rec)
