"""Deep models on KRX 000660. Train on TRAIN, pick the epoch by VAL macro-F1, then score TEST once.
  deeplob : DeepLOB (Zhang, Zohren & Roberts 2019) on the raw 10-level book, input (T=100, 40)
  gru     : conv + GRU on the 71 engineered per-event features, input (T=50, 71)
usage: python deep.py deeplob|gru H [epochs] [train_samples_per_epoch] [threads]
"""
import _paths  # noqa: F401  (src/common, src/direction, src/volatility, src/report on the import path)
import json, os, sys, time
import numpy as np
import torch, torch.nn as nn
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
import common as C

MODEL, H = sys.argv[1], int(sys.argv[2])
EPOCHS = int(sys.argv[3]) if len(sys.argv) > 3 else 8
PER_EPOCH = int(sys.argv[4]) if len(sys.argv) > 4 else 100_000
torch.set_num_threads(int(sys.argv[5]) if len(sys.argv) > 5 else 4)
torch.manual_seed(0); rng = np.random.default_rng(0)
T = 100 if MODEL == 'deeplob' else 50
KEY = 'lob' if MODEL == 'deeplob' else 'X'
hi = C.HORIZONS.index(H)
OUT = C.RES_DIR; os.makedirs(OUT, exist_ok=True)

TR, VA, TE = C.split_days()
data = {}
for d in TR + VA + TE:
    z = C.load_day(d)
    n = len(z['ts']); t = np.arange(n)
    ok = z['V'][:, hi] & (t >= T - 1)
    ok[T - 1:] &= z['seg'][T - 1:] == z['seg'][:n - T + 1]
    data[d] = dict(F=z[KEY].astype(np.float32), y=z['Y'][:, hi].astype(np.int64), idx=np.where(ok)[0],
                   cov=float(z['cov']), minute=z['minute'])
samp = np.concatenate([data[d]['F'][data[d]['idx'][::50]] for d in TR])
mu, sd = samp.mean(0), samp.std(0) + 1e-6
for d in data: data[d]['F'] = (data[d]['F'] - mu) / sd
NF = samp.shape[1]
pairs = lambda days, sub=1: [(d, i) for d in days for i in data[d]['idx'][::sub]]
tr_pairs = pairs(TR); va_pairs = pairs([d for d in VA if data[d]['cov'] >= C.MIN_COV], 10)
te_days = [d for d in TE if data[d]['cov'] >= C.MIN_COV]
off = np.arange(-T + 1, 1)


def batch(P):
    X = np.stack([data[d]['F'][i + off] for d, i in P]); y = np.array([data[d]['y'][i] for d, i in P])
    return torch.from_numpy(X), torch.from_numpy(y)


class DeepLOB(nn.Module):
    def __init__(s):
        super().__init__()
        def blk(first_act):
            return nn.Sequential(nn.Conv2d(32 if first_act != 'in' else 1, 32, (1, 2), stride=(1, 2)), nn.LeakyReLU(0.01), nn.BatchNorm2d(32),
                                 nn.Conv2d(32, 32, (4, 1), padding='same'), nn.LeakyReLU(0.01), nn.BatchNorm2d(32),
                                 nn.Conv2d(32, 32, (4, 1), padding='same'), nn.LeakyReLU(0.01), nn.BatchNorm2d(32))
        s.b1, s.b2 = blk('in'), blk('mid')
        s.b3 = nn.Sequential(nn.Conv2d(32, 32, (1, 10)), nn.LeakyReLU(0.01), nn.BatchNorm2d(32),
                             nn.Conv2d(32, 32, (4, 1), padding='same'), nn.LeakyReLU(0.01), nn.BatchNorm2d(32),
                             nn.Conv2d(32, 32, (4, 1), padding='same'), nn.LeakyReLU(0.01), nn.BatchNorm2d(32))
        inc = lambda k: nn.Sequential(nn.Conv2d(32, 64, 1), nn.LeakyReLU(0.01), nn.BatchNorm2d(64),
                                      nn.Conv2d(64, 64, (k, 1), padding='same'), nn.LeakyReLU(0.01), nn.BatchNorm2d(64))
        s.i1, s.i2 = inc(3), inc(5)
        s.i3 = nn.Sequential(nn.MaxPool2d((3, 1), stride=1, padding=(1, 0)), nn.Conv2d(32, 64, 1), nn.LeakyReLU(0.01), nn.BatchNorm2d(64))
        s.lstm = nn.LSTM(192, 64, batch_first=True); s.fc = nn.Linear(64, 3)

    def forward(s, x):                          # x: (B, T, 40)
        h = s.b3(s.b2(s.b1(x.unsqueeze(1))))   # (B, 32, T, 1)
        h = torch.cat([s.i1(h), s.i2(h), s.i3(h)], 1).squeeze(3).transpose(1, 2)   # (B, T, 192)
        return s.fc(s.lstm(h)[0][:, -1])


class ConvGRU(nn.Module):
    def __init__(s):
        super().__init__()
        s.conv = nn.Sequential(nn.Conv1d(NF, 64, 3, padding=1), nn.GELU(), nn.Conv1d(64, 64, 3, padding=1), nn.GELU())
        s.gru = nn.GRU(64, 64, batch_first=True); s.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(64, 3))

    def forward(s, x):
        return s.head(s.gru(s.conv(x.transpose(1, 2)).transpose(1, 2))[0][:, -1])


net = DeepLOB() if MODEL == 'deeplob' else ConvGRU()
opt = torch.optim.Adam(net.parameters(), 1e-3)
ytr_all = np.concatenate([data[d]['y'][data[d]['idx']] for d in TR])
w = torch.tensor(len(ytr_all) / (3 * np.bincount(ytr_all, minlength=3)), dtype=torch.float32)
lossf = nn.CrossEntropyLoss(weight=w)


def predict(P):
    net.eval(); out = []
    with torch.no_grad():
        for k in range(0, len(P), 2048): out.append(torch.softmax(net(batch(P[k:k + 2048])[0]), 1).numpy())
    return np.concatenate(out)


tag = f'{MODEL}_H{H}'
best, state, t0, bad = -1, None, time.time(), 0
for ep in range(EPOCHS):
    net.train(); sel = rng.choice(len(tr_pairs), min(PER_EPOCH, len(tr_pairs)), replace=False)
    tot = 0.0
    for k in range(0, len(sel), 256):
        X, y = batch([tr_pairs[i] for i in sel[k:k + 256]])
        opt.zero_grad(); loss = lossf(net(X), y); loss.backward(); opt.step(); tot += loss.item() * len(y)
    pv = predict(va_pairs).argmax(1); yv = np.array([data[d]['y'][i] for d, i in va_pairs])
    f = f1_score(yv, pv, average='macro')
    rec = dict(model=MODEL, H=H, epoch=ep, train_loss=round(tot / len(sel), 4), val_mF1=round(f, 4),
               val_acc=round(accuracy_score(yv, pv), 4), sec=round(time.time() - t0))
    print(json.dumps(rec), flush=True)
    with open(os.path.join(OUT, f'deep_{tag}_epochs.jsonl'), 'a') as fh: fh.write(json.dumps(rec) + '\n')
    if f > best: best, state, bad = f, {k: v.clone() for k, v in net.state_dict().items()}, 0
    else:
        bad += 1
        if bad >= 2: break
net.load_state_dict(state)
torch.save(state, os.path.join(OUT, f'deep_{tag}.pt'))
# final TEST evaluation (once), keep per-sample probabilities for plots
P = pairs(te_days); prob = predict(P)
y = np.array([data[d]['y'][i] for d, i in P]); dd = np.array([int(d) for d, _ in P]); mn = np.array([data[d]['minute'][i] for d, i in P])
np.savez(os.path.join(OUT, f'test_pred_{tag}.npz'), prob=prob, y=y, day=dd, minute=mn)
p = prob.argmax(1)
res = dict(model=MODEL, H=H, best_val_mF1=round(best, 4), test_mF1=round(f1_score(y, p, average='macro'), 4),
           test_acc=round(accuracy_score(y, p), 4), test_days=te_days, n_test=int(len(y)),
           per_day_mF1={str(k): round(f1_score(y[dd == k], p[dd == k], average='macro'), 4) for k in np.unique(dd)},
           confusion=confusion_matrix(y, p).tolist())
print(json.dumps(res))
with open(os.path.join(OUT, 'deep_test.jsonl'), 'a') as fh: fh.write(json.dumps(res) + '\n')
