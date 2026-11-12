"""conv+GRU hyper-parameter study on the VAL block (5 s horizon). TEST is not touched in 'tune' mode.
One factor at a time from the base setting (T=50, hidden=64, lr=1e-3, dropout=0.2, balanced class weights);
the base is repeated with 3 seeds to show run-to-run noise. Resumable: configs already in the results file are skipped.
usage: python tune_gru.py tune   -> results/tune_gru.jsonl, results/tune_gru_<name>.pt
       python tune_gru.py final  -> best config by val macro-F1, scored once on TEST -> results/tune_gru_final.jsonl
"""
import json, os, sys, time
import numpy as np
import torch, torch.nn as nn
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
import common as C

torch.set_num_threads(4)
MODE = sys.argv[1]
H = 5; hi = C.HORIZONS.index(H)
OUT = os.path.join(C.ROOT, 'results'); RES = os.path.join(OUT, 'tune_gru.jsonl')
EPOCHS, PER_EPOCH, PATIENCE, BS = 8, 200_000, 2, 256
BASE = dict(T=50, hidden=64, lr=1e-3, dropout=0.2, cw='balanced', seed=0)
CONFIGS = [('base', {}), ('base_seed1', dict(seed=1)), ('base_seed2', dict(seed=2)),
           ('T10', dict(T=10)), ('T25', dict(T=25)), ('T100', dict(T=100)),
           ('hidden32', dict(hidden=32)), ('hidden128', dict(hidden=128)),
           ('lr3e-4', dict(lr=3e-4)), ('lr3e-3', dict(lr=3e-3)),
           ('dropout0', dict(dropout=0.0)), ('dropout0.5', dict(dropout=0.5)),
           ('no_class_weight', dict(cw='none'))]

TR, VA, TE = C.split_days()
days = TR + VA + (TE if MODE == 'final' else [])
data = {}
for d in days:
    z = C.load_day(d)
    data[d] = dict(F=z['X'].astype(np.float32), y=z['Y'][:, hi].astype(np.int64), V=z['V'][:, hi], seg=z['seg'],
                   cov=float(z['cov']), minute=z['minute'])
samp = np.concatenate([data[d]['F'][np.where(data[d]['V'])[0][::50]] for d in TR])
mu, sd = samp.mean(0), samp.std(0) + 1e-6
for d in data: data[d]['F'] = (data[d]['F'] - mu) / sd
NF = samp.shape[1]


def valid_idx(d, T):
    x = data[d]; n = len(x['y']); t = np.arange(n)
    ok = x['V'] & (t >= T - 1)
    ok[T - 1:] &= x['seg'][T - 1:] == x['seg'][:n - T + 1]
    return np.where(ok)[0]


class ConvGRU(nn.Module):
    def __init__(s, hidden, dropout):
        super().__init__()
        s.conv = nn.Sequential(nn.Conv1d(NF, 64, 3, padding=1), nn.GELU(), nn.Conv1d(64, 64, 3, padding=1), nn.GELU())
        s.gru = nn.GRU(64, hidden, batch_first=True); s.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, 3))

    def forward(s, x):
        return s.head(s.gru(s.conv(x.transpose(1, 2)).transpose(1, 2))[0][:, -1])


def run(name, cfg):
    T = cfg['T']; torch.manual_seed(cfg['seed']); rng = np.random.default_rng(cfg['seed']); off = np.arange(-T + 1, 1)
    idx = {d: valid_idx(d, T) for d in data}
    tr = [(d, i) for d in TR for i in idx[d]]
    va = [(d, i) for d in VA if data[d]['cov'] >= C.MIN_COV for i in idx[d][::10]]

    def batch(P):
        return (torch.from_numpy(np.stack([data[d]['F'][i + off] for d, i in P])),
                torch.from_numpy(np.array([data[d]['y'][i] for d, i in P])))

    net = ConvGRU(cfg['hidden'], cfg['dropout']); opt = torch.optim.Adam(net.parameters(), cfg['lr'])
    ytr = np.concatenate([data[d]['y'][idx[d]] for d in TR])
    w = torch.tensor(len(ytr) / (3 * np.bincount(ytr, minlength=3)), dtype=torch.float32) if cfg['cw'] == 'balanced' else None
    lossf = nn.CrossEntropyLoss(weight=w)

    def predict(P):
        net.eval(); out = []
        with torch.no_grad():
            for k in range(0, len(P), 2048): out.append(torch.softmax(net(batch(P[k:k + 2048])[0]), 1).numpy())
        return np.concatenate(out)

    yv = np.array([data[d]['y'][i] for d, i in va])
    best, state, bad, curve, t0 = -1, None, 0, [], time.time()
    for ep in range(EPOCHS):
        net.train(); sel = rng.choice(len(tr), min(PER_EPOCH, len(tr)), replace=False)
        for k in range(0, len(sel), BS):
            X, y = batch([tr[i] for i in sel[k:k + BS]]); opt.zero_grad(); lossf(net(X), y).backward(); opt.step()
        pv = predict(va).argmax(1); f = f1_score(yv, pv, average='macro')
        curve.append(dict(epoch=ep, val_mF1=round(f, 4), val_acc=round(accuracy_score(yv, pv), 4)))
        print(name, curve[-1], round(time.time() - t0), flush=True)
        if f > best: best, state, bad, best_acc = f, {k: v.clone() for k, v in net.state_dict().items()}, 0, curve[-1]['val_acc']
        else:
            bad += 1
            if bad >= PATIENCE: break
    torch.save(state, os.path.join(OUT, f'tune_gru_{name}.pt'))
    return dict(name=name, **cfg, best_val_mF1=round(best, 4), best_val_acc=best_acc,
                best_epoch=int(np.argmax([c['val_mF1'] for c in curve])), n_params=sum(p.numel() for p in net.parameters()),
                curve=curve, sec=round(time.time() - t0)), net, idx, off


if MODE == 'tune':
    done = {json.loads(l)['name'] for l in open(RES)} if os.path.exists(RES) else set()
    for name, ch in CONFIGS:
        if name in done: continue
        rec = run(name, {**BASE, **ch})[0]
        with open(RES, 'a') as f: f.write(json.dumps(rec) + '\n')
elif MODE == 'final':
    recs = [json.loads(l) for l in open(RES)]
    best = max(recs, key=lambda r: r['best_val_mF1'])
    cfg = {k: best[k] for k in BASE}; T = cfg['T']; off = np.arange(-T + 1, 1)
    net = ConvGRU(cfg['hidden'], cfg['dropout']); net.load_state_dict(torch.load(os.path.join(OUT, f"tune_gru_{best['name']}.pt"))); net.eval()
    te = [d for d in TE if data[d]['cov'] >= C.MIN_COV]
    P = [(d, i) for d in te for i in valid_idx(d, T)]
    out = []
    with torch.no_grad():
        for k in range(0, len(P), 2048):
            out.append(torch.softmax(net(torch.from_numpy(np.stack([data[d]['F'][i + off] for d, i in P[k:k + 2048]]))), 1).numpy())
    prob = np.concatenate(out); p = prob.argmax(1)
    y = np.array([data[d]['y'][i] for d, i in P]); dd = np.array([int(d) for d, _ in P]); mn = np.array([data[d]['minute'][i] for d, i in P])
    np.savez(os.path.join(OUT, 'test_pred_gru_tuned_H5.npz'), prob=prob, y=y, day=dd, minute=mn)
    res = dict(chosen=best['name'], config=cfg, best_val_mF1=best['best_val_mF1'], test_mF1=round(f1_score(y, p, average='macro'), 4),
               test_acc=round(accuracy_score(y, p), 4), n_test=int(len(y)),
               per_day_mF1={str(k): round(f1_score(y[dd == k], p[dd == k], average='macro'), 4) for k in np.unique(dd)},
               confusion=confusion_matrix(y, p).tolist())
    print(json.dumps(res))
    with open(os.path.join(OUT, 'tune_gru_final.jsonl'), 'a') as f: f.write(json.dumps(res) + '\n')
