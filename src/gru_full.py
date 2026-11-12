"""conv+GRU trained to convergence on ALL training snapshots (5 s). Design fixed before running:
  configs   : full_h64 (the tuned base: T=50, hidden 64, dropout 0.2, balanced weights, seed 0)
              full_h128 (same, hidden 128) - more data may support more capacity
  training  : every epoch = every valid TRAIN snapshot once, shuffled (~4 M); Adam lr 1e-3, halved when VAL macro-F1
              does not improve; up to 20 epochs, stop after 4 epochs without improvement; checkpoint = best VAL epoch
              (VAL scored on every 5th valid snapshot)
  final     : the config with the best VAL macro-F1 is scored once on TEST (all valid snapshots); its probabilities
              are saved. Also reported for it, without further choice: the flat-calibration m chosen by gru_flat.py's rule
              on this model's VAL probabilities.
usage: python gru_full.py -> results/gru_full.jsonl, results/gru_full_<name>.pt, results/test_pred_gru_full_H5.npz
"""
import sys, json, os, time
import numpy as np, torch, torch.nn as nn
from sklearn.metrics import f1_score, accuracy_score, recall_score
sys.argv = ['tune_gru.py', 'none']
import tune_gru as G
import common as C

RES = os.path.join(G.OUT, 'gru_full.jsonl'); MAXEP, PAT, BS = 20, 4, 256
CONFIGS = [('full_h64', dict(G.BASE)), ('full_h128', dict(G.BASE, hidden=128))]
log = lambda r: (print(json.dumps(r), flush=True), open(RES, 'a').write(json.dumps(r) + '\n'))
done = {json.loads(l)['name'] for l in open(RES) if '"config"' in l} if os.path.exists(RES) else set()


def predict(net, P, off, data):
    net.eval(); out = []
    with torch.no_grad():
        for k in range(0, len(P), 4096):
            out.append(torch.softmax(net(torch.from_numpy(np.stack([data[d]['F'][i + off] for d, i in P[k:k + 4096]]))), 1).numpy())
    return np.concatenate(out)


for name, cfg in CONFIGS:
    if name in done: continue
    T = cfg['T']; off = np.arange(-T + 1, 1); torch.manual_seed(cfg['seed']); rng = np.random.default_rng(cfg['seed'])
    idx = {d: G.valid_idx(d, T) for d in G.data}
    tr = [(d, i) for d in G.TR for i in idx[d]]
    va = [(d, i) for d in G.VA if G.data[d]['cov'] >= C.MIN_COV for i in idx[d][::5]]
    yv = np.array([G.data[d]['y'][i] for d, i in va])
    ytr = np.concatenate([G.data[d]['y'][idx[d]] for d in G.TR])
    w = torch.tensor(len(ytr) / (3 * np.bincount(ytr, minlength=3)), dtype=torch.float32)
    net = G.ConvGRU(cfg['hidden'], cfg['dropout']); opt = torch.optim.Adam(net.parameters(), cfg['lr'])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=0.5, patience=0)
    lossf = nn.CrossEntropyLoss(weight=w); best, bad, curve, t0 = -1, 0, [], time.time()
    for ep in range(MAXEP):
        net.train(); perm = rng.permutation(len(tr)); tot = 0.0
        for k in range(0, len(perm), BS):
            P = [tr[j] for j in perm[k:k + BS]]
            X = torch.from_numpy(np.stack([G.data[d]['F'][i + off] for d, i in P])); y = torch.from_numpy(np.array([G.data[d]['y'][i] for d, i in P]))
            opt.zero_grad(); loss = lossf(net(X), y); loss.backward(); opt.step(); tot += loss.item() * len(P)
        pv = predict(net, va, off, G.data).argmax(1); f = f1_score(yv, pv, average='macro')
        curve.append(dict(epoch=ep, train_loss=round(tot / len(perm), 4), val_mF1=round(f, 4), val_acc=round(accuracy_score(yv, pv), 4),
                          lr=opt.param_groups[0]['lr'], min=round((time.time() - t0) / 60, 1)))
        print(name, curve[-1], flush=True); sched.step(f)
        if f > best: best, bad = f, 0; torch.save(net.state_dict(), os.path.join(G.OUT, f'gru_full_{name}.pt'))
        else:
            bad += 1
            if bad >= PAT: break
    log(dict(name=name, config=cfg, n_train=len(tr), best_val_mF1=round(best, 4), best_epoch=int(np.argmax([c['val_mF1'] for c in curve])),
             curve=curve, n_params=sum(p.numel() for p in net.parameters())))

recs = [json.loads(l) for l in open(RES) if '"config"' in l]
b = max(recs, key=lambda r: r['best_val_mF1']); cfg = b['config']; T = cfg['T']; off = np.arange(-T + 1, 1)
net = G.ConvGRU(cfg['hidden'], cfg['dropout']); net.load_state_dict(torch.load(os.path.join(G.OUT, f"gru_full_{b['name']}.pt")))
# VAL probabilities on all valid snapshots -> flat calibration rule of gru_flat.py
Pv = [(d, i) for d in G.VA if G.data[d]['cov'] >= C.MIN_COV for i in G.valid_idx(d, T)]
pv = predict(net, Pv, off, G.data); yv = np.array([G.data[d]['y'][i] for d, i in Pv])
MS = (1, 1.25, 1.5, 1.75, 2, 2.5, 3, 4)
vs = {m: (f1_score(yv, (pv * [1, m, 1]).argmax(1), average='macro'), accuracy_score(yv, (pv * [1, m, 1]).argmax(1))) for m in MS}
m_best = max((m for m in MS if vs[m][0] >= vs[1][0] - 0.01), key=lambda m: vs[m][1])
# TEST data (loaded only now), normalised with the TRAIN statistics of tune_gru
te = {}
for d in G.TE:
    z = C.load_day(d)
    if float(z['cov']) < C.MIN_COV: continue
    te[d] = dict(F=((z['X'].astype(np.float32) - G.mu) / G.sd), y=z['Y'][:, G.hi].astype(np.int64), V=z['V'][:, G.hi], seg=z['seg'], cov=float(z['cov']))
    G.data[d] = te[d]
P = [(d, i) for d in te for i in G.valid_idx(d, T)]
prob = predict(net, P, off, G.data); y = np.array([G.data[d]['y'][i] for d, i in P]); dd = np.array([int(d) for d, _ in P])
np.savez(os.path.join(G.OUT, 'test_pred_gru_full_H5.npz'), prob=prob, y=y, day=dd)
out = dict(stage='final', chosen=b['name'], val_mF1=b['best_val_mF1'], flat_m=m_best)
for m in sorted({1, m_best}):
    p = (prob * [1, m, 1]).argmax(1)
    out[f'test_m{m}'] = dict(mF1=round(f1_score(y, p, average='macro'), 4), acc=round(accuracy_score(y, p), 4),
                            recall=[round(x, 4) for x in recall_score(y, p, average=None)],
                            day_mF1_range=[round(min(f1_score(y[dd == k], p[dd == k], average='macro') for k in np.unique(dd)), 4),
                                           round(max(f1_score(y[dd == k], p[dd == k], average='macro') for k in np.unique(dd)), 4)])
log(out)
