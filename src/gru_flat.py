"""Make the tuned conv+GRU (tune_gru_base_seed1.pt, 5 s) answer 'flat' more often, without retraining:
multiply the flat probability by m before argmax. Rule fixed before running: m from {1, 1.25, 1.5, 1.75, 2, 2.5, 3, 4};
choose the m with the highest VAL accuracy among those whose VAL macro-F1 is at least (VAL macro-F1 at m=1) - 0.01.
TEST is then scored once with that m (TEST probabilities from test_pred_gru_tuned_H5.npz, same checkpoint).
usage: python gru_flat.py -> results/gru_flat.jsonl
"""
import sys, json, os
import numpy as np, torch
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
sys.argv = ['tune_gru.py', 'none']
import tune_gru as G
import common as C

MS = (1, 1.25, 1.5, 1.75, 2, 2.5, 3, 4)
cfg = dict(G.BASE, seed=1); T = cfg['T']; off = np.arange(-T + 1, 1)
net = G.ConvGRU(cfg['hidden'], cfg['dropout']); net.load_state_dict(torch.load(os.path.join(G.OUT, 'tune_gru_base_seed1.pt'))); net.eval()
P = [(d, i) for d in G.VA if G.data[d]['cov'] >= C.MIN_COV for i in G.valid_idx(d, T)]
out = []
with torch.no_grad():
    for k in range(0, len(P), 4096):
        out.append(torch.softmax(net(torch.from_numpy(np.stack([G.data[d]['F'][i + off] for d, i in P[k:k + 4096]]))), 1).numpy())
pv = np.concatenate(out); yv = np.array([G.data[d]['y'][i] for d, i in P])
np.savez(os.path.join(G.OUT, 'val_pred_gru_tuned_H5.npz'), prob=pv, y=yv)
zt = np.load(os.path.join(G.OUT, 'test_pred_gru_tuned_H5.npz')); pt, yt = zt['prob'], zt['y']


def score(prob, y, m):
    p = (prob * np.array([1, m, 1])).argmax(1)
    return dict(mF1=round(f1_score(y, p, average='macro'), 4), acc=round(accuracy_score(y, p), 4),
                recall=[round(x, 4) for x in recall_score(y, p, average=None)], precision=[round(x, 4) for x in precision_score(y, p, average=None)],
                pred_share=[round(float((p == c).mean()), 4) for c in range(3)])


res = RES = os.path.join(G.OUT, 'gru_flat.jsonl')
val = {m: score(pv, yv, m) for m in MS}
for m in MS:
    r = dict(stage='val', m=m, **val[m]); print(json.dumps(r))
    open(RES, 'a').write(json.dumps(r) + '\n')
floor = val[1]['mF1'] - 0.01
m_best = max((m for m in MS if val[m]['mF1'] >= floor), key=lambda m: val[m]['acc'])
for m in sorted({1, m_best}):
    r = dict(stage='test', m=m, chosen=bool(m == m_best), **score(pt, yt, m),
             always_flat=dict(acc=round(float((yt == 1).mean()), 4))); print(json.dumps(r))
    open(RES, 'a').write(json.dumps(r) + '\n')
