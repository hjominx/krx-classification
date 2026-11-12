"""Experiment 4: probability-averaging ensembles of GBM, logistic regression and conv+GRU (5 s horizon).
Members are the models already chosen on VAL: GBM with the grid-selected settings and LR exactly as in final_eval.py
(both trained on TRAIN, sub=2), and the conv+GRU picked by tune_gru.py (results/tune_gru_<name>.pt).
All 7 equal-weight subsets are compared on VAL rows usable by every member; the best subset is scored once on TEST.
usage: python ensemble.py -> results/ensemble.jsonl
"""
import json, os, itertools
import numpy as np
import torch, torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score
import common as C
from select_models import stack, gbm, OUT

torch.set_num_threads(4)
H = 5; hi = C.HORIZONS.index(H)
TR, VA, TE = C.split_days()
g = max((json.loads(l) for l in open(os.path.join(OUT, 'select.jsonl')) if '"gbm_grid"' in l), key=lambda r: r['mF1'])
HP = dict(learning_rate=g['learning_rate'], max_leaf_nodes=g['max_leaf_nodes'], l2_regularization=g['l2'])
tg = max((json.loads(l) for l in open(os.path.join(OUT, 'tune_gru.jsonl'))), key=lambda r: r['best_val_mF1'])
T = tg['T']

# --- GBM and LR (same recipe as final_eval.py)
Xtr, ytr, _ = stack(TR, hi, sub=2)
gb = gbm(**HP).fit(Xtr, ytr)
mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
lr = LogisticRegression(max_iter=500, class_weight='balanced').fit(((Xtr - mu) / sd)[::2], ytr[::2])
del Xtr

# --- conv+GRU (same normalisation as tune_gru.py)
samp = np.concatenate([C.load_day(d)['X'][np.where(C.load_day(d)['V'][:, hi])[0][::50]] for d in TR]).astype(np.float32)
gmu, gsd = samp.mean(0), samp.std(0) + 1e-6; NF = samp.shape[1]


class ConvGRU(nn.Module):
    def __init__(s, hidden, dropout):
        super().__init__()
        s.conv = nn.Sequential(nn.Conv1d(NF, 64, 3, padding=1), nn.GELU(), nn.Conv1d(64, 64, 3, padding=1), nn.GELU())
        s.gru = nn.GRU(64, hidden, batch_first=True); s.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, 3))

    def forward(s, x):
        return s.head(s.gru(s.conv(x.transpose(1, 2)).transpose(1, 2))[0][:, -1])


net = ConvGRU(tg['hidden'], tg['dropout']); net.load_state_dict(torch.load(os.path.join(OUT, f"tune_gru_{tg['name']}.pt"))); net.eval()
off = np.arange(-T + 1, 1)


def block_probs(days):
    P = {'gbm': [], 'logreg': [], 'gru': []}; Y, D = [], []
    for d in days:
        z = C.load_day(d)
        if z['cov'] < C.MIN_COV: continue
        r = np.where(z['V'][:, hi])[0]
        r = r[(r >= T - 1)]; r = r[z['seg'][r] == z['seg'][r - T + 1]]
        X = z['X'][r]; F = ((z['X'] - gmu) / gsd).astype(np.float32)
        P['gbm'].append(gb.predict_proba(X)); P['logreg'].append(lr.predict_proba((X - mu) / sd))
        out = []
        with torch.no_grad():
            for k in range(0, len(r), 4096):
                out.append(torch.softmax(net(torch.from_numpy(np.stack([F[i + off] for i in r[k:k + 4096]]))), 1).numpy())
        P['gru'].append(np.concatenate(out)); Y.append(z['Y'][r, hi].astype(int)); D.append(np.full(len(r), int(d)))
        print(d, len(r), flush=True)
    return {k: np.concatenate(v) for k, v in P.items()}, np.concatenate(Y), np.concatenate(D)


def score(P, y, members):
    p = np.mean([P[m] for m in members], 0).argmax(1)
    return dict(mF1=round(f1_score(y, p, average='macro'), 4), acc=round(accuracy_score(y, p), 4))


subsets = [list(c) for k in (1, 2, 3) for c in itertools.combinations(['gbm', 'logreg', 'gru'], k)]
Pv, yv, _ = block_probs(VA)
val = [dict(members=s, **score(Pv, yv, s)) for s in subsets]
best = max((v for v in val if len(v['members']) > 1), key=lambda v: v['mF1'])
Pt, yt, dt = block_probs(TE)
test = [dict(members=s, **score(Pt, yt, s)) for s in (['gbm'], ['logreg'], ['gru'], best['members'])]
pb = np.mean([Pt[m] for m in best['members']], 0)
np.savez(os.path.join(OUT, 'test_pred_ensemble_H5.npz'), prob=pb, y=yt, day=dt)
res = dict(H=H, gru=tg['name'], n_val=int(len(yv)), n_test=int(len(yt)), val=val, chosen=best['members'], test=test,
           test_per_day_mF1={str(k): round(f1_score(yt[dt == k], pb[dt == k].argmax(1), average='macro'), 4) for k in np.unique(dt)})
print(json.dumps(res))
with open(os.path.join(OUT, 'ensemble.jsonl'), 'a') as f: f.write(json.dumps(res) + '\n')
