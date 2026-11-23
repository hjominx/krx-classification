"""Final TEST evaluation for the non-deep models. GBM settings are read from the VAL selection results
(best val macro-F1 in the gbm_grid at H=5), never from TEST. Trains on TRAIN only (same data as the deep models).
usage: python final_eval.py -> results/direction/final.jsonl, results/direction/test_pred_gbm_H*.npz
"""
import _paths  # noqa: F401  (src/common, src/direction, src/volatility, src/report on the import path)
import json, os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
import common as C
from select_models import stack, gbm, OUT

if __name__ == '__main__':
    TR, VA, TE = C.split_days()
    sel = [json.loads(l) for l in open(os.path.join(OUT, 'select.jsonl'))]
    grid = [r for r in sel if r['exp'] == 'gbm_grid']
    best = max(grid, key=lambda r: r['mF1'])
    hp = dict(learning_rate=best['learning_rate'], max_leaf_nodes=best['max_leaf_nodes'], l2_regularization=best['l2'])
    print('GBM settings chosen on VAL:', hp, 'val mF1', best['mF1'])
    names = list(C.load_day(TR[0])['names'])
    for hi, H in enumerate(C.HORIZONS):
        Xtr, ytr, _ = stack(TR, hi, sub=2); Xte, yte, dte = stack(TE, hi, eval_only=True)
        te_days = sorted(set(dte.tolist()))
        mn = np.concatenate([C.load_day(str(d))['minute'][C.load_day(str(d))['V'][:, hi]] for d in te_days])
        res = dict(H=H, test_days=te_days, n_test=int(len(yte)), test_dist=np.round(np.bincount(yte, minlength=3) / len(yte), 3).tolist())
        maj = np.bincount(ytr, minlength=3).argmax(); pm = np.full_like(yte, maj)
        res['majority'] = dict(mF1=round(f1_score(yte, pm, average='macro'), 4), acc=round(accuracy_score(yte, pm), 4))
        r20 = Xte[:, names.index('ret20')]; pmo = np.where(r20 > 0, 2, np.where(r20 < 0, 0, 1))
        res['momentum'] = dict(mF1=round(f1_score(yte, pmo, average='macro'), 4), acc=round(accuracy_score(yte, pmo), 4))
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
        lr = LogisticRegression(max_iter=500, class_weight='balanced').fit(((Xtr - mu) / sd)[::2], ytr[::2])
        pl = lr.predict((Xte - mu) / sd)
        res['logreg'] = dict(mF1=round(f1_score(yte, pl, average='macro'), 4), acc=round(accuracy_score(yte, pl), 4))
        m = gbm(**hp).fit(Xtr, ytr); prob = m.predict_proba(Xte); p = prob.argmax(1)
        res['gbm'] = dict(mF1=round(f1_score(yte, p, average='macro'), 4), acc=round(accuracy_score(yte, p), 4), iters=int(m.n_iter_),
                          per_day_mF1={str(k): round(f1_score(yte[dte == k], p[dte == k], average='macro'), 4) for k in te_days},
                          per_day_acc={str(k): round(accuracy_score(yte[dte == k], p[dte == k]), 4) for k in te_days},
                          per_day_majority_acc={str(k): round(accuracy_score(yte[dte == k], pm[dte == k]), 4) for k in te_days},
                          confusion=confusion_matrix(yte, p).tolist())
        np.savez(os.path.join(OUT, f'test_pred_gbm_H{H}.npz'), prob=prob, y=yte, day=dte, minute=mn)
        print(json.dumps(res), flush=True)
        with open(os.path.join(OUT, 'final.jsonl'), 'a') as f: f.write(json.dumps(res) + '\n')
