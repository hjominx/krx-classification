"""Round 2, 5 s direction: the pipeline chosen in ens_v2.py scored on TEST. The only round-2 script that reads TEST.
Design fixed before running.
  primary   : every member as trained on TRAIN - deep members are the VAL-selected checkpoints; the GBM and the
              logistic model are refitted with the chosen settings (deterministic)
  secondary : every member re-trained on TRAIN + VAL with frozen settings - GBM: same number of trees; deep: same
              seed, as many epochs as its VAL-selected checkpoint had, raw or EMA weights as selected on VAL
  reference : the earlier best model (tuned conv+GRU of tune_gru.py, results/direction/test_pred_gru_tuned_H5.npz) on the same
              rows; day-block bootstrap (2,000 resamples of the test days) of the macro-F1 difference
usage: python final_v2.py primary|secondary [threads]  -> results/direction/v2_final.jsonl, results/direction/v2_test_<variant>.npz
"""
import _paths  # noqa: F401  (src/common, src/direction, src/volatility, src/report on the import path)
import json, os, sys, time
import numpy as np, torch, torch.nn as nn
from sklearn.ensemble import HistGradientBoostingClassifier as HGB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import common as C
import exp_v2 as E
import deep_v2 as D

OUT = E.OUT; RES = os.path.join(OUT, 'v2_final.jsonl'); mf1 = E.mf1; hi = D.hi; T = D.T; off = np.arange(-T + 1, 1)
KEY = lambda day, row: day.astype(np.int64) * 1_000_000 + row


def log(rec):
    print(json.dumps(rec)[:600], flush=True)
    with open(RES, 'a') as f: f.write(json.dumps(rec) + '\n')


def deep_prob(net, cfg, F, idx):
    nh = len(D.HS) if cfg['mt'] else 1; j = hi if cfg['mt'] else 0; net.eval(); o = []
    with torch.no_grad():
        for k in range(0, len(idx), 4096):
            z = net(torch.from_numpy(F[idx[k:k + 4096, None] + off])); o.append(torch.softmax(z.view(len(z), nh, 3)[:, j], 1).numpy())
    return np.concatenate(o)


def train_fixed(name, seed, epochs, which, F, Y, V, tr):
    """the training loop of deep_v2.run for a fixed number of epochs, without a validation set"""
    cfg = D.CONFIGS[name]; torch.manual_seed(seed); rng = np.random.default_rng(seed)
    net = D.make(cfg, F.shape[1]); nh = len(D.HS) if cfg['mt'] else 1; hs = list(range(len(D.HS))) if cfg['mt'] else [hi]
    W = [torch.tensor(len(tr) / (3 * np.bincount(Y[tr][V[tr, h], h], minlength=3)) * V[tr, h].mean(), dtype=torch.float32) for h in hs]
    W[hs.index(hi)] = torch.tensor(len(tr) / (3 * np.bincount(Y[tr, hi], minlength=3)), dtype=torch.float32)
    lossf = [nn.CrossEntropyLoss(weight=w, label_smoothing=cfg['ls'], ignore_index=-100) for w in W]
    opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=cfg['wd']) if cfg['wd'] else torch.optim.Adam(net.parameters(), 1e-3)
    ema = {k: v.detach().clone() for k, v in net.state_dict().items()}; step = 0
    for ep in range(epochs):
        net.train(); sel = rng.choice(tr, min(D.PER, len(tr)), replace=False)
        for k in range(0, len(sel), D.BS):
            b = sel[k:k + D.BS]; out = net(torch.from_numpy(F[b[:, None] + off])).view(len(b), nh, 3); loss = 0.0
            for j, h in enumerate(hs):
                l = lossf[j](out[:, j], torch.from_numpy(np.where(V[b, h], Y[b, h], -100)))
                loss = loss + (l if h == hi else 0.5 * l / max(len(hs) - 1, 1))
            opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                step += 1; dec = min(0.999, (1 + step) / (10 + step))
                for kk, v in net.state_dict().items(): ema[kk].mul_(dec).add_(v, alpha=1 - dec)
    if which == 'ema': net.load_state_dict(ema)
    return net


def boot(y, d, p_new, p_ref, n=2000, seed=1):
    days = np.unique(d); ix = {k: np.where(d == k)[0] for k in days}; rng = np.random.default_rng(seed); out = []
    for _ in range(n):
        m = np.concatenate([ix[k] for k in rng.choice(days, len(days))]); out.append(mf1(y[m], p_new[m]) - mf1(y[m], p_ref[m]))
    return dict(mean=round(float(np.mean(out)), 4), ci95=[round(float(np.quantile(out, .025)), 4), round(float(np.quantile(out, .975)), 4)],
                share_positive=round(float(np.mean(np.array(out) > 0)), 3))


if __name__ == '__main__':
    variant = sys.argv[1]; torch.set_num_threads(int(sys.argv[2]) if len(sys.argv) > 2 else 4); t0 = time.time()
    w = [json.loads(l) for l in open(os.path.join(OUT, 'v2_ens.jsonl')) if '"winner"' in l][-1]
    drec = {(r['name'], r['seed']): r for r in D.recs()}; tr_days = E.TR if variant == 'primary' else E.TR + E.VA
    te_days = [d for d in E.TE if float(C.load_day(d)['cov']) >= C.MIN_COV]
    P = {}; sname, _, cols = E.chosen_set()
    # ---- tabular members
    Xte, yte, dte, rte = E.stack(te_days); kte = KEY(dte, rte)
    if {'gbm', 'logreg'} & set(w['members']):
        Xtr, ytr, _, _ = E.stack(tr_days, sub=2)
        if 'gbm' in w['members']:
            if w['gbm_kind'] == 'multiclass':
                c = max(E.recs('C'), key=lambda r: r['mF1']); kw = {k: c[k] for k in ('learning_rate', 'max_leaf_nodes', 'min_samples_leaf', 'max_features') if k in c}
                P['gbm'] = E.gbm(max_iter=c['best_iter'], **kw).fit(Xtr[:, cols], ytr).predict_proba(Xte[:, cols])
            else:
                t = E.recs('T')[0]; k = ytr != 1; kwb = dict(E.HP0, max_iter=t['best_iter'], class_weight='balanced', early_stopping=False, random_state=0)
                p1 = HGB(**kwb).fit(Xtr[:, cols], (ytr != 1).astype(int)).predict_proba(Xte[:, cols])
                p2 = HGB(**kwb).fit(Xtr[k][:, cols], (ytr[k] == 2).astype(int)).predict_proba(Xte[:, cols])
                pr = np.column_stack([p1[:, 1] * p2[:, 0] * 4 / 3, p1[:, 0] * 2 / 3, p1[:, 1] * p2[:, 1] * 4 / 3]); P['gbm'] = pr / pr.sum(1, keepdims=True)
        if 'logreg' in w['members']:
            mu, sd = Xtr[:, cols].mean(0), Xtr[:, cols].std(0) + 1e-6
            P['logreg'] = LogisticRegression(max_iter=500, class_weight='balanced').fit(((Xtr[:, cols] - mu) / sd)[::2], ytr[::2]).predict_proba((Xte[:, cols] - mu) / sd)
        del Xtr
    print('tabular members done', round(time.time() - t0), flush=True)
    # ---- deep members
    bags = [n for n in w['members'] if n not in ('gbm', 'logreg')]; kd = None
    for key in sorted({D.CONFIGS[n]['cols'] for n in bags}):
        cc = D.COLS[key]; days = (te_days if variant == 'primary' else tr_days + te_days)
        F, Y, V, gseg, day, row, cov = D.load(days, cc); ok = D.windows(V, gseg); is_te = np.isin(day, [int(d) for d in te_days])
        ev = np.where(ok & is_te)[0]; tr = np.where(ok & ~is_te)[0]
        if variant == 'primary':
            ck0 = torch.load(os.path.join(OUT, f"v2_deep_{[n for n in bags if D.CONFIGS[n]['cols'] == key][0]}_s0.pt"), weights_only=False); mu, sd = ck0['mu'], ck0['sd']
        else:
            s_ = F[tr[::50]]; mu, sd = s_.mean(0), s_.std(0) + 1e-6
        np.subtract(F, mu, out=F); np.divide(F, sd, out=F); np.clip(F, -10, 10, out=F)
        kd = KEY(day[ev], row[ev]); yd = Y[ev, hi]; dd = day[ev]
        for n in [n for n in bags if D.CONFIGS[n]['cols'] == key]:
            cfg = D.CONFIGS[n]; probs = []
            for seed in sorted(s for (nn_, s) in drec if nn_ == n):
                if variant == 'primary':
                    ck = torch.load(os.path.join(OUT, f'v2_deep_{n}_s{seed}.pt'), weights_only=False); assert np.allclose(ck['mu'], mu)
                    net = D.make(cfg, len(cc)); net.load_state_dict(ck['state'])
                else:
                    r = drec[(n, seed)]; net = train_fixed(n, seed, r['best_epoch'] + 1, r['best_weights'], F, Y, V, tr)
                probs.append(deep_prob(net, cfg, F, ev)); print(n, seed, 'done', round(time.time() - t0), flush=True)
            P[n] = np.mean(probs, 0)
        del F
    # ---- common rows, ensemble, scores
    if kd is None: kd, yd, dd = kte, yte, dte
    ix = np.searchsorted(kte, kd); assert (kte[ix] == kd).all() and (yte[ix] == yd).all()
    for n in ('gbm', 'logreg'):
        if n in P: P[n] = P[n][ix]
    prob = np.mean([P[n] for n in w['members']], 0); p = (prob * [1, w['m'], 1]).argmax(1); y, d = yd, dd
    rec = dict(stage='test', variant=variant, pipeline=w['name'], m=w['m'], val_mF1=w['val_mF1'], n=int(len(y)), mF1=round(mf1(y, p), 4),
               acc=round(float(accuracy_score(y, p)), 4), mF1_m1=round(mf1(y, prob.argmax(1)), 4),
               recall=[round(float(x), 4) for x in confusion_matrix(y, p, normalize='true').diagonal()],
               day_mF1={str(k): round(mf1(y[d == k], p[d == k]), 4) for k in np.unique(d)}, confusion=confusion_matrix(y, p).tolist(),
               members={n: round(mf1(y, P[n].argmax(1)), 4) for n in w['members']})
    ref = os.path.join(OUT, 'test_pred_gru_tuned_H5.npz')
    if os.path.exists(ref):
        z = np.load(ref)
        if len(z['y']) == len(y) and (z['y'] == y).all():
            pr = z['prob'].argmax(1); rec.update(ref_tuned_gru_mF1=round(mf1(y, pr), 4), boot_new_minus_ref=boot(y, d, p, pr),
                                                 days_better=int(sum(mf1(y[d == k], p[d == k]) > mf1(y[d == k], pr[d == k]) for k in np.unique(d))))
        else: rec['ref_note'] = 'reference rows do not match'
    rec['sec'] = round(time.time() - t0); log(rec)
    np.savez_compressed(os.path.join(OUT, f'v2_test_{variant}.npz'), prob=prob.astype(np.float32), y=y, day=d, row=kd % 1_000_000)
