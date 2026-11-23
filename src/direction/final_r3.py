"""Round 3, 5 s direction: the pipeline chosen on VAL by post_r3.py (pool 'main') scored on TEST, once.
The only round-3 script that reads TEST. Design fixed before running.
  members   : as trained on TRAIN - deep members are the VAL-selected checkpoints of deep_v2.py / deep_r3.py, the
              round-2 GBM and logistic model are refitted with their chosen settings (deterministic, as final_v2.py),
              the round-3 GBMs are the saved models at their VAL-chosen number of trees
  combining : exactly as chosen on VAL (plain mean / class multipliers / smoothing, or the stacking model refitted on
              all VAL rows with the same settings as in post_r3.py)
  reference : the round-2 pipeline on TEST (results/direction/v2_test_primary.npz, flat multiplier 1.1) on the same rows;
              day-block bootstrap (2,000 resamples of the test days) of the macro-F1 difference
Note: TEST was already used by round 1, round 2 and lob_models.py, so it is no longer an untouched hold-out.
usage: python final_r3.py [threads]  -> results/direction/r3_final.jsonl, results/direction/r3_test.npz
"""
import _paths  # noqa: F401  (src/common, src/direction, src/volatility, src/report on the import path)
import json, os, pickle, sys, time
import numpy as np, torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import common as C
import exp_v2 as E

argv = sys.argv; sys.argv = argv[:1] + ['main']
import deep_v2 as D
import post_r3 as PR
import gbm_r3 as GR
sys.argv = argv
torch.set_num_threads(int(sys.argv[1]) if len(sys.argv) > 1 else 4)
OUT = E.OUT; RES = os.path.join(OUT, 'r3_final.jsonl'); mf1 = PR.mf1; hi = D.hi
KEY = lambda day, row: day.astype(np.int64) * 1_000_000 + row.astype(np.int64)


def log(rec):
    print(json.dumps(rec)[:600], flush=True)
    with open(RES, 'a') as f: f.write(json.dumps(rec) + '\n')


def boot(y, d, p_new, p_ref, n=2000, seed=1):
    days = np.unique(d); ix = {k: np.where(d == k)[0] for k in days}; rng = np.random.default_rng(seed); out = []
    for _ in range(n):
        m = np.concatenate([ix[k] for k in rng.choice(days, len(days))]); out.append(mf1(y[m], p_new[m]) - mf1(y[m], p_ref[m]))
    return dict(mean=round(float(np.mean(out)), 4), ci95=[round(float(np.quantile(out, .025)), 4), round(float(np.quantile(out, .975)), 4)],
                share_positive=round(float(np.mean(np.array(out) > 0)), 3))


def deep_member(name, te_days):
    """mean TEST probability of all seeds of one deep bag, with the keys of its rows"""
    paths = PR.pool_files()[name]; probs, keys = [], None; cache = {}
    for p in paths:
        b = os.path.basename(p); s = int(b.split('_s')[-1][:-4])
        ck = os.path.join(OUT, f'v2_deep_{name}_s{s}.pt' if b.startswith('v2_') else f'r3_deep_{name}_s{s}.pt'); ck = torch.load(ck, weights_only=False)
        cfg = ck['cfg']; st = cfg.get('stride', 1); span = st * (D.T - 1); off = np.arange(-span, 1, st)
        if 'F' not in cache:
            F, Y, V, gseg, day, row, cov = D.load(te_days, ck['cols']); F = (F - ck['mu']) / ck['sd']; np.clip(F, -10, 10, out=F)
            cache.update(F=F.astype(np.float32), V=V, gseg=gseg, day=day, row=row, mu=ck['mu'])
        assert np.allclose(cache['mu'], ck['mu'])
        F, V, gseg = cache['F'], cache['V'], cache['gseg']
        ok = V[:, hi].copy(); ok[:span] = False; ok[span:] &= gseg[span:] == gseg[:-span]; ev = np.where(ok)[0]
        net = D.make(cfg, len(ck['cols'])); net.load_state_dict(ck['state']); net.eval(); nh = len(D.HS) if cfg['mt'] else 1; j = hi if cfg['mt'] else 0; o = []
        with torch.no_grad():
            for k in range(0, len(ev), 4096):
                z = net(torch.from_numpy(F[ev[k:k + 4096, None] + off])); o.append(torch.softmax(z.view(len(z), nh, 3)[:, j], 1).numpy())
        k_ = KEY(cache['day'][ev], cache['row'][ev])
        if keys is None: keys = k_
        else: assert (keys == k_).all()
        probs.append(np.concatenate(o)); print(name, s, 'done', flush=True)
    return np.mean(probs, 0), keys


if __name__ == '__main__':
    t0 = time.time()
    w = [json.loads(l) for l in open(os.path.join(OUT, 'r3_post.jsonl')) if '"winner"' in l and '"pool": "main"' in l][-1]
    te_days = [d for d in E.TE if float(C.load_day(d)['cov']) >= C.MIN_COV]; P, K = {}, {}
    for n in w['members']:
        if n in ('base', 'mt', 'tcn') or n.startswith('mt_'): P[n], K[n] = deep_member(n, te_days)
        elif n in ('gbm', 'logreg'):
            if 'tab' not in K:
                Xtr, ytr, _, _ = E.stack(E.TR, sub=2); Xte, yte, dte, rte = E.stack(te_days); K['tab'] = KEY(dte, rte); _, _, cols = E.chosen_set()
            if n == 'gbm':
                c = max(E.recs('C'), key=lambda r: r['mF1']); kw = {k: c[k] for k in ('learning_rate', 'max_leaf_nodes', 'min_samples_leaf', 'max_features') if k in c}
                P[n] = E.gbm(max_iter=c['best_iter'], **kw).fit(Xtr[:, cols], ytr).predict_proba(Xte[:, cols])
            else:
                mu, sd = Xtr[:, cols].mean(0), Xtr[:, cols].std(0) + 1e-6
                P[n] = LogisticRegression(max_iter=500, class_weight='balanced').fit(((Xtr[:, cols] - mu) / sd)[::2], ytr[::2]).predict_proba((Xte[:, cols] - mu) / sd)
            K[n] = K['tab']
        elif n.startswith('gbm_'):
            g = pickle.load(open(os.path.join(OUT, f'r3_{n}.pkl'), 'rb')); X, yy, dd, rr = GR.stack(te_days, g['lag'])
            P[n] = next(pr for it, pr in enumerate(g['model'].staged_predict_proba(X), 1) if it == g['best_iter']); K[n] = KEY(dd, rr)
        print('member', n, 'done', round(time.time() - t0), flush=True)
    # common rows of all members and the reference
    ref = np.load(os.path.join(OUT, 'v2_test_primary.npz')); kref = KEY(ref['day'], ref['row'])
    keys = kref
    for n in w['members']: keys = np.intersect1d(keys, K[n])
    on = lambda pr, k: pr[np.searchsorted(k, keys)]
    for n in w['members']: assert (K[n][np.searchsorted(K[n], keys)] == keys).all()
    Pm = {n: on(P[n], K[n]) for n in w['members']}; y = ref['y'][np.searchsorted(kref, keys)]; d = keys // 1_000_000
    pref = (on(ref['prob'], kref) * [1, 1.1, 1]).argmax(1)
    if w['method'] == 'stack':
        Pv, kv = PR.load_pool(PR.pool_files()); yv = Pv.pop('y'); names = sorted(Pv); assert names == sorted(w['members'])
        lv = np.log(np.clip(np.hstack([Pv[n] for n in names]), 1e-6, 1)); lt = np.log(np.clip(np.hstack([Pm[n] for n in names]), 1e-6, 1))
        if w['ctx']: lv = np.hstack([lv, PR.context(kv)[0]]); lt = np.hstack([lt, PR.context(keys)[0]])
        sm = PR.stack_model(w['stack_kind'], w.get('class_weight', 'balanced'))
        if w['stack_kind'] == 'lr':
            mu, sd = lv.mean(0), lv.std(0) + 1e-6
            prob = sm.fit(((lv - mu) / sd)[::3], yv[::3]).predict_proba((lt - mu) / sd)
        else:
            prob = sm.fit(lv[::3], yv[::3]).predict_proba(lt)
    else:
        prob = np.mean([Pm[n] for n in w['members']], 0)
        if w['method'] == 'smooth': prob = PR.smooth(prob, keys, w['alpha'])
    p = (prob * w['mult']).argmax(1)
    rec = dict(stage='test', pipeline=w['name'], method=w['method'], members=w['members'], mult=w['mult'], val_mF1=w['mF1'], n=int(len(y)),
               mF1=round(mf1(y, p), 4), acc=round(float(accuracy_score(y, p)), 4), mF1_m1=round(mf1(y, prob.argmax(1)), 4),
               recall=[round(float(x), 4) for x in confusion_matrix(y, p, normalize='true').diagonal()],
               member_mF1={n: round(mf1(y, Pm[n].argmax(1)), 4) for n in w['members']},
               r2_pipeline_mF1=round(mf1(y, pref), 4), boot_new_minus_r2=boot(y, d, p, pref),
               days_better=int(sum(mf1(y[d == k], p[d == k]) > mf1(y[d == k], pref[d == k]) for k in np.unique(d))),
               day_mF1={str(k): round(mf1(y[d == k], p[d == k]), 4) for k in np.unique(d)}, confusion=confusion_matrix(y, p).tolist(),
               sec=round(time.time() - t0))
    log(rec); np.savez_compressed(os.path.join(OUT, 'r3_test.npz'), prob=prob.astype(np.float32), y=y, day=d, row=keys % 1_000_000)
