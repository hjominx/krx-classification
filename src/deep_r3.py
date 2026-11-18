"""Round 3, 5 s direction: more seeds of the round-2 deep members and new training variants of them.
Everything here is chosen on VAL; TEST is read only by final_r3.py. Design fixed before running.
Same data, inputs, budget and checkpoint rules as deep_v2.py (T = 50, 12 epochs x 200,000 windows, batch 256,
VAL macro-F1 on every 10th VAL window after each epoch, best epoch kept, patience 3). Results go to separate files so
the round-2 files (v2_deep.jsonl, read by ens_v2.py and final_v2.py) stay as they were.
Configs
  base, mt, tcn  : the round-2 configs of deep_v2.py, run here with new seeds (5-9) to enlarge the bags
  mt_st2         : mt with input stride 2 - the 50 snapshots cover twice the time (about 20 s instead of 10 s)
  mt_rec         : mt with recency-weighted sampling of training windows (weight exp(1.1 * position of the day in
                   TRAIN), so the last TRAIN day is drawn 3 times as often as the first)
  mt_cos         : mt with lr 2e-3 decayed by a cosine over the 12 epochs instead of a constant 1e-3
  mt_big         : mt with 400,000 windows per epoch, batch 512
Stage R1: the four variants with seeds 0, 1 (compare with mt seeds 0, 1 of deep_v2.py, mean full-VAL 0.5129).
Stage R2: variants whose seed-0/1 mean beats mt's get seeds 2-4 (bag of 5, as in round 2).
Stage S : base, mt, tcn seeds 5-9.
Each run saves its probabilities on every valid VAL window -> results/r3_val_<cfg>_s<seed>.npz (prob, y, day, row)
and its weights -> results/r3_deep_<cfg>_s<seed>.pt.
usage: python deep_r3.py R1|R2|S|S:<cfg>|<cfg>:<seed>,... [threads]   -> results/r3_deep.jsonl
"""
import json, math, os, sys, time
import numpy as np, torch, torch.nn as nn
from sklearn.metrics import f1_score, accuracy_score
import exp_v2 as E

argv = sys.argv; sys.argv = argv[:1]
import deep_v2 as D
sys.argv = argv
torch.set_num_threads(int(sys.argv[2]) if len(sys.argv) > 2 else 2)
OUT = E.OUT; RES = os.path.join(OUT, 'r3_deep.jsonl')
T, PAT, EPOCHS = D.T, D.PAT, D.EPOCHS; HS, hi = D.HS, D.hi
VAR = dict(mt_st2=dict(base='mt', stride=2), mt_rec=dict(base='mt', rec=1.1), mt_cos=dict(base='mt', lr=2e-3, cos=True),
           mt_big=dict(base='mt', per=400_000, bs=512))
for n in ('base', 'mt', 'tcn'): VAR.setdefault(n, dict(base=n))
DEF = dict(stride=1, rec=0.0, lr=1e-3, cos=False, per=D.PER, bs=D.BS)


def spec(name):
    v = dict(DEF, **VAR[name])
    if D.SMOKE: v['per'] = D.PER
    return dict(D.CONFIGS[v['base']], **v)


def run(name, seed):
    cfg = spec(name); cols = D.COLS[cfg['cols']]; tag = f'{name}_s{seed}'; s_ = cfg['stride']; span = s_ * (T - 1)
    torch.manual_seed(seed); rng = np.random.default_rng(seed); off = np.arange(-span, 1, s_)
    F, Y, V, day, row, tr, va_all, mu, sd = D.get_data(cfg['cols'])
    keep = lambda idx: idx[(idx >= span)] if s_ == 1 else idx[(idx >= span) & (SEG[np.maximum(idx - span, 0)] == SEG[idx])]
    tr, va_all = keep(tr), keep(va_all); va = va_all[::10]
    net = D.make(cfg, len(cols)); nh = len(HS) if cfg['mt'] else 1; hs = list(range(len(HS))) if cfg['mt'] else [hi]
    W = [torch.tensor(len(tr) / (3 * np.bincount(Y[tr][V[tr, h], h], minlength=3)) * V[tr, h].mean(), dtype=torch.float32) for h in hs]
    W[hs.index(hi)] = torch.tensor(len(tr) / (3 * np.bincount(Y[tr, hi], minlength=3)), dtype=torch.float32)
    lossf = [nn.CrossEntropyLoss(weight=w, label_smoothing=cfg['ls'], ignore_index=-100) for w in W]
    opt = torch.optim.AdamW(net.parameters(), cfg['lr'], weight_decay=cfg['wd']) if cfg['wd'] else torch.optim.Adam(net.parameters(), cfg['lr'])
    ema = {k: v.detach().clone() for k, v in net.state_dict().items()} if cfg['ema'] else None
    if cfg['rec']:                                         # sampling weight by the position of the day in TRAIN
        dd = np.unique(day[tr]); pos = np.searchsorted(dd, day[tr]) / max(len(dd) - 1, 1); wt = np.exp(cfg['rec'] * pos)

    def draw(n):
        if not cfg['rec']: return rng.choice(tr, min(n, len(tr)), replace=False)
        key = np.log(rng.random(len(tr))) / wt             # Efraimidis-Spirakis weighted sampling without replacement
        return tr[np.argpartition(-key, n)[:n]]

    def probs(m, idx):
        m.eval(); o = []
        with torch.no_grad():
            for k in range(0, len(idx), 4096):
                z = m(torch.from_numpy(F[idx[k:k + 4096, None] + off])); o.append(torch.softmax(z.view(len(z), nh, 3)[:, hs.index(hi)], 1).numpy())
        return np.concatenate(o)

    shadow = D.make(cfg, len(cols)) if cfg['ema'] else None
    yv = Y[va, hi]; best, bad, curve, t0, step = -1, 0, [], time.time(), 0
    for ep in range(EPOCHS):
        if cfg['cos']:
            for g in opt.param_groups: g['lr'] = cfg['lr'] * 0.5 * (1 + math.cos(math.pi * ep / EPOCHS))
        net.train(); sel = draw(cfg['per']); rng.shuffle(sel)
        for k in range(0, len(sel), cfg['bs']):
            b = sel[k:k + cfg['bs']]; out = net(torch.from_numpy(F[b[:, None] + off])).view(len(b), nh, 3); loss = 0.0
            for j, h in enumerate(hs):
                l = lossf[j](out[:, j], torch.from_numpy(np.where(V[b, h], Y[b, h], -100)))
                loss = loss + (l if h == hi else 0.5 * l / max(len(hs) - 1, 1))
            opt.zero_grad(); loss.backward(); opt.step()
            if ema is not None:
                with torch.no_grad():
                    step += 1; dec = min(0.999, (1 + step) / (10 + step))
                    for kk, v in net.state_dict().items(): ema[kk].mul_(dec).add_(v, alpha=1 - dec)
        rec = dict(epoch=ep, raw=round(float(f1_score(yv, probs(net, va).argmax(1), average='macro')), 4))
        cand = [(rec['raw'], {k: v.clone() for k, v in net.state_dict().items()}, 'raw')]
        if ema is not None:
            shadow.load_state_dict(ema); rec['ema'] = round(float(f1_score(yv, probs(shadow, va).argmax(1), average='macro')), 4)
            cand.append((rec['ema'], {k: v.clone() for k, v in ema.items()}, 'ema'))
        f, st, which = max(cand, key=lambda c: c[0]); rec['min'] = round((time.time() - t0) / 60, 1); curve.append(rec); print(tag, rec, flush=True)
        if f > best: best, bad, best_state, best_which, best_ep = f, 0, st, which, ep
        else:
            bad += 1
            if bad >= PAT: break
    net.load_state_dict(best_state); prob = probs(net, va_all); y = Y[va_all, hi]; p = prob.argmax(1); m, fm, _ = E.best_m(prob, y)
    rec = dict(name=name, seed=seed, **{k: v for k, v in cfg.items()}, n_params=sum(q.numel() for q in net.parameters()), best_epoch=best_ep,
               best_weights=best_which, val10_mF1=round(float(best), 4), val_mF1=round(E.mf1(y, p), 4), val_acc=round(float(accuracy_score(y, p)), 4),
               n_val=int(len(y)), best_m=m, mF1_at_best_m=fm, curve=curve, sec=round(time.time() - t0))
    np.savez_compressed(os.path.join(OUT, f'r3_val_{tag}.npz'), prob=prob.astype(np.float32), y=y, day=day[va_all], row=row[va_all])
    torch.save(dict(state=best_state, mu=mu, sd=sd, cfg=cfg, cols=cols, epochs=best_ep + 1), os.path.join(OUT, f'r3_deep_{tag}.pt'))
    with open(RES, 'a') as fh: fh.write(json.dumps(rec) + '\n')
    print(json.dumps({k: v for k, v in rec.items() if k != 'curve'}), flush=True)


def recs(): return [json.loads(l) for l in open(RES)] if os.path.exists(RES) else []


SEG = None
if __name__ == '__main__':
    what = sys.argv[1]; have = {(r['name'], r['seed']) for r in recs()}
    variants = ['mt_st2', 'mt_rec', 'mt_cos', 'mt_big']
    if what == 'R1': todo = [(n, s) for s in (0, 1) for n in variants]
    elif what == 'R2':
        ref = np.mean([r['val_mF1'] for r in D.recs() if r['name'] == 'mt' and r['seed'] in (0, 1)])
        mean = {n: np.mean([r['val_mF1'] for r in recs() if r['name'] == n and r['seed'] in (0, 1)]) for n in variants}
        top = [n for n in variants if mean[n] > ref]; print('R2 configs', top, 'ref', round(float(ref), 4), {k: round(float(v), 4) for k, v in mean.items()}, flush=True)
        todo = [(n, s) for n in top for s in (2, 3, 4)]
    elif what.startswith('S'):
        names = what[2:].split(',') if what.startswith('S:') else ['mt', 'tcn', 'base']
        todo = [(n, s) for n in names for s in range(5, 10)]
    else: todo = [(t.split(':')[0], int(t.split(':')[1])) for t in what.split(',')]
    for n, s in todo:
        if (n, s) in have: continue
        if spec(n)['stride'] > 1 and SEG is None:          # segment id of every row, for windows longer than T rows
            F0, Y0, V0, gseg0, *_ = D.load(E.TR + E.VA, D.COLS['full'][:1]); SEG = gseg0; del F0, Y0, V0
        run(n, s)
