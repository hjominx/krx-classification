"""Improvement round 2 for the 5 s direction task: deep sequence models on the v2 features.
Everything here is chosen on VAL; TEST is only read by final_v2.py. Design fixed before running.
Data   : days with a trade file (TRAIN 46, VAL 12); windows of T = 50 snapshots inside one segment; inputs standardised
         with TRAIN statistics and clipped to +-10.
Budget : up to 12 epochs x 200,000 random training windows, batch 256, lr 1e-3; VAL macro-F1 after every epoch on every
         10th valid VAL window; checkpoint = best epoch; stop after 3 epochs without improvement.
Configs (input set 'full' = 127 columns of exp_v2.py unless stated):
  base_ref : conv+GRU (hidden 64, dropout 0.2), input set refc77  -> the earlier model on the earlier information
  base     : the same model on 'full'                              -> effect of the new features
  ema      : base + exponential moving average of the weights (decay 0.999 after warm-up); the better of raw / EMA
             weights is kept
  reg      : ema + dropout 0.3 + AdamW weight decay 1e-4 + label smoothing 0.05
  mt       : ema + auxiliary heads for the 1/2/10/30 s labels (loss = 5 s + 0.5 x mean of the others)
  h128     : ema + hidden 128 + dropout 0.3
  tcn      : dilated residual 1-D conv net (dilations 1,2,4,8), EMA
  tfm      : conv stem + 2-layer Transformer encoder (d 64, 4 heads), EMA
Stage D1: every config with seeds 0, 1.  Stage D2: the 3 configs with the best mean full-VAL macro-F1 get seeds 2, 3, 4.
Each run saves its probabilities on every valid VAL window -> results/direction/v2_val_deep_<cfg>_s<seed>.npz (prob, y, day, row).
usage: python deep_v2.py D1|D2|<cfg>:<seed> [threads]   -> results/direction/v2_deep.jsonl
The shared parts (models, budget, loader) are in src/common/seq_models.py.
"""
import _paths  # noqa: F401  (src/common, src/direction, src/volatility, src/report on the import path)
import json, os, sys, time
import numpy as np, torch, torch.nn as nn
from sklearn.metrics import f1_score, accuracy_score
import exp_v2 as E
from seq_models import SMOKE, T, BS, EPOCHS, PER, PAT, HS, hi, ConvGRU, TCN, Tfm, make, load, windows  # noqa: F401  (also used as D.<name>)

torch.set_num_threads(int(sys.argv[2]) if len(sys.argv) > 2 else 2)
OUT = E.OUT; RES = os.path.join(OUT, 'v2_deep.jsonl'); PFX = 'v2'      # PFX: file-name prefix (width_r8.py sets 'r8')
BASE = dict(arch='convgru', cols='full', hidden=64, dropout=0.2, ema=False, wd=0.0, ls=0.0, mt=False)
CONFIGS = dict(base_ref=dict(BASE, cols='refc77'), base=dict(BASE), ema=dict(BASE, ema=True),
               reg=dict(BASE, ema=True, dropout=0.3, wd=1e-4, ls=0.05), mt=dict(BASE, ema=True, mt=True),
               h128=dict(BASE, ema=True, hidden=128, dropout=0.3), tcn=dict(BASE, arch='tcn', ema=True),
               tfm=dict(BASE, arch='tfm', ema=True, dropout=0.1))
COLS = dict(full=E.FULL, refc77=E.REFC77)


DATA = {}


def get_data(key):
    """standardised matrix and window indices for one column set (cached; only one set is kept in memory)"""
    if key not in DATA:
        DATA.clear(); F, Y, V, gseg, day, row, cov = load(E.TR + E.VA, COLS[key])
        is_tr = day <= int(E.TR[-1]); ok = windows(V, gseg)
        tr = np.where(ok & is_tr)[0]; va_all = np.where(ok & ~is_tr & cov)[0]
        if SMOKE: va_all = va_all[::200]
        s = F[tr[::50]]; mu, sd = s.mean(0), s.std(0) + 1e-6
        np.subtract(F, mu, out=F); np.divide(F, sd, out=F); np.clip(F, -10, 10, out=F)
        DATA[key] = (F, Y, V, day, row, tr, va_all, mu, sd)
    return DATA[key]


def run(name, seed, save=True):
    cfg = CONFIGS[name]; cols = COLS[cfg['cols']]; tag = f'{name}_s{seed}'
    torch.manual_seed(seed); rng = np.random.default_rng(seed); off = np.arange(-T + 1, 1)
    F, Y, V, day, row, tr, va_all, mu, sd = get_data(cfg['cols']); va = va_all[::10]
    net = make(cfg, len(cols)); nh = len(HS) if cfg['mt'] else 1; hs = list(range(len(HS))) if cfg['mt'] else [hi]
    W = [torch.tensor(len(tr) / (3 * np.bincount(Y[tr][V[tr, h], h], minlength=3)) * V[tr, h].mean(), dtype=torch.float32) for h in hs]
    W[hs.index(hi)] = torch.tensor(len(tr) / (3 * np.bincount(Y[tr, hi], minlength=3)), dtype=torch.float32)
    lossf = [nn.CrossEntropyLoss(weight=w, label_smoothing=cfg['ls'], ignore_index=-100) for w in W]
    opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=cfg['wd']) if cfg['wd'] else torch.optim.Adam(net.parameters(), 1e-3)
    ema = {k: v.detach().clone() for k, v in net.state_dict().items()} if cfg['ema'] else None

    def logits5(m, idx):
        m.eval(); o = []
        with torch.no_grad():
            for k in range(0, len(idx), 4096):
                z = m(torch.from_numpy(F[idx[k:k + 4096, None] + off])); o.append(torch.softmax(z.view(len(z), nh, 3)[:, hs.index(hi)], 1).numpy())
        return np.concatenate(o)

    shadow = make(cfg, len(cols)) if cfg['ema'] else None
    yv = Y[va, hi]; best, bad, curve, t0, best_state, step = -1, 0, [], time.time(), None, 0
    for ep in range(EPOCHS):
        net.train(); sel = rng.choice(tr, min(PER, len(tr)), replace=False)
        for k in range(0, len(sel), BS):
            b = sel[k:k + BS]; out = net(torch.from_numpy(F[b[:, None] + off])).view(len(b), nh, 3)
            loss = 0.0
            for j, h in enumerate(hs):
                tgt = torch.from_numpy(np.where(V[b, h], Y[b, h], -100)); l = lossf[j](out[:, j], tgt)
                loss = loss + (l if h == hi else 0.5 * l / max(len(hs) - 1, 1))
            opt.zero_grad(); loss.backward(); opt.step()
            if ema is not None:
                with torch.no_grad():
                    step += 1; dec = min(0.999, (1 + step) / (10 + step))          # warm-up so early averages are not tied to the initial weights
                    for kk, v in net.state_dict().items(): ema[kk].mul_(dec).add_(v, alpha=1 - dec)
        f_raw = f1_score(yv, logits5(net, va).argmax(1), average='macro'); rec = dict(epoch=ep, raw=round(float(f_raw), 4))
        cand = [(f_raw, {k: v.clone() for k, v in net.state_dict().items()}, 'raw')]
        if ema is not None:
            shadow.load_state_dict(ema); f_e = f1_score(yv, logits5(shadow, va).argmax(1), average='macro')
            rec['ema'] = round(float(f_e), 4); cand.append((f_e, {k: v.clone() for k, v in ema.items()}, 'ema'))
        f, st, which = max(cand, key=lambda c: c[0]); rec['min'] = round((time.time() - t0) / 60, 1); curve.append(rec)
        print(tag, rec, flush=True)
        if f > best: best, bad, best_state, best_which, best_ep = f, 0, st, which, ep
        else:
            bad += 1
            if bad >= PAT: break
    net.load_state_dict(best_state); prob = logits5(net, va_all); y = Y[va_all, hi]; p = prob.argmax(1)
    m, fm, tab = E.best_m(prob, y)
    rec = dict(name=name, seed=seed, **{k: v for k, v in cfg.items()}, n_features=len(cols), n_params=sum(q.numel() for q in net.parameters()),
               best_epoch=best_ep, best_weights=best_which, val10_mF1=round(float(best), 4), val_mF1=round(E.mf1(y, p), 4),
               val_acc=round(float(accuracy_score(y, p)), 4), best_m=m, mF1_at_best_m=fm, curve=curve, sec=round(time.time() - t0))
    if save and not SMOKE:
        np.savez_compressed(os.path.join(OUT, f'{PFX}_val_deep_{tag}.npz'), prob=prob.astype(np.float32), y=y, day=day[va_all], row=row[va_all])
        torch.save(dict(state=best_state, mu=mu, sd=sd, cfg=cfg, cols=cols, epochs=best_ep + 1), os.path.join(OUT, f'{PFX}_deep_{tag}.pt'))
        with open(RES, 'a') as fh: fh.write(json.dumps(rec) + '\n')
    print(json.dumps({k: v for k, v in rec.items() if k != 'curve'}), flush=True)
    return rec


def recs(): return [json.loads(l) for l in open(RES)] if os.path.exists(RES) else []


if __name__ == '__main__':
    what = sys.argv[1]; have = {(r['name'], r['seed']) for r in recs()}
    if what == 'D1': todo = sorted([(n, s) for s in (0, 1) for n in CONFIGS], key=lambda t: (CONFIGS[t[0]]['cols'] != 'full', t[1]))
    elif what == 'D2':
        R = recs(); mean = {n: np.mean([r['val_mF1'] for r in R if r['name'] == n and r['seed'] in (0, 1)]) for n in CONFIGS}
        top = sorted(mean, key=mean.get, reverse=True)[:3]; print('D2 configs', top, {k: round(float(v), 4) for k, v in mean.items()}, flush=True)
        todo = [(n, s) for s in (2, 3, 4) for n in top]
    else:
        todo = [(t.split(':')[0], int(t.split(':')[1])) for t in what.split(',')]
    for n, s in todo:
        if (n, s) not in have: run(n, s)
