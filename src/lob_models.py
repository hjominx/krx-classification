"""MLPLOB and TLOB (Berti & Kasneci, "TLOB: A Novel Transformer Model with Dual Attention for Price Trend Prediction
with Limit Order Book Data", arXiv:2502.15757) on the 5 s direction task, compared with the conv+GRU used so far.
Reconstructed from the paper's description; settings the paper does not give are marked 'ours'. Design fixed before
running; everything is chosen on VAL, TEST is read only in the 'test' stage.

Models (both start with a Bilinear Normalization layer, Tran et al. 2021, as in the paper)
  MLPLOB : linear projection of each snapshot to hidden size N, then L blocks of
           feature-mixing MLP  U_i = GELU(LN(GELU(X_i W1) W2 + X_i))   (per time step, eq. 8)
           temporal-mixing MLP Z_j = GELU(LN(GELU(U_j W3) W4 + U_j))   (per feature, eq. 9)
           then a dimension-reducing head (N -> N/4 per step, flatten, FC layers shrinking to 3 classes)
  TLOB   : projection to N + sinusoidal positional encoding, then L blocks of
           temporal self-attention (tokens = time steps) + feature-mixing MLP,
           spatial self-attention (tokens = features, after transposing) + temporal-mixing MLP,
           pre-LayerNorm and skip connections, then the same head as MLPLOB
Paper settings: MLPLOB 3 layers, lr 3e-3, sequence 384; TLOB 4 layers, 1 head, lr 1e-4, sequence 128; Adam.
Ours: hidden N = 64, MLP expansion x4, dropout 0.1, class-balanced cross-entropy, 200,000 random training windows per
epoch, batch 256, up to 12 epochs, VAL macro-F1 on every 10th VAL window after each epoch, checkpoint = best epoch
(raw or EMA weights), stop after 3 epochs without improvement - the same budget and rules as deep_v2.py.

Configs (stage S, seed 0; inputs standardised with TRAIN statistics, clipped to +-10)
  mlplob_raw50 / tlob_raw50   : raw 10-level book (40 columns), 50 snapshots   ref: conv+GRU raw 40, VAL 0.4972
  mlplob_full50 / tlob_full50 : the 127 v2 columns, 50 snapshots              ref: conv+GRU 'base' seeds 0-4
  tlob_full128                : 127 columns, 128 snapshots (the paper's TLOB length)
  mlplob_full128              : 127 columns, 128 snapshots (the paper's 384 is not affordable on this CPU; ours)
Stage M: the config with the best VAL macro-F1 among the four 'full' ones gets seeds 1-2 (3-seed bag; 5 seeds are not
         affordable for the slow configs). Reference on VAL: the conv+GRU 'base' bag of seeds 0-2 (deep_v2.py), same rows.
Stage test: that bag on TEST once. Reference on the same rows: the conv+GRU 'base' bag (seeds 0-2, VAL-selected
         checkpoints of deep_v2.py); day-block bootstrap (2,000 resamples) of the macro-F1 difference.
usage: python lob_models.py S | M | bagval | test | smoke   -> results/lob_models.jsonl
"""
import json, math, os, sys, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as Fn
from sklearn.metrics import accuracy_score, confusion_matrix
import common as C
import exp_v2 as E

torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', 2)))
OUT = E.OUT; RES = os.path.join(OUT, 'lob_models.jsonl'); mf1 = E.mf1
HS = list(C.HORIZONS); hi = HS.index(5); BS, EPOCHS, PER, PAT = 256, 12, 200_000, 3
RAW = list(range(40))                                    # ask_dist0-9, bid_dist0-9, log_aq0-9, log_bq0-9 of common.py
COLS = dict(raw=RAW, full=E.FULL)
CONFIGS = dict(mlplob_raw50=('mlplob', 'raw', 50), tlob_raw50=('tlob', 'raw', 50),
               mlplob_full50=('mlplob', 'full', 50), tlob_full50=('tlob', 'full', 50),
               tlob_full128=('tlob', 'full', 128), mlplob_full128=('mlplob', 'full', 128))
LR = dict(mlplob=3e-3, tlob=1e-4); LAYERS = dict(mlplob=3, tlob=4); HID = 64


class BiN(nn.Module):
    """Bilinear normalization: z-score along time (per feature) and along features (per time step), mixed by learnable
    weights, each with a learnable scale and shift"""
    def __init__(s, nf, T):
        super().__init__()
        s.g1, s.b1 = nn.Parameter(torch.ones(nf)), nn.Parameter(torch.zeros(nf))
        s.g2, s.b2 = nn.Parameter(torch.ones(T, 1)), nn.Parameter(torch.zeros(T, 1))
        s.w = nn.Parameter(torch.tensor([0.5, 0.5]))

    def forward(s, x):                                   # x: (B, T, F)
        xt = (x - x.mean(1, keepdim=True)) / (x.std(1, keepdim=True) + 1e-4)
        xf = (x - x.mean(2, keepdim=True)) / (x.std(2, keepdim=True) + 1e-4)
        return s.w[0] * (xt * s.g1 + s.b1) + s.w[1] * (xf * s.g2 + s.b2)


class MixMLP(nn.Module):
    """sigma(LN(sigma(x W1) W2 + x)) along the last axis (eqs. 8-9)"""
    def __init__(s, d, drop):
        super().__init__(); s.fc1, s.fc2, s.ln, s.drop = nn.Linear(d, 4 * d), nn.Linear(4 * d, d), nn.LayerNorm(d), nn.Dropout(drop)

    def forward(s, x): return Fn.gelu(s.ln(s.drop(s.fc2(Fn.gelu(s.fc1(x)))) + x))


class Head(nn.Module):
    def __init__(s, T, d):
        super().__init__(); n = T * (d // 4)
        s.red = nn.Linear(d, d // 4); s.fc = nn.Sequential(nn.Linear(n, 256), nn.GELU(), nn.Linear(256, 64), nn.GELU(), nn.Linear(64, 3))

    def forward(s, x): return s.fc(Fn.gelu(s.red(x)).flatten(1))


class MLPLOB(nn.Module):
    def __init__(s, nf, T, d=HID, L=3, drop=0.1):
        super().__init__()
        s.bin, s.proj = BiN(nf, T), nn.Linear(nf, d)
        s.feat = nn.ModuleList([MixMLP(d, drop) for _ in range(L)]); s.temp = nn.ModuleList([MixMLP(T, drop) for _ in range(L)])
        s.head = Head(T, d)

    def forward(s, x):
        h = s.proj(s.bin(x))
        for f, t in zip(s.feat, s.temp): h = t(f(h).transpose(1, 2)).transpose(1, 2)
        return s.head(h)


class AttnMLP(nn.Module):
    """pre-LN self-attention with skip, then a mixing MLP along the same token dimension"""
    def __init__(s, d, heads, drop):
        super().__init__(); s.ln = nn.LayerNorm(d); s.att = nn.MultiheadAttention(d, heads, dropout=drop, batch_first=True); s.mlp = MixMLP(d, drop)

    def forward(s, x):
        y = s.ln(x); return s.mlp(x + s.att(y, y, y, need_weights=False)[0])


class TLOB(nn.Module):
    def __init__(s, nf, T, d=HID, L=4, heads=1, drop=0.1):
        super().__init__()
        s.bin, s.proj = BiN(nf, T), nn.Linear(nf, d)
        pos = torch.arange(T)[:, None]; div = torch.exp(torch.arange(0, d, 2) * (-math.log(10000.0) / d))
        pe = torch.zeros(T, d); pe[:, 0::2] = torch.sin(pos * div); pe[:, 1::2] = torch.cos(pos * div); s.register_buffer('pe', pe)
        s.temporal = nn.ModuleList([AttnMLP(d, heads, drop) for _ in range(L)])
        s.spatial = nn.ModuleList([AttnMLP(T, heads, drop) for _ in range(L)])
        s.head = Head(T, d)

    def forward(s, x):
        h = s.proj(s.bin(x)) + s.pe
        for ta, sa in zip(s.temporal, s.spatial): h = sa(ta(h).transpose(1, 2)).transpose(1, 2)
        return s.head(h)


def make(arch, nf, T): return MLPLOB(nf, T, L=LAYERS['mlplob']) if arch == 'mlplob' else TLOB(nf, T, L=LAYERS['tlob'])


DATA = {}


def get_data(key, T, days=None):
    """standardised matrix of TRAIN + VAL (or `days`) for one column set and valid windows of length T"""
    argv = sys.argv; sys.argv = argv[:1]; import deep_v2 as D; sys.argv = argv
    days = days or E.TR + E.VA; k = (key, T, tuple(days))
    if k not in DATA:
        DATA.clear(); F, Y, V, gseg, day, row, cov = D.load(days, COLS[key])
        ok = V[:, hi].copy(); ok[:T - 1] = False; ok[T - 1:] &= gseg[T - 1:] == gseg[:-(T - 1)]
        is_tr = day <= int(E.TR[-1]); tr = np.where(ok & is_tr)[0]; va = np.where(ok & ~is_tr & cov)[0]
        s_ = F[tr[::50]]; mu, sd = s_.mean(0), s_.std(0) + 1e-6
        np.subtract(F, mu, out=F); np.divide(F, sd, out=F); np.clip(F, -10, 10, out=F)
        DATA[k] = (F, Y, V, day, row, tr, va, mu, sd)
    return DATA[k]


def predict(net, F, idx, off):
    net.eval(); o = []
    with torch.no_grad():
        for j in range(0, len(idx), 2048): o.append(torch.softmax(net(torch.from_numpy(F[idx[j:j + 2048, None] + off])), 1).numpy())
    return np.concatenate(o)


def log(rec):
    print(json.dumps({k: v for k, v in rec.items() if k != 'curve'})[:500], flush=True)
    with open(RES, 'a') as f: f.write(json.dumps(rec) + '\n')


def recs(stage=None):
    R = [json.loads(l) for l in open(RES)] if os.path.exists(RES) else []
    return [r for r in R if stage is None or r.get('stage') == stage]


def run(name, seed, smoke=False):
    arch, key, T = CONFIGS[name]; tag = f'{name}_s{seed}'; off = np.arange(-T + 1, 1)
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    F, Y, V, day, row, tr, va_all, mu, sd = get_data(key, T, days=(E.TR[:2] + E.VA[:1]) if smoke else None)
    va = va_all[::10]; per, epochs = (2048, 1) if smoke else (PER, EPOCHS)
    net = make(arch, F.shape[1], T); shadow = make(arch, F.shape[1], T)
    w = torch.tensor(len(tr) / (3 * np.bincount(Y[tr, hi], minlength=3)), dtype=torch.float32); lossf = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.Adam(net.parameters(), LR[arch]); ema = {k: v.detach().clone() for k, v in net.state_dict().items()}; step = 0
    yv = Y[va, hi]; best, bad, curve, t0 = -1, 0, [], time.time()
    for ep in range(epochs):
        net.train(); sel = rng.choice(tr, min(per, len(tr)), replace=False)
        for k in range(0, len(sel), BS):
            b = sel[k:k + BS]; loss = lossf(net(torch.from_numpy(F[b[:, None] + off])), torch.from_numpy(Y[b, hi]))
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
            with torch.no_grad():
                step += 1; dec = min(0.999, (1 + step) / (10 + step))
                for kk, v in net.state_dict().items(): ema[kk].mul_(dec).add_(v, alpha=1 - dec)
        shadow.load_state_dict(ema)
        fr, fe = mf1(yv, predict(net, F, va, off).argmax(1)), mf1(yv, predict(shadow, F, va, off).argmax(1))
        curve.append(dict(epoch=ep, raw=round(fr, 4), ema=round(fe, 4), min=round((time.time() - t0) / 60, 1))); print(tag, curve[-1], flush=True)
        f, st, wh = max((fr, {a: b.clone() for a, b in net.state_dict().items()}, 'raw'), (fe, {a: b.clone() for a, b in ema.items()}, 'ema'), key=lambda c: c[0])
        if f > best: best, bad, state, best_ep, which = f, 0, st, ep, wh
        else:
            bad += 1
            if bad >= PAT: break
    net.load_state_dict(state); prob = predict(net, F, va_all, off); y = Y[va_all, hi]; p = prob.argmax(1)
    rec = dict(stage='M' if seed else 'S', name=name, seed=seed, arch=arch, cols=key, T=T, n_features=F.shape[1],
               n_params=sum(q.numel() for q in net.parameters()), lr=LR[arch], layers=LAYERS[arch], best_epoch=best_ep, best_weights=which,
               val10_mF1=round(best, 4), val_mF1=round(mf1(y, p), 4), val_acc=round(float(accuracy_score(y, p)), 4),
               min_per_epoch=round((time.time() - t0) / 60 / len(curve), 1), curve=curve)
    if smoke: print('smoke', tag, rec['n_params'], rec['min_per_epoch'], rec['val_mF1'], flush=True); return rec
    np.savez_compressed(os.path.join(OUT, f'lob_val_{tag}.npz'), prob=prob.astype(np.float32), y=y, day=day[va_all], row=row[va_all])
    torch.save(dict(state=state, mu=mu, sd=sd), os.path.join(OUT, f'lob_{tag}.pt')); log(rec); return rec


def boot(y, d, p_new, p_ref, n=2000, seed=1):
    days = np.unique(d); ix = {k: np.where(d == k)[0] for k in days}; rng = np.random.default_rng(seed); out = []
    for _ in range(n):
        m = np.concatenate([ix[k] for k in rng.choice(days, len(days))]); out.append(mf1(y[m], p_new[m]) - mf1(y[m], p_ref[m]))
    return dict(mean=round(float(np.mean(out)), 4), ci95=[round(float(np.quantile(out, .025)), 4), round(float(np.quantile(out, .975)), 4)])


if __name__ == '__main__':
    what = sys.argv[1]; have = {(r['name'], r['seed']) for r in recs()}
    if what == 'smoke':
        for n in CONFIGS: run(n, 0, smoke=True)
    elif what == 'S':
        order = ['mlplob_raw50', 'tlob_raw50', 'mlplob_full50', 'tlob_full50', 'tlob_full128', 'mlplob_full128']
        for n in order:
            if (n, 0) not in have: run(n, 0)
    elif what == 'M':
        full = [r for r in recs('S') if r['cols'] == 'full']; best = max(full, key=lambda r: r['val_mF1'])['name']
        print('stage M config', best, flush=True)
        for s in (1, 2):
            if (best, s) not in have: run(best, s)
    elif what == 'bagval':
        full = [r for r in recs('S') if r['cols'] == 'full']; best = max(full, key=lambda r: r['val_mF1'])['name']
        Z = [np.load(os.path.join(OUT, f'lob_val_{best}_s{s}.npz')) for s in (0, 1, 2)]
        G = [np.load(os.path.join(OUT, f'v2_val_deep_base_s{s}.npz')) for s in (0, 1, 2)]
        k = lambda z: z['day'].astype(np.int64) * 1_000_000 + z['row']; kz, kg = k(Z[0]), k(G[0]); common = np.intersect1d(kz, kg)
        iz, ig = np.searchsorted(kz, common), np.searchsorted(kg, common); y = Z[0]['y'][iz]; assert (G[0]['y'][ig] == y).all()
        pz = np.mean([z['prob'][iz] for z in Z], 0); pg = np.mean([g['prob'][ig] for g in G], 0)
        log(dict(stage='bagval', name=best, n=int(len(y)), bag_mF1=round(mf1(y, pz.argmax(1)), 4), convgru_bag_mF1=round(mf1(y, pg.argmax(1)), 4),
                 both_mF1=round(mf1(y, (pz + pg).argmax(1)), 4), seed_mF1=[round(mf1(y, z['prob'][iz].argmax(1)), 4) for z in Z]))
    elif what == 'test' and not recs('test'):
        full = [r for r in recs('S') if r['cols'] == 'full']; best = max(full, key=lambda r: r['val_mF1'])['name']
        arch, key, T = CONFIGS[best]; seeds = sorted(r['seed'] for r in recs() if r['name'] == best and r.get('stage') in ('S', 'M'))
        argv = sys.argv; sys.argv = argv[:1]; import deep_v2 as D; sys.argv = argv
        te = [d for d in E.TE if float(C.load_day(d)['cov']) >= C.MIN_COV]
        F, Y, V, gseg, day, row, cov = D.load(te, E.FULL)
        def windows(T):
            ok = V[:, hi].copy(); ok[:T - 1] = False; ok[T - 1:] &= gseg[T - 1:] == gseg[:-(T - 1)]; return np.where(ok)[0]
        ev, ev50 = windows(T), windows(50); ev = np.intersect1d(ev, ev50)
        cc = [E.FULL.index(c) for c in COLS[key]]
        def probs_of(paths, build, Tm, sub):
            o = []; off = np.arange(-Tm + 1, 1)
            for p in paths:
                ck = torch.load(p, weights_only=False); G_ = (F[:, sub] - ck['mu']) / ck['sd']; np.clip(G_, -10, 10, out=G_)
                net = build(); net.load_state_dict(ck['state']); o.append(predict(net, G_.astype(np.float32), ev, off)); del G_
            return o
        P = probs_of([os.path.join(OUT, f'lob_{best}_s{s}.pt') for s in seeds], lambda: make(arch, len(cc), T), T, cc)
        R = probs_of([os.path.join(OUT, f'v2_deep_base_s{s}.pt') for s in (0, 1, 2)], lambda: D.make(D.CONFIGS['base'], len(E.FULL)), 50, list(range(len(E.FULL))))
        y = Y[ev, hi]; d = day[ev]; prob = np.mean(P, 0); p = prob.argmax(1); pr = np.mean(R, 0).argmax(1)
        rec = dict(stage='test', name=best, seeds=seeds, n=int(len(y)), mF1=round(mf1(y, p), 4), acc=round(float(accuracy_score(y, p)), 4),
                   seed_mF1=[round(mf1(y, q.argmax(1)), 4) for q in P], convgru_bag_mF1=round(mf1(y, pr), 4),
                   convgru_seed_mF1=[round(mf1(y, q.argmax(1)), 4) for q in R], both_mF1=round(mf1(y, (prob + np.mean(R, 0)).argmax(1)), 4),
                   boot_vs_convgru=boot(y, d, p, pr), confusion=confusion_matrix(y, p).tolist(),
                   day_mF1={str(k): round(mf1(y[d == k], p[d == k]), 4) for k in np.unique(d)})
        log(rec)
        np.savez_compressed(os.path.join(OUT, f'lob_test_{best}.npz'), prob=prob.astype(np.float32), y=y, day=d, row=row[ev])
