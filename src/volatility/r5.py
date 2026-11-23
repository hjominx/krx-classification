"""Round 5: conv+GRU with extra targets, regularisation and a joint model for both tasks. Design fixed before running;
everything is chosen on VAL, TEST is read only in the final stages. Uses the GPU when present (CUDA or Apple MPS;
env KRX_DEVICE=cpu|mps|cuda to force one); conv+GRU trains about 7x faster on an M1 Max GPU than on 2 CPU threads.
Model : the conv+GRU of seq_models.py (two 1D convolutions, GRU 64, EMA weights, 50 snapshots), inputs: the 145 'allv'
        columns of vol_v2.py (they contain the 127 'full' columns used for direction so far), standardised with TRAIN
        statistics, clipped to +-10. Budget and rules of round 2 (src/direction/deep_v2.py): up to 12 epochs x 200,000
        windows, batch 256, after each epoch VAL on every 10th VAL window, best epoch kept (raw or EMA weights), stop
        after 3 epochs without gain.
Heads : direction - 5 horizons (1, 2, 5, 10, 30 s), class-balanced cross-entropy, 5 s main, others weight 0.5 in total
        volatility - log(1 + RV) of the next 60 s (main), and with 'mh' also of the next 10, 30, 120 s (weight 0.5 in
        total); each standardised with TRAIN statistics, mean squared error; classes from two thresholds chosen on VAL
VAL score per epoch: direction - macro-F1 of the most probable 5 s class; volatility - macro-F1 with the best two
        thresholds (as vol_r4.py). A joint model keeps the best epoch of each task separately.
Configs (all 'allv')
  vreg_mh  : volatility regression + 10 / 30 / 120 s volatility as extra targets
  vreg_lr  : volatility regression, lr 3e-4 instead of 1e-3 (round 4 was best after its first epoch)
  vreg_reg : volatility regression, dropout 0.3 + AdamW weight decay 1e-4
  vreg_st2 : volatility regression, every 2nd snapshot (the 50 inputs cover ~20 s instead of ~10 s)
  joint    : one model for both tasks: the 5 direction horizons + the 4 volatility horizons
  mt_allv  : direction only (5 horizons) on allv - separates the effect of the inputs from that of the joint training
Stage screen: every config, seeds 0 and 1. A config passes for a task if its seed-0/1 mean VAL macro-F1 beats
        volatility: round-4 conv+GRU regression seeds 0/1 (results/volatility/vol_r4.jsonl, mean 0.6328)
        direction : mt seeds 0/1 of round 2 (results/direction/v2_deep.jsonl, mean 0.5129)
Stage more: passing configs get seeds 2, 3, 4 (5-seed bags).
labels: builds the 10 / 30 / 120 s volatility labels (data/feat_vol_mh) and, where missing, the 60 s ones exactly as
        vol_cls.py (data/feat_vol); puts the class-cuts record next to the results so that vol_v2.py can run there.
usage: python r5.py labels | train <cfg>[,<cfg>] <seed>[,<seed>] [threads] | more [threads]   -> results/volatility/r5.jsonl
  env SMOKE=1 (seq_models.py's switch): 1 epoch of 4,096 windows on 2 TRAIN days and 1 VAL day - a code check only
"""
import _paths  # noqa: F401  (src/common, src/direction, src/volatility, src/report on the import path)
import json, os, sys, time
import numpy as np, torch, torch.nn as nn
import common as C
import data_v2 as E

argv = sys.argv; sys.argv = argv[:1]
import seq_models as D
import vol_v2 as V
import vol_r4 as R4
sys.argv = argv
OUT = C.RES_VOL; os.makedirs(OUT, exist_ok=True); RES = os.path.join(OUT, 'r5.jsonl'); VD = os.path.join(C.DATA, 'feat_vol'); VM = os.path.join(C.DATA, 'feat_vol_mh')
REPO_CUTS = os.path.join(C.ROOT, 'results', 'volatility', 'vol_cls.jsonl')
cls = lambda x: np.digitize(x, V.CUTS, right=True); mf1 = R4.mf1; tune = R4.tune
WV = (10, 30, 60, 120); IV = WV.index(60); HS, hi, T = D.HS, D.hi, D.T
SMOKE = D.SMOKE; TR, VA = (E.TR[:2], E.VA[:1]) if SMOKE else (E.TR, E.VA)
CFG = dict(vreg_mh=dict(vol=True, mh=True), vreg_lr=dict(vol=True, lr=3e-4), vreg_reg=dict(vol=True, dropout=0.3, wd=1e-4),
           vreg_st2=dict(vol=True, stride=2), joint=dict(dir=True, vol=True, mh=True), mt_allv=dict(dir=True))
DEF = dict(dir=False, vol=False, mh=False, lr=1e-3, wd=0.0, dropout=0.2, stride=1, hidden=64)
REF = dict(vol=('vol_r4.jsonl', 'convgru_reg'), dir=('v2_deep.jsonl', 'mt'))


def device():
    want = os.environ.get('KRX_DEVICE', 'auto')
    if want != 'auto': return torch.device(want)
    return torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')


DEV = device()


def log(rec):
    print(json.dumps({k: v for k, v in rec.items() if k != 'curve'})[:600], flush=True)
    with open(RES, 'a') as f: f.write(json.dumps(rec) + '\n')


def recs(stage=None):
    R = [json.loads(l) for l in open(RES)] if os.path.exists(RES) else []
    return [r for r in R if stage is None or r.get('stage') == stage]


def labels():
    argv = sys.argv; sys.argv = argv[:1] + ['import']; import vol_cls as VC; sys.argv = argv      # rv() of vol_cls.py
    os.makedirs(VD, exist_ok=True); os.makedirs(VM, exist_ok=True); made = [0, 0]
    for d in VC.TR + VC.VA + VC.TE:
        p, q = os.path.join(VD, f'{d}.npz'), os.path.join(VM, f'{d}.npz')
        if os.path.exists(p) and os.path.exists(q): continue
        z = C.load_day(d); ts, mid, seg = z['ts'], z['mid'], z['seg']
        if not os.path.exists(p):                                     # exactly vol_cls.py's 'labels' stage
            f, ok = VC.rv(ts, mid, seg, 60 * 10**9, True)
            P = np.stack([np.log1p(VC.rv(ts, mid, seg, s * 10**9, False)[0]) for s in (10, 60, 300)], 1).astype(np.float32)
            np.savez(p, rv=f.astype(np.float32), ok=ok, past=P, past60=VC.rv(ts, mid, seg, 60 * 10**9, False)[0]); made[0] += 1
        if not os.path.exists(q):
            o = {}
            for w in (10, 30, 120):
                f, ok = VC.rv(ts, mid, seg, w * 10**9, True); o[f'rv{w}'] = f.astype(np.float32); o[f'ok{w}'] = ok
            np.savez(q, **o); made[1] += 1
    here = os.path.join(OUT, 'vol_cls.jsonl')
    if os.path.abspath(here) != os.path.abspath(REPO_CUTS) and not os.path.exists(here):
        open(here, 'w').write([l for l in open(REPO_CUTS) if '"labels"' in l][-1])
    log(dict(stage='labels', built_60s=made[0], built_mh=made[1]))


DATA = {}


def get_data():
    """TRAIN + VAL: standardised allv matrix, direction labels, volatility targets for the 4 horizons"""
    if not DATA:
        days = TR + VA; F, Y, Vd, gseg, day, row, cov = D.load(days, list(V.ALLV)); n = len(F)
        zv = np.zeros((n, len(WV)), np.float32); okv = np.zeros((n, len(WV)), bool); yv = np.zeros(n, np.int64); k = 0
        for d in days:
            a, b = np.load(os.path.join(VD, f'{d}.npz')), np.load(os.path.join(VM, f'{d}.npz')); m = len(a['ok'])
            for j, w in enumerate(WV):
                rv, ok = (a['rv'], a['ok']) if w == 60 else (b[f'rv{w}'], b[f'ok{w}'])
                zv[k:k + m, j] = np.log1p(np.nan_to_num(rv)); okv[k:k + m, j] = ok
            yv[k:k + m] = cls(a['rv']); k += m
        assert k == n
        is_tr = np.isin(day, [int(d) for d in TR]); base = is_tr & (Vd[:, hi] | okv[:, IV])
        s_ = F[np.where(base)[0][::50]]; mu, sd = s_.mean(0), s_.std(0) + 1e-6
        np.subtract(F, mu, out=F); np.divide(F, sd, out=F); np.clip(F, -10, 10, out=F)
        zm = np.array([zv[is_tr & okv[:, j], j].mean() for j in range(len(WV))], np.float32)
        zs = np.array([zv[is_tr & okv[:, j], j].std() for j in range(len(WV))], np.float32)
        DATA.update(F=F, Y=Y, Vd=Vd, zv=zv, okv=okv, yv=yv, gseg=gseg, day=day, row=row, cov=cov, is_tr=is_tr, mu=mu, sd=sd, zm=zm, zs=zs)
    return DATA


def outputs(net, F, idx, off):
    net.eval(); o = []
    with torch.no_grad():
        for j in range(0, len(idx), 4096): o.append(net(torch.from_numpy(F[idx[j:j + 4096, None] + off]).to(DEV)).float().cpu().numpy())
    return np.concatenate(o)


def run(name, seed):
    cfg = dict(DEF, **CFG[name]); st = cfg['stride']; span = st * (T - 1); off = np.arange(-span, 1, st); X = get_data()
    F, Y, Vd, zv, okv, gseg = X['F'], X['Y'], X['Vd'], X['zv'], X['okv'], X['gseg']
    seg_ok = np.zeros(len(F), bool); seg_ok[span:] = gseg[span:] == gseg[:-span]
    dir_ok, vol_ok = seg_ok & Vd[:, hi], seg_ok & okv[:, IV]
    use = (dir_ok if cfg['dir'] else 0) | (vol_ok if cfg['vol'] else 0)
    tr = np.where(use & X['is_tr'])[0]; ev = ~X['is_tr'] & X['cov']
    tasks = (['dir'] if cfg['dir'] else []) + (['vol'] if cfg['vol'] else [])
    va_all = {'dir': np.where(dir_ok & ev)[0], 'vol': np.where(vol_ok & ev)[0]}; va = {t: va_all[t][::10] for t in tasks}
    nd = 3 * len(HS) if cfg['dir'] else 0; hv = list(range(len(WV))) if cfg['mh'] else [IV]; nv = len(hv) if cfg['vol'] else 0
    torch.manual_seed(seed); rng = np.random.default_rng(seed); t0 = time.time()
    net = D.ConvGRU(F.shape[1], cfg['hidden'], cfg['dropout'], nd + nv).to(DEV); sh = D.ConvGRU(F.shape[1], cfg['hidden'], cfg['dropout'], nd + nv).to(DEV)
    opt = (torch.optim.AdamW(net.parameters(), cfg['lr'], weight_decay=cfg['wd']) if cfg['wd'] else torch.optim.Adam(net.parameters(), cfg['lr']))
    ema = {a: b.detach().clone() for a, b in net.state_dict().items()}; step = 0
    if cfg['dir']:
        W = [torch.tensor(Vd[tr, h].sum() / (3 * np.bincount(Y[tr][Vd[tr, h], h], minlength=3)), dtype=torch.float32, device=DEV) for h in range(len(HS))]
        ce = [nn.CrossEntropyLoss(weight=w, ignore_index=-100) for w in W]
    zm, zs = X['zm'], X['zs']

    def loss_of(out, b):
        loss = 0.0
        if cfg['dir']:
            for j in range(len(HS)):
                t = torch.from_numpy(np.where(Vd[b, j], Y[b, j], -100))
                if (t != -100).any(): l = ce[j](out[:, 3 * j:3 * j + 3], t.to(DEV)); loss = loss + (l if j == hi else 0.5 * l / (len(HS) - 1))
        if cfg['vol']:
            for i, j in enumerate(hv):
                m = torch.from_numpy(okv[b, j].astype(np.float32)).to(DEV); t = torch.from_numpy((zv[b, j] - zm[j]) / zs[j]).to(DEV)
                l = ((out[:, nd + i] - t) ** 2 * m).sum() / m.sum().clamp(min=1.0); loss = loss + (l if j == IV else 0.5 * l / (len(hv) - 1))
        return loss

    def score(o, task, idx):
        if task == 'dir': return mf1(Y[idx, hi], o[:, 3 * hi:3 * hi + 3].argmax(1))
        return tune(o[:, nd + hv.index(IV)] * zs[IV] + zm[IV], X['yv'][idx])[0]

    best = {t: dict(f=-1.0, bad=0) for t in tasks}; curve = []; ev_idx = np.unique(np.concatenate([va[t] for t in tasks]))
    for ep in range(D.EPOCHS):
        net.train(); sel = rng.choice(tr, min(D.PER, len(tr)), replace=False)
        for k in range(0, len(sel), D.BS):
            b = sel[k:k + D.BS]; opt.zero_grad(); loss_of(net(torch.from_numpy(F[b[:, None] + off]).to(DEV)), b).backward(); opt.step()
            with torch.no_grad():
                step += 1; dec = min(0.999, (1 + step) / (10 + step))
                for a, b2 in net.state_dict().items(): ema[a].mul_(dec).add_(b2, alpha=1 - dec)
        sh.load_state_dict(ema); o_raw, o_ema = outputs(net, F, ev_idx, off), outputs(sh, F, ev_idx, off); c = dict(epoch=ep)
        for t in tasks:
            pos = np.searchsorted(ev_idx, va[t]); fr, fe = score(o_raw[pos], t, va[t]), score(o_ema[pos], t, va[t])
            c[t] = [round(fr, 4), round(fe, 4)]; f, wh = max((fr, 'raw'), (fe, 'ema'))
            if f > best[t]['f']:
                best[t].update(f=f, bad=0, ep=ep, which=wh, state={a: b.detach().clone() for a, b in (net.state_dict() if wh == 'raw' else ema).items()})
            else: best[t]['bad'] += 1
        c['min'] = round((time.time() - t0) / 60, 1); curve.append(c); print(name, seed, c, flush=True)
        if all(best[t]['bad'] >= D.PAT for t in tasks): break
    for t in tasks:
        sh.load_state_dict(best[t]['state']); idx = va_all[t]; o = outputs(sh, F, idx, off); tag = f'{name}_s{seed}_{t}'
        if t == 'dir':
            p = np.exp(o[:, 3 * hi:3 * hi + 3] - o[:, 3 * hi:3 * hi + 3].max(1, keepdims=True)); p /= p.sum(1, keepdims=True); y = Y[idx, hi]
            m, fm, _ = E.best_m(p, y); f = mf1(y, p.argmax(1)); extra = dict(best_m=m, mF1_at_best_m=fm)
            np.savez_compressed(os.path.join(OUT, f'r5_val_{tag}.npz'), prob=p.astype(np.float32), y=y, day=X['day'][idx], row=X['row'][idx])
        else:
            pr = o[:, nd + hv.index(IV)] * zs[IV] + zm[IV]; y = X['yv'][idx]; f, thr = tune(pr, y); extra = dict(thr=thr)
            np.savez_compressed(os.path.join(OUT, f'r5_val_{tag}.npz'), pred=pr.astype(np.float32), y=y, day=X['day'][idx], row=X['row'][idx])
        torch.save(dict(state={a: b.cpu() for a, b in best[t]['state'].items()}, mu=X['mu'], sd=X['sd'], zm=zm, zs=zs, cfg=cfg, nout=nd + nv, hv=hv),
                   os.path.join(OUT, f'r5_{tag}.pt'))
        log(dict(stage='train', name=name, seed=seed, task=t, device=DEV.type, best_epoch=best[t]['ep'], best_weights=best[t]['which'],
                 val10=round(best[t]['f'], 4), val_mF1=round(f, 4), n_val=int(len(idx)), **extra, smoke=SMOKE, sec=round(time.time() - t0),
                 cfg=cfg, curve=curve))


def reference(task):
    f, name = REF[task]; rr = [json.loads(l) for l in open(os.path.join(C.RES_DIR if task == 'dir' else OUT, f))]
    key = 'val_mF1'; vals = [r[key] for r in rr if r.get('name') == name and r.get('seed') in (0, 1) and key in r]
    return float(np.mean(vals)) if len(vals) == 2 else None


def passing():
    out = []
    for name in CFG:
        for t in ('dir', 'vol'):
            v = [r['val_mF1'] for r in recs('train') if r['name'] == name and r['task'] == t and r['seed'] in (0, 1)]
            ref = reference(t)
            if len(v) == 2 and ref is not None and np.mean(v) > ref: out.append((name, t, round(float(np.mean(v)), 4), round(ref, 4)))
    return out


if __name__ == '__main__':
    what = sys.argv[1]
    if what == 'labels': labels()
    elif what == 'train':
        torch.set_num_threads(int(sys.argv[4]) if len(sys.argv) > 4 else 2); have = {(r['name'], r['seed']) for r in recs('train')}
        for name in sys.argv[2].split(','):
            for s in [int(x) for x in sys.argv[3].split(',')]:
                if (name, s) not in have: run(name, s)
    elif what == 'more':
        torch.set_num_threads(int(sys.argv[2]) if len(sys.argv) > 2 else 2); P = passing(); print('passing', P, flush=True)
        have = {(r['name'], r['seed']) for r in recs('train')}
        for name in dict.fromkeys(n for n, _, _, _ in P):
            for s in (2, 3, 4):
                if (name, s) not in have: run(name, s)
