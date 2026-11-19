"""CPU run vs Apple GPU (MPS) run of the MLPLOB / TLOB comparison (lob_models.py; LOB_MODELS.md section 9).
CPU run: results/lob_models.jsonl, results/v2_deep.jsonl. GPU run (run_lob_gpu.py, Apple M1 Max): results_gpu/.
Same code, data, rules and seeds in both runs; MLPLOB / TLOB trained on 2 CPU threads vs on MPS, the conv+GRU reference
('base', seeds 0-2) trained on the CPU in both runs.
  records : every S / M run, the VAL bag and the TEST record of the two runs side by side, and the conv+GRU seeds
  same    : each checkpoint of the GPU run predicts every 10th valid VAL window on the CPU and on MPS -> the difference that
            comes from the device alone (the CPU run's checkpoints are not on this machine)
  speed   : one training step (batch 256, as in the training scripts) and one forward pass of 256 windows, 2 CPU threads vs
            MPS on the same machine; random inputs of the right shape (the time does not depend on the values)
usage: python cpu_gpu.py records | same | speed   -> results/cpu_gpu.jsonl (running a stage again replaces its records)
"""
import json, os, subprocess, sys, time
import numpy as np, torch, torch.nn as nn

argv = sys.argv; sys.argv = argv[:1]
import common as C
import exp_v2 as E
import lob_models as L
import deep_v2 as D
sys.argv = argv
CPU_R, GPU_R = os.path.join(C.ROOT, 'results'), os.path.join(C.ROOT, 'results_gpu'); RES = os.path.join(CPU_R, 'cpu_gpu.jsonl')
rdj = lambda d, f: [json.loads(l) for l in open(os.path.join(d, f))]
best_of = lambda c: max(c.get('raw', -1), c.get('ema', -1))          # the checkpoint rule: better of raw / EMA weights
CLASSES = ('down', 'flat', 'up')


def save(stage, recs):
    keep = [json.loads(l) for l in open(RES) if json.loads(l).get('stage') != stage and json.loads(l).get('stage', '').split(':')[0] != stage] if os.path.exists(RES) else []
    with open(RES, 'w') as f:
        for r in keep + recs: f.write(json.dumps(r) + '\n')
    for r in recs: print(json.dumps(r)[:300], flush=True)


def records():
    cpu, gpu = rdj(CPU_R, 'lob_models.jsonl'), rdj(GPU_R, 'lob_models.jsonl'); out = []
    runs = lambda R: {(r['name'], r['seed']): r for r in R if r.get('stage') in ('S', 'M')}
    rc, rg = runs(cpu), runs(gpu)
    for k, a in rc.items():
        b = rg[k]; n = min(len(a['curve']), len(b['curve']))
        out.append(dict(stage='records:run', name=k[0], seed=k[1], arch=a['arch'], cols=a['cols'], T=a['T'],
                        cpu_val_mF1=a['val_mF1'], gpu_val_mF1=b['val_mF1'], gpu_minus_cpu=round(b['val_mF1'] - a['val_mF1'], 4),
                        cpu_val_acc=a['val_acc'], gpu_val_acc=b['val_acc'], cpu_best_epoch=a['best_epoch'], gpu_best_epoch=b['best_epoch'],
                        cpu_best_weights=a['best_weights'], gpu_best_weights=b['best_weights'], cpu_epochs=len(a['curve']), gpu_epochs=len(b['curve']),
                        curve_max_abs_diff=round(max(abs(best_of(a['curve'][i]) - best_of(b['curve'][i])) for i in range(n)), 4),
                        cpu_min_per_epoch=a['min_per_epoch'], gpu_min_per_epoch=b['min_per_epoch']))
    pick = lambda R: sorted(((r['name'], r['val_mF1']) for r in R.values() if r['cols'] == 'full' and r['seed'] == 0), key=lambda t: -t[1])
    out.append(dict(stage='records:choice', cpu_full_ranking=pick(rc), gpu_full_ranking=pick(rg), cpu_chosen=pick(rc)[0][0], gpu_chosen=pick(rg)[0][0]))
    bv = lambda R: [r for r in R if r.get('stage') == 'bagval'][-1]; a, b = bv(cpu), bv(gpu)
    out.append(dict(stage='records:bagval', name=a['name'], n_cpu=a['n'], n_gpu=b['n'], **{f'{w}_{k}': r[k] for w, r in (('cpu', a), ('gpu', b))
                                                                                          for k in ('bag_mF1', 'convgru_bag_mF1', 'both_mF1', 'seed_mF1')}))
    te = lambda R: [r for r in R if r.get('stage') == 'test'][-1]; a, b = te(cpu), te(gpu)
    rec = lambda cm: [round(cm[i][i] / sum(cm[i]), 4) for i in range(3)]
    days = sorted(a['day_mF1']); dd = [round(b['day_mF1'][d] - a['day_mF1'][d], 4) for d in days]
    out.append(dict(stage='records:test', name=a['name'], n_cpu=a['n'], n_gpu=b['n'],
                    **{f'{w}_{k}': r[k] for w, r in (('cpu', a), ('gpu', b)) for k in ('mF1', 'acc', 'seed_mF1', 'convgru_bag_mF1', 'convgru_seed_mF1', 'both_mF1', 'boot_vs_convgru')},
                    cpu_recall=dict(zip(CLASSES, rec(a['confusion']))), gpu_recall=dict(zip(CLASSES, rec(b['confusion']))),
                    cpu_pred_share=dict(zip(CLASSES, np.round(np.sum(a['confusion'], 0) / a['n'], 4).tolist())),
                    gpu_pred_share=dict(zip(CLASSES, np.round(np.sum(b['confusion'], 0) / b['n'], 4).tolist())),
                    day_gpu_minus_cpu=dict(zip(days, dd)), day_max_abs_diff=max(abs(x) for x in dd)))
    ca = {r['seed']: r for r in rdj(CPU_R, 'v2_deep.jsonl') if r.get('name') == 'base'}
    for r in rdj(GPU_R, 'v2_deep.jsonl'):
        a = ca[r['seed']]; n = min(len(a['curve']), len(r['curve']))
        diff = [abs(a['curve'][i]['raw'] - r['curve'][i]['raw']) for i in range(n)]
        out.append(dict(stage='records:convgru', seed=r['seed'], device='cpu (both runs)', cpu_run_val_mF1=a['val_mF1'], gpu_run_val_mF1=r['val_mF1'],
                        cpu_run_val_acc=a['val_acc'], gpu_run_val_acc=r['val_acc'], cpu_run_best_epoch=a['best_epoch'], gpu_run_best_epoch=r['best_epoch'],
                        epochs=[len(a['curve']), len(r['curve'])], curve_max_abs_diff=round(max(diff), 4),
                        curve_same_up_to_epoch=next((i - 1 for i, x in enumerate(diff) if x > 0), n - 1)))
    save('records', out)


def same():
    torch.set_num_threads(8); cpu, mps = torch.device('cpu'), torch.device('mps')
    days = [d for d in E.VA if float(C.load_day(d)['cov']) >= C.MIN_COV]
    F, Y, V, gseg, day, row, cov = D.load(days, E.FULL)
    def windows(T):
        ok = V[:, L.hi].copy(); ok[:T - 1] = False; ok[T - 1:] &= gseg[T - 1:] == gseg[:-(T - 1)]; return np.where(ok)[0]
    ev = np.intersect1d(windows(128), windows(50))[::10]; y = Y[ev, L.hi]
    def pred(net, G, off, dev):
        net = net.to(dev).eval(); o = []
        with torch.no_grad():
            for j in range(0, len(ev), 2048): o.append(torch.softmax(net(torch.from_numpy(G[ev[j:j + 2048, None] + off]).to(dev)), 1).cpu().numpy())
        return np.concatenate(o)
    cks = [(f'conv+GRU base s{s}', 'cpu', 50, list(range(len(E.FULL))), lambda: D.make(D.CONFIGS['base'], len(E.FULL)), f'v2_deep_base_s{s}.pt') for s in (0, 1, 2)]
    for r in rdj(GPU_R, 'lob_models.jsonl'):
        if r.get('stage') not in ('S', 'M'): continue
        cols = [E.FULL.index(c) for c in L.COLS[r['cols']]]
        cks.append((f"{r['name']} s{r['seed']}", 'mps', r['T'], cols, (lambda a=r['arch'], n=len(cols), T=r['T']: L.make(a, n, T)), f"lob_{r['name']}_s{r['seed']}.pt"))
    out = []
    for name, trained, T, cols, build, path in cks:
        ck = torch.load(os.path.join(GPU_R, path), weights_only=False, map_location='cpu')
        G = ((F[:, cols] - ck['mu']) / ck['sd']).astype(np.float32); np.clip(G, -10, 10, out=G); off = np.arange(-T + 1, 1)
        net = build(); net.load_state_dict(ck['state']); pc, pm = pred(net, G, off, cpu), pred(net, G, off, mps); d = np.abs(pc - pm)
        out.append(dict(stage='same', checkpoint=name, trained_on=trained, n_windows=int(len(ev)), max_abs_prob_diff=float(f'{d.max():.3g}'),
                        mean_abs_prob_diff=float(f'{d.mean():.3g}'), argmax_changed=int((pc.argmax(1) != pm.argmax(1)).sum()),
                        cpu_mF1=round(L.mf1(y, pc.argmax(1)), 4), mps_mF1=round(L.mf1(y, pm.argmax(1)), 4)))
        del G
    save('same', out)


def speed():
    power = subprocess.run(['pmset', '-g', 'ps'], capture_output=True, text=True).stdout.split('\n')[0].split("'")[1:2]
    lid = 'closed' if 'Yes' in subprocess.run(['ioreg', '-r', '-k', 'AppleClamshellState', '-d', '4'], capture_output=True, text=True).stdout.split('AppleClamshellState" = ')[1][:3] else 'open'
    nets = [('convgru_full50', 'convgru', 50)] + [(f'{a}_full{T}', a, T) for a in ('mlplob', 'tlob') for T in (50, 128)]
    NF = len(E.FULL); out = []
    def sync(dev):
        if dev.type == 'mps': torch.mps.synchronize()
    def run(name, arch, T, dev, n):
        torch.manual_seed(0); net = (D.make(D.CONFIGS['base'], NF) if arch == 'convgru' else L.make(arch, NF, T)).to(dev)
        opt = torch.optim.Adam(net.parameters(), 1e-3); lossf = nn.CrossEntropyLoss(weight=torch.tensor([1.3, 0.55, 1.3], device=dev))
        ema = {k: v.detach().clone() for k, v in net.state_dict().items()}
        x, yb = torch.randn(256, T, NF), torch.randint(0, 3, (256,))
        def step():                                                # the training step of deep_v2.py (conv+GRU) / lob_models.py
            net.train(); loss = lossf(net(x.to(dev)), yb.to(dev)); opt.zero_grad(); loss.backward()
            if arch != 'convgru':
                nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
                with torch.no_grad():
                    for k, v in net.state_dict().items(): ema[k].mul_(0.999).add_(v, alpha=0.001)
            else: opt.step()
        def fwd():
            net.eval()
            with torch.no_grad(): net(x.to(dev)).cpu()
        res = {}
        for key, fn in (('train_ms', step), ('fwd256_ms', fwd)):
            for _ in range(3): fn()
            sync(dev); t = time.perf_counter()
            for _ in range(n): fn()
            sync(dev); res[key] = (time.perf_counter() - t) / n * 1e3
        return res
    for name, arch, T in nets:
        torch.set_num_threads(2); c = run(name, arch, T, torch.device('cpu'), 10 if T == 50 else 5)
        g = run(name, arch, T, torch.device('mps'), 30)
        out.append(dict(stage='speed', name=name, T=T, cpu_threads=2, power=(power or ['?'])[0], lid=lid,
                        cpu_train_ms=round(c['train_ms'], 1), mps_train_ms=round(g['train_ms'], 1), train_speedup=round(c['train_ms'] / g['train_ms'], 1),
                        cpu_fwd256_ms=round(c['fwd256_ms'], 1), mps_fwd256_ms=round(g['fwd256_ms'], 1), fwd_speedup=round(c['fwd256_ms'] / g['fwd256_ms'], 1),
                        cpu_epoch_train_min=round(c['train_ms'] * 782 / 6e4, 1), mps_epoch_train_min=round(g['train_ms'] * 782 / 6e4, 1)))
    save('speed', out)


if __name__ == '__main__':
    dict(records=records, same=same, speed=speed)[sys.argv[1]]()
