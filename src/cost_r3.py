"""Round 3: prediction time of the deep members, 2 CPU threads, a batch of 256 windows and a single window.
Random inputs of the right shape (the time does not depend on the values). Times vary a little between runs.
usage: python cost_r3.py   -> results/r3_cost.jsonl
"""
import json, os, sys, time
import torch

argv = sys.argv; sys.argv = argv[:1] + ['main']
import deep_v2 as D
import lob_models as L
sys.argv = argv
torch.set_num_threads(2)
OUT = D.OUT
RES = os.path.join(OUT, 'r3_cost.jsonl')


def timeit(net, x, n):
    with torch.no_grad():
        for _ in range(2): net(x)
        t = time.perf_counter()
        for _ in range(n): net(x)
    return (time.perf_counter() - t) / n * 1e3


if __name__ == '__main__':
    with open(RES, 'a') as fh:
        nets = [(n, f'v2_deep_{n}_s0.pt') for n in ('base', 'mt', 'tcn')] + [('mt_st2', 'r3_deep_mt_st2_s5.pt')]
        for name, f in nets:
            ck = torch.load(os.path.join(OUT, f), weights_only=False); net = D.make(ck['cfg'], len(ck['cols'])); net.load_state_dict(ck['state']); net.eval()
            x = torch.randn(256, D.T, len(ck['cols']))
            rec = dict(stage='cost', name=name, n_params=sum(p.numel() for p in net.parameters()), threads=2,
                       ms_per_256=round(timeit(net, x, 20), 1), ms_per_1=round(timeit(net, x[:1], 200), 2))
            print(json.dumps(rec), flush=True); fh.write(json.dumps(rec) + '\n')
        ck = torch.load(os.path.join(OUT, 'lob_tlob_full128_s0.pt'), weights_only=False); nf = len(ck['mu'])
        net = L.make('tlob', nf, 128); net.load_state_dict(ck['state']); net.eval(); x = torch.randn(256, 128, nf)
        rec = dict(stage='cost', name='tlob_full128', n_params=sum(p.numel() for p in net.parameters()), threads=2,
                   ms_per_256=round(timeit(net, x, 5), 1), ms_per_1=round(timeit(net, x[:1], 100), 2))
        print(json.dumps(rec), flush=True); fh.write(json.dumps(rec) + '\n')
