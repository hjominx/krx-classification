"""Cost of one prediction for the models compared in LOB_MODELS.md (inputs: the 127 'full' columns).
  n_params   : trainable parameters
  macs       : multiply-adds for one window, counted with torch.utils.flop_counter (flops / 2); for conv+GRU the count
               is checked against the hand count (2 convolutions + GRU: 3 gates x (input x hidden + hidden x hidden)
               x time steps + head)
  ms_per_256 : forward time for a batch of 256 windows, 2 CPU threads (mean of repeated runs; varies a little)
  ms_per_1   : forward time for a single window, 2 CPU threads
  train_min  : training time of the seed-0 run (minutes, from results/lob_models.jsonl and results/v2_deep.jsonl)
Random inputs of the right shape; the time does not depend on the values.
usage: python lob_cost.py   -> results/lob_cost.jsonl
"""
import json, os, sys, time
import torch
from torch.utils.flop_counter import FlopCounterMode

argv = sys.argv; sys.argv = argv[:1] + ['main']
import deep_v2 as D
import lob_models as L
sys.argv = argv
torch.set_num_threads(2)
OUT = D.OUT; RES = os.path.join(OUT, 'lob_cost.jsonl'); NF = len(D.COLS['full'])


def macs(net, T):
    x = torch.randn(1, T, NF)
    with FlopCounterMode(display=False) as fc, torch.no_grad(): net(x)
    return int(fc.get_total_flops() / 2)


def convgru_by_hand(net, T):
    c1, c2 = net.conv[0], net.conv[2]; g = net.gru; h = net.head[1]
    conv = sum(c.in_channels * c.out_channels * c.kernel_size[0] * T for c in (c1, c2))
    return conv + 3 * (g.input_size * g.hidden_size + g.hidden_size ** 2) * T + h.in_features * h.out_features


def timeit(net, x, n):
    with torch.no_grad():
        for _ in range(2): net(x)
        t = time.perf_counter()
        for _ in range(n): net(x)
    return (time.perf_counter() - t) / n * 1e3


def train_min(name):
    if name == 'convgru_full50':
        r = [r for r in D.recs() if r.get('name') == 'base' and r.get('seed') == 0][-1]; return round(r['sec'] / 60, 1)
    r = [r for r in L.recs() if r.get('stage') == 'S' and r.get('name') == name][-1]; return r['curve'][-1]['min']


if __name__ == '__main__':
    nets = [('convgru_full50', D.make(D.CONFIGS['base'], NF), 50)]
    for arch in ('mlplob', 'tlob'):
        for T in (50, 128): nets.append((f'{arch}_full{T}', L.make(arch, NF, T), T))
    base = None
    with open(RES, 'a') as fh:
        for name, net, T in nets:
            net.eval(); x = torch.randn(256, T, NF); m = macs(net, T); base = base or m
            if name.startswith('convgru'): assert m == convgru_by_hand(net, T), (m, convgru_by_hand(net, T))
            rec = dict(name=name, T=T, n_features=NF, n_params=sum(p.numel() for p in net.parameters()), macs=m, macs_vs_convgru=round(m / base, 1),
                       threads=2, ms_per_256=round(timeit(net, x, 20 if T == 50 else 5), 1), ms_per_1=round(timeit(net, x[:1], 100), 2),
                       train_min=train_min(name))
            print(json.dumps(rec), flush=True); fh.write(json.dumps(rec) + '\n')
