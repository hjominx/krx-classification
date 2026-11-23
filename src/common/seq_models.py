"""Shared sequence models on the v2 features, used by both tasks: conv+GRU, a dilated TCN and a small Transformer
(make() builds one from a config), the training budget of round 2 and the loader that puts days into one matrix.
Budget : windows of T = 50 snapshots inside one segment, batch 256, up to 12 epochs x 200,000 random training windows,
         stop after 3 epochs without improvement on VAL.
env SMOKE=1: 1 epoch of 4,096 windows - a code check only.
Importing this module sets torch to 2 threads, as importing deep_v2.py did; callers may set another count afterwards.
Taken out of the round-2 direction script (src/direction/deep_v2.py) without changes.
"""
import _paths  # noqa: F401  (src/common, src/direction, src/volatility, src/report on the import path)
import os
import numpy as np, torch, torch.nn as nn, torch.nn.functional as Fn
import common as C
import data_v2 as E

torch.set_num_threads(2)
SMOKE = bool(os.environ.get('SMOKE'))
T, BS, EPOCHS, PER, PAT = 50, 256, (1 if SMOKE else 12), (4096 if SMOKE else 200_000), 3
HS = list(C.HORIZONS); hi = HS.index(5)


class ConvGRU(nn.Module):
    def __init__(s, nf, hidden, dropout, nout):
        super().__init__()
        s.conv = nn.Sequential(nn.Conv1d(nf, 64, 3, padding=1), nn.GELU(), nn.Conv1d(64, 64, 3, padding=1), nn.GELU())
        s.gru = nn.GRU(64, hidden, batch_first=True); s.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, nout))

    def forward(s, x): return s.head(s.gru(s.conv(x.transpose(1, 2)).transpose(1, 2))[0][:, -1])


class TCN(nn.Module):
    def __init__(s, nf, dropout, nout, c=64):
        super().__init__()
        s.stem = nn.Conv1d(nf, c, 1); s.drop = nn.Dropout(dropout)
        s.a = nn.ModuleList([nn.Conv1d(c, c, 3, padding=d, dilation=d) for d in (1, 2, 4, 8)])
        s.b = nn.ModuleList([nn.Conv1d(c, c, 3, padding=d, dilation=d) for d in (1, 2, 4, 8)])
        s.head = nn.Linear(c, nout)

    def forward(s, x):
        h = s.stem(x.transpose(1, 2))
        for a, b in zip(s.a, s.b): h = h + Fn.gelu(b(s.drop(Fn.gelu(a(h)))))
        return s.head(s.drop(h[:, :, -1]))


class Tfm(nn.Module):
    def __init__(s, nf, dropout, nout, d=64):
        super().__init__()
        s.stem = nn.Sequential(nn.Conv1d(nf, d, 3, padding=1), nn.GELU())
        s.pos = nn.Parameter(torch.zeros(1, T, d)); nn.init.normal_(s.pos, std=0.02)
        s.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, 4, 2 * d, dropout, batch_first=True, norm_first=True, activation='gelu'), 2)
        s.norm = nn.LayerNorm(d); s.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * d, nout))

    def forward(s, x):
        h = s.norm(s.enc(s.stem(x.transpose(1, 2)).transpose(1, 2) + s.pos))
        return s.head(torch.cat([h[:, -1], h.mean(1)], 1))


def make(cfg, nf):
    nout = 3 * (len(HS) if cfg['mt'] else 1)
    if cfg['arch'] == 'convgru': return ConvGRU(nf, cfg['hidden'], cfg['dropout'], nout)
    if cfg['arch'] == 'tcn': return TCN(nf, cfg['dropout'], nout)
    return Tfm(nf, cfg['dropout'], nout)


def load(days, cols):
    """one concatenated matrix for all days; returns F, Y(n,5), V(n,5), gseg, day, row, cov_ok"""
    parts = [C.load_day(d) for d in days]; n = sum(len(z['ts']) for z in parts)
    F = np.empty((n, len(cols)), np.float32); Y = np.empty((n, len(HS)), np.int64); V = np.empty((n, len(HS)), bool)
    gseg = np.empty(n, np.int64); day = np.empty(n, np.int64); row = np.empty(n, np.int64); cov = np.empty(n, bool); k = 0
    for j, (d, z) in enumerate(zip(days, parts)):
        m = len(z['ts']); X = np.hstack([z['X'], np.load(os.path.join(E.FT, f'{d}.npz'))['T'], np.load(os.path.join(E.F2, f'{d}.npz'))['X2']])
        F[k:k + m] = X[:, cols]; Y[k:k + m] = z['Y']; V[k:k + m] = z['V']; gseg[k:k + m] = j * 100000 + z['seg']
        day[k:k + m] = int(d); row[k:k + m] = np.arange(m); cov[k:k + m] = float(z['cov']) >= C.MIN_COV; k += m
        parts[j] = None
    return F, Y, V, gseg, day, row, cov


def windows(V, gseg):
    ok = V[:, hi].copy(); ok[:T - 1] = False; ok[T - 1:] &= gseg[T - 1:] == gseg[:-(T - 1)]; return ok
