"""Shared backtest helpers (copied from backtest.py)."""
import _paths  # noqa: F401  (src/common, src/direction, src/volatility, src/report on the import path)
import numpy as np
import common as C

CAP, COMM, TAX = 10_000_000, 0.00015, 0.0020


def book(z):
    L = z['lob'].reshape(-1, 10, 4).astype(np.float64); mid = z['mid'].astype(np.float64)[:, None]
    return mid + L[:, :, 0] * C.TICK, np.round(np.expm1(L[:, :, 1])), mid + L[:, :, 2] * C.TICK, np.round(np.expm1(L[:, :, 3]))


def fill(px, qty, n):                                # walk the book for n shares; returns total cost/proceeds or None
    tot, left = 0.0, n
    for p, q in zip(px, qty):
        take = min(left, q); tot += take * p; left -= take
        if left <= 0: return tot
    return None


