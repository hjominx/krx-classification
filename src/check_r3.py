"""Round 3, 5 s direction: split-half check of the stacking models' class multipliers on VAL. TEST is not read here.
Added after post_r3.py had run (it checks the split-half only for avg / cls / smooth candidates). The stackers'
VAL probabilities are leave-one-day-out, but their multipliers were chosen on all 12 VAL days. Here the multipliers
are chosen on VAL days 1-6 and scored on days 7-12, and the reverse, next to the round-2 winner on the same rows.
usage: python check_r3.py   -> results/r3_check.jsonl
"""
import json, os, sys
import numpy as np

sys.argv = sys.argv[:1] + ['main']
import post_r3 as PR

OUT = PR.OUT; RES = os.path.join(OUT, 'r3_check.jsonl')
NAMES = ['stack_lr', 'stack_lr_uw', 'stack_lr_ctx', 'stack_lr_ctx_uw', 'stack_gbm', 'stack_gbm_uw', 'stack_gbm_ctx', 'stack_gbm_ctx_uw']

if __name__ == '__main__':
    z = np.load(os.path.join(OUT, 'r3_val_stack_lr.npz')); keys = PR.KEY(z); y = z['y'].astype(int); d = keys // 1_000_000
    g = np.load(os.path.join(OUT, 'v2_val_gbm.npz')); kg = PR.KEY(g); ix = np.searchsorted(kg, keys); assert (kg[ix] == keys).all()
    ref = PR.r2_winner({'gbm': g['prob'][ix]}, keys); days = np.unique(d); H = (days[:len(days) // 2], days[len(days) // 2:])
    with open(RES, 'a') as fh:
        for nm in NAMES:
            s = np.load(os.path.join(OUT, f'r3_val_{nm}.npz')); assert (PR.KEY(s) == keys).all(); pr = s['prob']
            for a, b in ((0, 1), (1, 0)):
                sel = np.where(np.isin(d, H[a]))[0]; ev = np.isin(d, H[b]); f, mm = PR.best_cls(pr, y, sel, grid=PR.MFW)
                rec = dict(stage='stack_half', name=nm, chosen_on=[int(x) for x in H[a]], mult=mm, mF1_chosen_half=round(f, 4),
                           mF1_other_half=round(PR.mf1(y[ev], PR.apply(pr[ev], mm)), 4),
                           r2_winner_other_half=round(PR.mf1(y[ev], PR.apply(ref[ev], [1, 1.1, 1])), 4))
                print(json.dumps(rec), flush=True); fh.write(json.dumps(rec) + '\n')
