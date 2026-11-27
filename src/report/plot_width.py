"""Figure for round 8 (layer widths of the conv+GRU, width_r8.py of both tasks), from the two width_r8.jsonl files only.
usage: python plot_width.py -> figures/29_width_sweep.png
"""
import _paths  # noqa: F401  (src/common, src/direction, src/volatility, src/report on the import path)
import json, os
import numpy as np, torch
import common as C
from plot_style import FIG, S1, S2, INK, INK2
import matplotlib.pyplot as plt


def panel(ax, task, title):
    R = [json.loads(l) for l in open(os.path.join(C.RESULTS, task, 'width_r8.jsonl'))]
    sm = [r for r in R if r.get('stage') == 'screen_summary'][-1]
    pts = [('64×64', sm['reference_seeds01'], True, False)] + [(f"{w['width'][0]}×{w['width'][1]}", w['seeds01'], False, w['passes']) for w in sm['widths']]
    n = {f"{w['width'][0]}×{w['width'][1]}": w['n_params'] for w in sm['widths']}
    f = 'r8_deep_mt_c64_h64_s0.pt' if task == 'direction' else 'r8_w64x64_s0_vol.pt'    # the 64 x 64 check run of round 8
    n['64×64'] = int(sum(v.numel() for v in torch.load(os.path.join(C.RESULTS, task, f), weights_only=False)['state'].values()))
    pts.sort(key=lambda p: n[p[0]])                                             # by number of weights
    ref = float(np.mean(sm['reference_seeds01'])); x = np.arange(len(pts))
    ax.axhline(ref, color=S2, lw=1.2, ls=(0, (4, 3)), zorder=1)
    for k, (lab, v, is_ref, ok) in enumerate(pts):
        c = S2 if is_ref else S1; m = float(np.mean(v))
        ax.scatter([k, k], v, s=34, facecolor='white', edgecolor=c, lw=1.4, zorder=2)
        ax.scatter([k], [m], s=80, color=c, edgecolor='white', lw=1.2, zorder=3)
        ax.annotate(f'{m:.5f}' + ('\npasses' if ok else ''), (k, max(v)), xytext=(0, 8), textcoords='offset points', ha='center', va='bottom',
                    fontsize=9, color=INK if (ok or is_ref) else INK2, weight='bold' if ok else 'normal')
    ax.set_xticks(x, [f"{p[0]}\n{round(n[p[0]] / 1000)}k" for p in pts])
    lo = min(min(p[1]) for p in pts); hi = max(max(p[1]) for p in pts); pad = (hi - lo) * 0.45
    ax.set_ylim(lo - pad * 0.4, hi + pad); ax.set_xlim(-0.6, len(pts) - 0.4)
    ax.set_ylabel('VAL macro-F1'); ax.set_xlabel('conv channels × GRU units (weights)')
    ax.text(len(pts) - 0.45, ref, '64×64 mean', ha='right', va='bottom', fontsize=9, color=S2)
    ax.set_title(title, fontsize=12)
    if task == 'direction':
        h = [ax.scatter([], [], s=34, facecolor='white', edgecolor=INK2, lw=1.4), ax.scatter([], [], s=80, color=INK2, edgecolor='white')]
        ax.legend(h, ['one seed', 'mean of seeds 0 and 1'], loc='upper left', frameon=False, ncol=2, fontsize=9, handletextpad=0.2, columnspacing=1.0)


fig, (a, b) = plt.subplots(1, 2, figsize=(12.4, 4.9))
panel(a, 'direction', 'Direction 5 s (mt recipe): seeds 0 and 1, mean')
panel(b, 'volatility', 'Volatility 60 s (vreg_mh recipe): seeds 0 and 1, mean')
fig.tight_layout(); fig.savefig(os.path.join(FIG, '29_width_sweep.png'), dpi=200); plt.close(fig); print('wrote 29_width_sweep.png')
