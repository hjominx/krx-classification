"""Figures for VOLATILITY.md section 11 (calm -> turbulent check), from results/volatility/transition.jsonl only.
usage: python plot_transition.py -> figures/27_vol_confusion_final.png, figures/28_vol_transitions.png
"""
import _paths  # noqa: F401  (src/common, src/direction, src/volatility, src/report on the import path)
import json, os
import numpy as np
import common as C
from plot_style import FIG, S1, S2, INK, INK2, GRID
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

R = [json.loads(l) for l in open(os.path.join(C.RES_VOL, 'transition.jsonl'))]
one = lambda **k: next(r for r in R if all(r.get(a) == b for a, b in k.items()))
LAB = ['calm', 'normal', 'turbulent']


def save(fig, name):
    fig.tight_layout(); fig.savefig(os.path.join(FIG, name), dpi=200); plt.close(fig); print('wrote', name)


def confusion():
    """round 7's TEST confusion matrix (the final model until round 8), rows = actual class (as 18_vol_confusion.png for round 1)"""
    r = one(stage='confusion', split='test', name='r7'); n = np.array(r['n']); cm = n / n.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    ax.imshow(cm, cmap=LinearSegmentedColormap.from_list('b', ['#f3f7fd', S1]), vmin=0, vmax=1)
    for i in range(3):
        for j in range(3):
            c = 'white' if cm[i, j] > 0.5 else INK
            ax.text(j, i - 0.09, f'{cm[i, j]:.1%}', ha='center', va='center', fontsize=13, color=c, weight='bold' if i == j else 'normal')
            ax.text(j, i + 0.2, f'{n[i, j]:,}', ha='center', va='center', fontsize=8.5, color=c)
    ax.set_xticks(range(3), LAB); ax.set_yticks(range(3), LAB); ax.set_xlabel('predicted'); ax.set_ylabel('actual (next 60 s)')
    ax.set_title(f"Round 7 on TEST: share of each actual class\n{r['rows']:,} rows", fontsize=11, color=INK, loc='left')
    ax.grid(False)
    for s in ax.spines.values(): s.set_visible(False)
    save(fig, '27_vol_confusion_final.png')


def transitions():
    """left: how often each transition happens from its state vs how often the final model calls it (TEST);
    right: calm moments ranked by the final forecast, share that turns turbulent and share of all onsets caught (VAL)"""
    c = one(stage='cells', split='test'); n = np.array(c['n']); f = np.array(one(stage='forecast', split='test', name='r7')['cell_forecast_share'])
    rows = [(s, y) for s in range(3) for y in range(3)][::-1]
    base = [n[s, y] / n[s].sum() for s, y in rows]; hit = [f[s, y, y] for s, y in rows]
    fig, (a, b) = plt.subplots(1, 2, figsize=(12.4, 5.2), gridspec_kw=dict(width_ratios=[1.15, 1]))
    yy = np.arange(len(rows))
    for k in yy: a.plot([base[k], hit[k]], [k, k], color=GRID, lw=2, zorder=1)
    a.scatter(base, yy, s=70, facecolor='white', edgecolor=INK2, lw=1.6, zorder=2, label='happens from that state')
    a.scatter(hit, yy, s=70, color=S1, edgecolor='white', lw=1.2, zorder=3, label='called correctly by the model')
    for k, (s, y) in enumerate(rows):
        rare = s != y and abs(s - y) == 2
        a.annotate(f'{hit[k]:.1%}' if hit[k] >= 0.001 else f'{hit[k]:.2%}', (max(hit[k], base[k]), k), xytext=(9, 0),
                   textcoords='offset points', ha='left', va='center', fontsize=9.5, color=S1, weight='bold' if rare else 'normal')
    a.set_yticks(yy, [f'{LAB[s]} → {LAB[y]}' + ('  (two steps)' if abs(s - y) == 2 else '') for s, y in rows])
    a.set_xlim(-0.03, 1.03); a.set_xticks([0, .25, .5, .75, 1], ['0%', '25%', '50%', '75%', '100%'])
    a.set_xlabel('share of rows in that transition')
    a.legend(frameon=False, loc='lower left', bbox_to_anchor=(0, 1.0), ncol=2, fontsize=9.5, handletextpad=0.3, columnspacing=1.2)
    a.set_title('Previous 60 s → next 60 s (TEST, round 7)', fontsize=12, pad=26)
    t = one(stage='topk', split='val', name='r7'); q = [100 * x['top_share'] for x in t['top']]
    prec = [x['turbulent'] for x in t['top']]; rec = [x['share_of_calm_turb'] for x in t['top']]
    b.plot(q, prec, color=S2, marker='o', ms=6, lw=2); b.plot(q, rec, color=S1, marker='s', ms=6, lw=2)
    b.axhline(t['base_rate'], color=INK2, lw=1.2, ls='--')
    b.annotate('turned turbulent', (q[0], prec[0]), xytext=(0, 10), textcoords='offset points', ha='left', fontsize=10, color=S2)
    b.annotate('of all calm → turbulent caught', (q[-1], rec[-1]), xytext=(-6, 10), textcoords='offset points', ha='right', fontsize=10, color=S1)
    b.annotate(f"base rate {t['base_rate']:.1%}", (1.4, t['base_rate']), xytext=(0, 6), textcoords='offset points', ha='center', fontsize=9.5, color=INK2)
    for x, y1 in zip(q, prec): b.annotate(f'{y1:.0%}', (x, y1), xytext=(0, -15), textcoords='offset points', ha='center', fontsize=8.5, color=S2)
    b.set_xscale('log'); b.set_xticks(q, [f'{x:g}%' for x in q]); b.minorticks_off()
    b.set_ylim(0, 0.8); b.set_yticks([0, .2, .4, .6, .8], ['0%', '20%', '40%', '60%', '80%'])
    b.set_xlabel('calm moments flagged: top share by the forecast'); b.set_title('Calm moments ranked by the forecast (VAL, round 7)', fontsize=12, pad=26)
    save(fig, '28_vol_transitions.png')


if __name__ == '__main__':
    confusion(); transitions()
