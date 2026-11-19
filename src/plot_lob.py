"""Figures for LOB_MODELS.md (MLPLOB / TLOB vs conv+GRU, 5 s direction), from the result files only.
usage: python plot_lob.py -> figures/20_lob_val.png, 21_lob_val_test.png, 22_lob_cost.png, 23_lob_val_curves.png
"""
import json, os, sys
import numpy as np, matplotlib
matplotlib.use('Agg')
sys.argv = ['plots.py']
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plots.py')).read().split('\ndef ')[0])  # style header only
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def save(fig, name):
    fig.tight_layout(); fig.savefig(os.path.join(FIG, name), dpi=200); plt.close(fig); print('wrote', name)


STY = {'convgru': (S3, '^', 'conv+GRU'), 'mlplob': (S1, 's', 'MLPLOB'), 'tlob': (S2, 'o', 'TLOB')}
LOB = rd('lob_models.jsonl'); S = {r['name']: r for r in LOB if r.get('stage') == 'S'}
BASE = [r for r in rd('v2_deep.jsonl') if r.get('name') == 'base' and r.get('seed') is not None]
RAW = {r['stage']: r for r in rd('raw_cmp.jsonl')}
SETS = [('raw50', '40 raw book columns, 50 snapshots'), ('full50', '127 features, 50 snapshots'), ('full128', '127 features, 128 snapshots')]


def legend(ax, archs, loc='lower right'):
    ax.legend([Line2D([], [], color=STY[a][0], marker=STY[a][1], ls='', ms=8) for a in archs], [STY[a][2] for a in archs], loc=loc, frameon=False, fontsize=9)


def val_by_input():
    fig, ax = plt.subplots(figsize=(8.5, 4.2)); yt, yl = [], []
    for i, (key, label) in enumerate(SETS):
        y0 = -i * 1.0; yt.append(y0); yl.append(label)
        for j, a in enumerate(('convgru', 'mlplob', 'tlob')):
            y = y0 + (1 - j) * 0.22; c, mk, _ = STY[a]
            if a == 'convgru':
                if key == 'raw50': v, lo, hi = RAW['val']['gru_raw']['mF1'], None, None
                elif key == 'full50':
                    s = [r['val_mF1'] for r in BASE]; v, lo, hi = float(np.mean(s)), min(s), max(s)
                else: continue
                if lo is not None: ax.plot([lo, hi], [y, y], color=c, lw=2, alpha=0.5, solid_capstyle='round')
                txt = f'{v:.4f}' + (f' ({len(BASE)}-seed mean)' if lo is not None else '')
            else:
                v = S[f'{a}_{key}']['val_mF1']; txt = f'{v:.4f}'; lo = None
            ax.plot(v, y, mk, color=c, ms=9, mec=SURF, mew=2, zorder=3)
            ax.annotate(txt, (hi if a == 'convgru' and lo is not None else v, y), xytext=(9, 0), textcoords='offset points', va='center', color=INK2, fontsize=9)
    ax.set_yticks(yt); ax.set_yticklabels(yl); ax.grid(axis='y', visible=False); ax.set_ylim(-2.5, 0.5); ax.set_xlim(0.494, 0.5195)
    ax.set_xlabel('validation macro-F1 (seed 0; conv+GRU on 127 features: 5 seeds, line = range)')
    ax.set_title('5 s direction: input matters more than architecture (validation)'); legend(ax, ('convgru', 'mlplob', 'tlob'), 'lower left')
    save(fig, '20_lob_val.png')


def val_test():
    bv = [r for r in LOB if r.get('stage') == 'bagval'][-1]; te = [r for r in LOB if r.get('stage') == 'test'][-1]
    cg_val_seeds = [r['val_mF1'] for r in BASE if r['seed'] in te['seeds']]
    panels = [('Validation (12 days)', dict(convgru=(cg_val_seeds, bv['convgru_bag_mF1']), tlob=(bv['seed_mF1'], bv['bag_mF1'])), bv['both_mF1']),
              ('Test (10 days, scored once)', dict(convgru=(te['convgru_seed_mF1'], te['convgru_bag_mF1']), tlob=(te['seed_mF1'], te['mF1'])), te['both_mF1'])]
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.6), sharex=True)
    for ax, (title, d, both) in zip(axs, panels):
        rows = [('convgru', 'conv+GRU'), ('tlob', 'TLOB, 128 snapshots')]
        for k, (a, lab) in enumerate(rows):
            y = -k; c, mk, _ = STY[a]; seeds, bag = d[a]
            ax.plot(seeds, [y] * len(seeds), mk, color=c, ms=7, mfc=SURF, mec=c, mew=1.5, zorder=3)
            ax.plot(bag, y, 'D', color=c, ms=10, mec=SURF, mew=2, zorder=4)
            ax.annotate(f'{bag:.4f}', (bag, y), xytext=(0, 11), textcoords='offset points', ha='center', color=INK2, fontsize=9)
        ax.plot(both, -2, 'D', color=INK2, ms=9, mec=SURF, mew=2, zorder=4)
        ax.annotate(f'{both:.4f}', (both, -2), xytext=(0, 11), textcoords='offset points', ha='center', color=INK2, fontsize=9)
        ax.set_yticks([0, -1, -2]); ax.set_yticklabels([r[1] for r in rows] + ['both, averaged']); ax.set_ylim(-2.6, 0.6)
        ax.grid(axis='y', visible=False); ax.set_title(title, fontsize=11); ax.set_xlabel('macro-F1')
    axs[1].annotate(f"TLOB - conv+GRU: {te['boot_vs_convgru']['mean']:+.4f}, 95% CI [{te['boot_vs_convgru']['ci95'][0]:+.4f}, {te['boot_vs_convgru']['ci95'][1]:+.4f}]",
                    (0.02, 0.02), xycoords='axes fraction', fontsize=8.5, color=INK2)
    axs[0].legend([Line2D([], [], color=INK2, marker='o', mfc=SURF, ls='', ms=7), Line2D([], [], color=INK2, marker='D', ls='', ms=8)],
                  ['single seed (0, 1, 2)', 'mean of the 3 seeds\' probabilities'], loc='lower left', frameon=False, fontsize=8.5)
    fig.suptitle('TLOB led on validation but fell behind on test', x=0.01, ha='left', fontsize=13, fontweight='bold')
    save(fig, '21_lob_val_test.png')


def cost():
    rr = {r['name']: r for r in rd('lob_cost.jsonl')}
    if not rr: return
    pts = [('convgru_full50', 'convgru', 'conv+GRU, 50', float(np.mean([r['val_mF1'] for r in BASE]))),
           ('mlplob_full50', 'mlplob', 'MLPLOB, 50', S['mlplob_full50']['val_mF1']), ('mlplob_full128', 'mlplob', 'MLPLOB, 128', S['mlplob_full128']['val_mF1']),
           ('tlob_full50', 'tlob', 'TLOB, 50', S['tlob_full50']['val_mF1']), ('tlob_full128', 'tlob', 'TLOB, 128', S['tlob_full128']['val_mF1'])]
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, key, xl in ((axs[0], 'macs', 'multiply-adds per window (log scale)'), (axs[1], 'train_min', 'training time of one run, minutes (log scale)')):
        for n, a, lab, v in pts:
            c, mk, _ = STY[a]; x = rr[n][key]
            ax.plot(x, v, mk, color=c, ms=9, mec=SURF, mew=2, zorder=3)
            ax.annotate(lab, (x, v), xytext=(9, 0) if a == 'convgru' else (0, 10), textcoords='offset points', ha='left' if a == 'convgru' else 'center',
                        va='center' if a == 'convgru' else 'bottom', color=INK2, fontsize=8.5)
        ax.set_xscale('log'); ax.set_xlabel(xl)
    axs[0].set_ylabel('validation macro-F1\n(seed 0; conv+GRU: 5-seed mean)'); axs[0].set_ylim(0.5085, 0.5160)
    legend(axs[1], ('convgru', 'mlplob', 'tlob'), 'upper left')
    fig.suptitle('Cost vs validation macro-F1 (127 features; 2 CPU threads)', x=0.01, ha='left', fontsize=13, fontweight='bold')
    save(fig, '22_lob_cost.png')


def curves():
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
    b0 = [r for r in BASE if r['seed'] == 0][-1]
    cg = {'raw50': [(c['epoch'], c['val_mF1']) for c in RAW['gru_curve']['curve']],
          'full50': [(c['epoch'], max(c.get('raw', -1), c.get('ema', -1))) for c in b0['curve']]}
    for ax, (key, label) in zip(axs, SETS):
        lines = [('convgru', cg.get(key))] + [(a, [(c['epoch'], max(c.get('raw', -1), c.get('ema', -1))) for c in S[f'{a}_{key}']['curve']]) for a in ('mlplob', 'tlob')]
        for a, pts in lines:
            if not pts: continue
            c, mk, lab = STY[a]; x = [p[0] + 1 for p in pts]; y = [p[1] for p in pts]
            ax.plot(x, y, color=c, lw=2, marker=mk, ms=6, mec=SURF, mew=1.5)
        ax.set_title(label, fontsize=10.5); ax.set_xlabel('epoch')
    axs[0].set_ylabel('validation macro-F1 (every 10th window)'); legend(axs[2], ('convgru', 'mlplob', 'tlob'))
    fig.suptitle('Validation macro-F1 per epoch, seed 0 (better of raw / EMA weights)', x=0.01, ha='left', fontsize=13, fontweight='bold')
    save(fig, '23_lob_val_curves.png')


if __name__ == '__main__':
    for fn in (val_by_input, val_test, cost, curves): fn()
