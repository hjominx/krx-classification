"""Figures for LOB_MODELS.md section 9 (CPU run vs Apple GPU run of the MLPLOB / TLOB comparison), from the result files only.
usage: python plot_cpu_gpu.py -> figures/24_cpu_gpu_val_test.png, 25_cpu_gpu_curves.png, 26_cpu_gpu_speed.png
"""
import _paths  # noqa: F401  (src/common, src/direction, src/volatility, src/report on the import path)
import json, os, sys
import numpy as np, matplotlib
matplotlib.use('Agg')
sys.argv = ['plots.py']
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plots.py')).read().split('\ndef ')[0])  # style header only
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator


def save(fig, name):
    fig.tight_layout(); fig.savefig(os.path.join(FIG, name), dpi=200); plt.close(fig); print('wrote', name)


STY = {'convgru': (S3, '^', 'conv+GRU'), 'mlplob': (S1, 's', 'MLPLOB'), 'tlob': (S2, 'o', 'TLOB')}
CG = rd('cpu_gpu.jsonl'); st = lambda s: [r for r in CG if r['stage'] == s]
GPU = [json.loads(l) for l in open(os.path.join(C.ROOT, 'results_gpu', 'direction', 'lob_models.jsonl'))]
CPU = rd('lob_models.jsonl')
INP = {'raw': 'raw 40', 'full': '127 feat.'}
best_of = lambda c: max(c.get('raw', -1), c.get('ema', -1))


def pair(ax, y, a, b, color, mk, label_diff=True):
    ax.plot([a, b], [y, y], color=color, lw=1.5, alpha=0.6, zorder=2)
    ax.plot(a, y, mk, color=color, ms=8, mfc=SURF, mec=color, mew=1.6, zorder=3)
    ax.plot(b, y, mk, color=color, ms=8, mec=SURF, mew=1.5, zorder=4)
    if label_diff: ax.annotate(f'{b - a:+.4f}', (max(a, b), y), xytext=(9, 0), textcoords='offset points', va='center', color=INK2, fontsize=8.5)


def val_test():
    fig, axs = plt.subplots(1, 2, figsize=(12, 5.2), gridspec_kw=dict(width_ratios=[1.15, 1]))
    ax = axs[0]; labels = []
    rows = [(r['arch'], f"{STY[r['arch']][2]}, {INP[r['cols']]}, {r['T']}, seed {r['seed']}", r['cpu_val_mF1'], r['gpu_val_mF1']) for r in st('records:run')]
    rows += [('convgru', f"conv+GRU, 127 feat., 50, seed {r['seed']} (CPU in both)", r['cpu_run_val_mF1'], r['gpu_run_val_mF1']) for r in st('records:convgru')]
    for i, (a, lab, c, g) in enumerate(rows):
        pair(ax, -i, c, g, STY[a][0], STY[a][1]); labels.append(lab)
    ax.set_yticks(-np.arange(len(rows))); ax.set_yticklabels(labels, fontsize=8.5); ax.grid(axis='y', visible=False)
    ax.set_xlim(0.494, 0.5185); ax.set_xlabel('validation macro-F1 (each run on its own VAL rows)'); ax.set_title('Every training run', fontsize=11)
    ax.legend([Line2D([], [], color=INK2, marker='o', mfc=SURF, ls='', ms=8), Line2D([], [], color=INK2, marker='o', ls='', ms=8)],
              ['CPU run (2 threads)', 'GPU run (Apple MPS)'], loc='lower left', frameon=False, fontsize=8.5)
    ax = axs[1]; bv, te = st('records:bagval')[-1], st('records:test')[-1]
    rows = [('tlob', 'VAL: TLOB 128, 3 seeds', bv['cpu_bag_mF1'], bv['gpu_bag_mF1']), ('convgru', 'VAL: conv+GRU, 3 seeds', bv['cpu_convgru_bag_mF1'], bv['gpu_convgru_bag_mF1']),
            ('tlob', 'TEST: TLOB 128, 3 seeds', te['cpu_mF1'], te['gpu_mF1']), ('convgru', 'TEST: conv+GRU, 3 seeds', te['cpu_convgru_bag_mF1'], te['gpu_convgru_bag_mF1'])]
    for i, (a, lab, c, g) in enumerate(rows): pair(ax, -i - (i >= 2) * 0.6, c, g, STY[a][0], STY[a][1])
    ax.set_yticks([0, -1, -2.6, -3.6]); ax.set_yticklabels([r[1] for r in rows], fontsize=9); ax.grid(axis='y', visible=False); ax.set_ylim(-4.6, 0.6)
    ax.set_xlim(0.5070, 0.5215); ax.xaxis.set_major_locator(MultipleLocator(0.005))
    ax.set_xlabel('macro-F1 (same rows in both runs)'); ax.set_title('Means of 3 seeds', fontsize=11)
    bc, bg = te['cpu_boot_vs_convgru'], te['gpu_boot_vs_convgru']
    ax.annotate(f"TEST, TLOB - conv+GRU (day-block bootstrap 95% CI)\nCPU run {bc['mean']:+.4f} [{bc['ci95'][0]:+.4f}, {bc['ci95'][1]:+.4f}]\n"
                f"GPU run {bg['mean']:+.4f} [{bg['ci95'][0]:+.4f}, {bg['ci95'][1]:+.4f}]", (0.02, 0.02), xycoords='axes fraction', fontsize=8.5, color=INK2)
    fig.suptitle('CPU run vs Apple GPU run: differences are about the size of seed-to-seed differences', x=0.01, ha='left', fontsize=13, fontweight='bold')
    save(fig, '24_cpu_gpu_val_test.png')


def curves():
    cs = {r['name']: r for r in CPU if r.get('stage') == 'S'}; gs = {r['name']: r for r in GPU if r.get('stage') == 'S'}
    names = ['mlplob_raw50', 'tlob_raw50', 'mlplob_full50', 'tlob_full50', 'mlplob_full128', 'tlob_full128']
    fig, axs = plt.subplots(2, 3, figsize=(12, 6.2), sharey=True)
    for ax, n in zip(axs.T.flatten(), names):
        a = cs[n]['arch']; c, mk, lab = STY[a]
        for r, ls, mfc, who in ((cs[n], '-', SURF, 'CPU'), (gs[n], '--', c, 'GPU')):
            x = [e['epoch'] + 1 for e in r['curve']]; y = [best_of(e) for e in r['curve']]
            ax.plot(x, y, ls, color=c, lw=1.8, marker=mk, ms=5, mfc=mfc, mec=c, mew=1.2)
            ax.plot(r['best_epoch'] + 1, r['val10_mF1'], '*', color=INK2 if who == 'CPU' else c, ms=11, zorder=5, mec=SURF)
        ax.set_title(f"{lab}, {INP[cs[n]['cols']]}, {cs[n]['T']} snapshots", fontsize=10); ax.set_xticks(range(1, 13, 2))
    for ax in axs[1]: ax.set_xlabel('epoch')
    for ax in axs[:, 0]: ax.set_ylabel('VAL macro-F1\n(every 10th window)')
    axs[0, 0].legend([Line2D([], [], color=INK2, ls='-', marker='o', mfc=SURF), Line2D([], [], color=INK2, ls='--', marker='o'), Line2D([], [], color=INK2, marker='*', ls='', ms=10)],
                     ['CPU run', 'GPU run', 'kept checkpoint'], loc='lower right', frameon=False, fontsize=8.5)
    fig.suptitle('Validation macro-F1 per epoch, seed 0: same start, different paths (dropout random numbers differ by device)', x=0.01, ha='left', fontsize=12.5, fontweight='bold')
    save(fig, '25_cpu_gpu_curves.png')


def speed():
    rr = st('speed')
    if not rr: return
    lab = lambda r: {'convgru': 'conv+GRU', 'mlplob': 'MLPLOB', 'tlob': 'TLOB'}[r['name'].split('_')[0]] + f", {r['T']}"
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for ax, (kc, kg, ks, title) in zip(axs, (('cpu_train_ms', 'mps_train_ms', 'train_speedup', 'one training step, batch 256'),
                                             ('cpu_fwd256_ms', 'mps_fwd256_ms', 'fwd_speedup', 'prediction of 256 windows'))):
        for i, r in enumerate(rr):
            a = r['name'].split('_')[0]; c, mk, _ = STY[a]; y = -i
            ax.barh(y + 0.18, r[kc], height=0.34, color=c, alpha=0.35, edgecolor=c)
            ax.barh(y - 0.18, r[kg], height=0.34, color=c, edgecolor=SURF)
            ax.annotate(f'x{r[ks]:.1f}', (r[kc], y + 0.18), xytext=(6, 0), textcoords='offset points', va='center', fontsize=8.5, color=INK2)
        ax.set_xscale('log'); ax.set_xlabel('milliseconds (log scale)'); ax.set_title(title, fontsize=11); ax.grid(axis='y', visible=False)
    axs[0].set_yticks(-np.arange(len(rr))); axs[0].set_yticklabels([lab(r) for r in rr]); axs[1].set_xlim(right=2000)   # room for the legend
    axs[1].legend([plt.Rectangle((0, 0), 1, 1, color=INK2, alpha=0.35), plt.Rectangle((0, 0), 1, 1, color=INK2)], [f"CPU, {rr[0]['cpu_threads']} threads", 'Apple GPU (MPS)'],
                  loc='upper right', frameon=False, fontsize=8.5)
    fig.suptitle('Same machine (Apple M1 Max), 127 features: GPU speed-up (x = CPU time / GPU time)', x=0.01, ha='left', fontsize=13, fontweight='bold')
    save(fig, '26_cpu_gpu_speed.png')


if __name__ == '__main__':
    for fn in (val_test, curves, speed): fn()
