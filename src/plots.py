"""Presentation figures from the result files (no numbers typed by hand). usage: python plots.py -> figures/*.png
Static PNGs for slides: identity is carried by direct labels / markers as well as color (palette validated; two slots
are below 3:1 contrast, so every series is direct-labelled)."""
import json, os, glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
import common as C

R = os.path.join(C.ROOT, 'results'); FIG = os.path.join(C.ROOT, 'figures'); os.makedirs(FIG, exist_ok=True)
S1, S2, S3, S4 = '#2a78d6', '#eb6834', '#1baf7a', '#eda100'
INK, INK2, GRID, SURF = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
SEQ = ['#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95', '#0d366b']
plt.rcParams.update({'figure.facecolor': SURF, 'axes.facecolor': SURF, 'axes.edgecolor': GRID, 'axes.labelcolor': INK2,
                     'xtick.color': INK2, 'ytick.color': INK2, 'text.color': INK, 'axes.grid': True, 'grid.color': GRID,
                     'grid.linewidth': 0.8, 'axes.spines.top': False, 'axes.spines.right': False, 'font.size': 11,
                     'axes.titlesize': 13, 'axes.titleweight': 'bold', 'axes.titlelocation': 'left'})
rd = lambda f: [json.loads(l) for l in open(os.path.join(R, f))] if os.path.exists(os.path.join(R, f)) else []


def save(fig, name):
    fig.tight_layout(); fig.savefig(os.path.join(FIG, name), dpi=200); plt.close(fig); print('wrote', name)


def coverage():
    TR, VA, TE = C.split_days(); days = C.all_days()
    n = [len(C.load_day(d)['ts']) for d in days]; cov = [float(C.load_day(d)['cov']) for d in days]
    col = [S1 if d in TR else S2 if d in VA else S3 for d in days]
    fig, ax = plt.subplots(figsize=(12, 3.8))
    x = np.arange(len(days))
    bars = ax.bar(x, np.array(n) / 1000, color=col, width=0.8, edgecolor=SURF, linewidth=1)
    for b, c in zip(bars, cov):
        if c < C.MIN_COV: b.set_hatch('////'); b.set_edgecolor(INK2)
    ax.set_xticks(x[::5]); ax.set_xticklabels([f'{d[4:6]}/{d[6:]}' for d in days[::5]])
    ax.set_ylabel('thousand snapshots / day'); ax.set_title('KRX 000660 continuous-session snapshots per trading day')
    from matplotlib.patches import Patch
    hs = [Patch(facecolor=c, label=f'{lab} ({len(d)} days)') for lab, c, d in (('train', S1, TR), ('val', S2, VA), ('test', S3, TE))]
    hs.append(Patch(facecolor=S1, edgecolor=INK2, hatch='////', label='coverage < 60% (train only)'))
    ax.legend(handles=hs, frameon=False, ncol=4, loc='upper left', bbox_to_anchor=(0, -0.12)); save(fig, '01_data_coverage.png')


def horizon_curve():
    fin = rd('final.jsonl')
    if not fin: return
    Hs = [r['H'] for r in fin]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for key, lab, c, mk in (('gbm', 'Gradient boosting', S1, 'o'), ('logreg', 'Logistic regression', S2, 's'),
                            ('momentum', 'Momentum rule', S3, '^'), ('majority', 'Majority class', S4, 'D')):
        y = [r[key]['mF1'] for r in fin]
        ax.plot(Hs, y, color=c, lw=2, marker=mk, ms=8, mec=SURF, mew=2)
        dy = {'gbm': 7, 'logreg': -7}.get(key, 0)
        ax.annotate(lab, (Hs[-1], y[-1]), xytext=(8, dy), textcoords='offset points', va='center', color=INK2)
    for r in rd('deep_test.jsonl'):
        ax.plot([r['H']], [r['test_mF1']], marker='*', ms=14, color=INK, ls='none')
        ax.annotate({'gru': 'conv+GRU', 'deeplob': 'DeepLOB'}[r['model']], (r['H'], r['test_mF1']), xytext=(6, 6), textcoords='offset points', color=INK2)
    ax.set_xscale('log'); ax.set_xticks(Hs); ax.set_xticklabels([f'{h}s' for h in Hs]); ax.set_xlim(0.8, 120)
    ax.set_xlabel('prediction horizon'); ax.set_ylabel('test macro-F1 (10 days)'); ax.set_ylim(0, None)
    ax.set_title('Test macro-F1 by horizon'); save(fig, '03_horizon_curve.png')


def horizon_val():
    sel = [r for r in rd('select.jsonl') if r['exp'] == 'horizon']
    if not sel: return
    Hs = sorted({r['H'] for r in sel})
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for key, lab, c, mk in (('gbm all', 'Gradient boosting', S1, 'o'), ('logreg all', 'Logistic regression (all)', S2, 's'),
                            ('momentum(sign ret20)', 'Momentum rule', S3, '^'), ('majority', 'Majority class', S4, 'D')):
        y = [next(r['mF1'] for r in sel if r['H'] == h and r['model'] == key) for h in Hs]
        ax.plot(Hs, y, color=c, lw=2, marker=mk, ms=8, mec=SURF, mew=2)
        dy = {'gbm all': 7, 'logreg all': -7}.get(key, 0)
        ax.annotate(lab, (Hs[-1], y[-1]), xytext=(8, dy), textcoords='offset points', va='center', color=INK2)
    ax.set_xscale('log'); ax.set_xticks(Hs); ax.set_xticklabels([f'{h}s' for h in Hs]); ax.set_xlim(0.8, 120)
    ax.set_xlabel('prediction horizon'); ax.set_ylabel('validation macro-F1 (12 days)'); ax.set_ylim(0, None)
    ax.set_title('Validation macro-F1 by horizon'); save(fig, '02_horizon_val.png')


def ablation():
    sel = [r for r in rd('select.jsonl') if r['exp'] == 'ablation']
    if not sel: return
    fig, ax = plt.subplots(figsize=(7, 3.6))
    y = np.arange(len(sel))[::-1]
    ax.hlines(y, [r['mF1'] - r['mF1_day_sd'] for r in sel], [r['mF1'] + r['mF1_day_sd'] for r in sel], color='#9ec5f4', lw=3)
    ax.plot([r['mF1'] for r in sel], y, 'o', color=S1, ms=9, mec=SURF, mew=2)
    for yy, r in zip(y, sel): ax.annotate(f"{r['mF1']:.3f}", (r['mF1'] + r['mF1_day_sd'], yy), xytext=(6, 0), textcoords='offset points', va='center', color=INK2)
    ax.set_yticks(y); ax.set_yticklabels([f"{r['features']} ({r['n_features']})" for r in sel]); ax.set_ylim(-0.6, len(sel) - 0.4)
    ax.set_xlabel('validation macro-F1 (gradient boosting, 5 s); bar = \u00b11 sd across days')
    ax.set_title('Feature groups added one at a time (validation)'); ax.grid(axis='y', visible=False); save(fig, '04_feature_ablation_val.png')


def gbm_grid():
    g = [r for r in rd('select.jsonl') if r['exp'] == 'gbm_grid']
    if not g: return
    g = sorted(g, key=lambda r: r['mF1']); best = g[-1]
    fig, ax = plt.subplots(figsize=(7, 3.8)); y = np.arange(len(g))
    ax.plot([r['mF1'] for r in g], y, 'o', color='#6da7ec', ms=8, mec=SURF, mew=2)
    ax.plot([best['mF1']], [y[-1]], 'o', color=S1, ms=10, mec=SURF, mew=2)
    for yy, r in zip(y, g): ax.annotate(f"{r['mF1']:.4f}", (r['mF1'], yy), xytext=(8, 0), textcoords='offset points', va='center', color=INK2, fontsize=9)
    ax.set_yticks(y); ax.set_yticklabels([f"lr={r['learning_rate']}, leaves={r['max_leaf_nodes']}, l2={r['l2']}" for r in g], fontsize=9)
    sd = float(np.mean([r['mF1_day_sd'] for r in g]))
    ax.set_xlim(g[0]['mF1'] - 0.004, best['mF1'] + 0.006)
    ax.set_xlabel(f'validation macro-F1 (5 s); whole spread {best["mF1"] - g[0]["mF1"]:.4f} vs day-to-day sd \u2248 {sd:.3f}')
    ax.set_title('Gradient boosting hyper-parameters (validation)'); ax.grid(axis='y', visible=False); save(fig, '05_gbm_grid_val.png')


def per_day():
    fin = [r for r in rd('final.jsonl') if r['H'] == 5]
    if not fin: return
    g = fin[0]['gbm']; days = sorted(g['per_day_mF1'])
    fig, ax = plt.subplots(figsize=(9, 3.8)); x = np.arange(len(days))
    ax.plot(x, [g['per_day_mF1'][d] for d in days], color=S1, lw=2, marker='o', ms=8, mec=SURF, mew=2)
    ax.annotate('Gradient boosting', (x[-1], g['per_day_mF1'][days[-1]]), xytext=(8, 0), textcoords='offset points', va='center', color=INK2)
    for r in rd('deep_test.jsonl'):
        if r['H'] != 5: continue
        yy = [r['per_day_mF1'].get(d, np.nan) for d in days]
        c = S2 if r['model'] == 'deeplob' else S3
        ax.plot(x, yy, color=c, lw=2, marker='s' if r['model'] == 'deeplob' else '^', ms=8, mec=SURF, mew=2)
        ax.annotate({'gru': 'conv+GRU', 'deeplob': 'DeepLOB'}[r['model']], (x[-1], yy[-1]), xytext=(8, 0), textcoords='offset points', va='center', color=INK2)
    ax.set_xticks(x); ax.set_xticklabels([f'{d[4:6]}/{d[6:]}' for d in days]); ax.set_xlim(-0.5, len(days) + 1.5)
    ax.set_ylabel('macro-F1'); ax.set_title('Test macro-F1 per day (5 s horizon)'); save(fig, '06_test_per_day.png')


def _sel_curve(prob, y, fr):
    conf = np.maximum(prob[:, 0], prob[:, 2]); call = np.where(prob[:, 2] > prob[:, 0], 2, 0)
    hit, hit_mv = [], []
    for q in fr:
        s = conf >= np.quantile(conf, 1 - q); hit.append((call[s] == y[s]).mean())
        mv = s & (y != 1); hit_mv.append((call[mv] == y[mv]).mean())
    return np.array(hit), np.array(hit_mv)


def coverage_curve():
    fr = np.array([1.0, 0.5, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01])
    srcs = [('Gradient boosting', 'test_pred_gbm_H5.npz', S1, 'o'), ('DeepLOB', 'test_pred_deeplob_H5.npz', S2, 's'),
            ('conv+GRU', 'test_pred_gru_H5.npz', S3, '^')]
    srcs = [s for s in srcs if os.path.exists(os.path.join(R, s[1]))]
    if not srcs: return
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.6), sharex=True)
    for lab, f, c, mk in srcs:
        z = np.load(os.path.join(R, f)); hit, hit_mv = _sel_curve(z['prob'], z['y'], fr)
        for ax, h in zip(axs, (hit, hit_mv)):
            ax.plot(fr * 100, h, color=c, lw=2, marker=mk, ms=8, mec=SURF, mew=2)
            ax.annotate(lab, (fr[-1] * 100, h[-1]), xytext=(6, 0), textcoords='offset points', va='center', color=INK2)
    y = np.load(os.path.join(R, srcs[0][1]))['y']; pm = float((y != 1).mean())
    axs[0].axhline(pm / 2, color=INK2, lw=1, ls='--'); axs[0].annotate(f'random up/down call ({pm / 2:.2f})', (100, pm / 2), xytext=(4, 5), textcoords='offset points', color=INK2, fontsize=9)
    axs[1].axhline(0.5, color=INK2, lw=1, ls='--'); axs[1].annotate('coin flip (0.50)', (100, 0.5), xytext=(4, 5), textcoords='offset points', color=INK2, fontsize=9)
    axs[0].set_ylim(pm / 2 - 0.03, None); axs[1].set_ylim(0.47, None)
    axs[0].set_ylabel('hit rate, all calls (flat outcome = miss)'); axs[1].set_ylabel('hit rate, only when the price moved')
    axs[0].set_title('All up/down calls (test, 5 s)'); axs[1].set_title('Calls where the mid actually moved')
    for ax in axs:
        ax.set_xscale('log'); ax.invert_xaxis(); ax.set_xticks([100, 50, 20, 10, 5, 2, 1]); ax.set_xticklabels(['100%', '50%', '20%', '10%', '5%', '2%', '1%'])
        ax.set_xlabel('share of snapshots acted on (most confident first)'); ax.set_xlim(130, 0.55)
    save(fig, '07_confidence_coverage.png')


def time_of_day():
    p = os.path.join(R, 'test_pred_gbm_H5.npz')
    if not os.path.exists(p): return
    z = np.load(p); pr = z['prob'].argmax(1); y = z['y']; m = z['minute']
    edges = np.append(np.arange(540, 901, 30), 921); lab, val = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        s = (m >= a) & (m < b)
        if s.sum() > 1000: lab.append(f'{int(a)//60:02d}:{int(a)%60:02d}'); val.append(f1_score(y[s], pr[s], average='macro'))
    fig, ax = plt.subplots(figsize=(9, 3.6)); x = np.arange(len(val))
    ax.plot(x, val, color=S1, lw=2, marker='o', ms=8, mec=SURF, mew=2)
    for xx, v in zip(x, val): ax.annotate(f'{v:.2f}', (xx, v), xytext=(0, 9), textcoords='offset points', ha='center', color=INK2, fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(lab); ax.set_ylim(min(val) - 0.04, max(val) + 0.04)
    ax.set_ylabel('macro-F1'); ax.set_xlabel('start of 30-minute window (KST)')
    ax.set_title('Test macro-F1 by time of day (gradient boosting, 5 s)'); save(fig, '08_time_of_day.png')


def confusion():
    fin = [r for r in rd('final.jsonl') if r['H'] == 5]
    if not fin: return
    cm = np.array(fin[0]['gbm']['confusion'], float); cmn = cm / cm.sum(1, keepdims=True)
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list('seq', SEQ)
    fig, ax = plt.subplots(figsize=(4.8, 4.2)); ax.grid(False)
    ax.imshow(cmn, cmap=cmap, vmin=0, vmax=1)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f'{cmn[i, j]:.2f}\n({int(cm[i, j]):,})', ha='center', va='center', fontsize=9,
                    color='white' if cmn[i, j] > 0.5 else INK)
    ax.set_xticks(range(3)); ax.set_yticks(range(3)); L = ['down', 'flat', 'up']
    ax.set_xticklabels(L); ax.set_yticklabels(L); ax.set_xlabel('predicted'); ax.set_ylabel('true')
    ax.set_title('Confusion (row-normalised), GBM 5 s'); save(fig, '09_confusion_gbm_5s.png')


def cross():
    cr = rd('cross.jsonl')
    if not cr: return
    fig, ax = plt.subplots(figsize=(7.5, 4)); w = 0.25; x = np.arange(len(cr))
    for k, (key, lab, c) in enumerate((('hl_only', 'Hyperliquid perp only', S3), ('krx_only', 'KRX book only', S1), ('krx+hl', 'KRX + perp', S2))):
        v = [r[key]['mF1'] for r in cr]
        ax.bar(x + (k - 1) * w, v, w * 0.92, color=c, label=lab, edgecolor=SURF, linewidth=2)
        for xx, vv in zip(x + (k - 1) * w, v): ax.text(xx, vv + 0.005, f'{vv:.3f}', ha='center', fontsize=8, color=INK2)
    ax.set_xticks(x); ax.set_xticklabels([f"{r['H']}s" for r in cr]); ax.set_ylabel('test macro-F1 (08/10-08/28)')
    ax.set_title('Does the 24/7 perp help predict the KRX book?'); ax.legend(frameon=False, ncol=3, loc='upper left', bbox_to_anchor=(0, -0.12))
    ax.grid(axis='x', visible=False); save(fig, '10_cross_venue.png')


def dl_curves():
    fs = sorted(glob.glob(os.path.join(R, 'deep_*_epochs.jsonl')))
    if not fs: return
    sty = {'gru': (S3, '^', 'conv+GRU'), 'deeplob': (S2, 's', 'DeepLOB')}
    fig, ax = plt.subplots(figsize=(7.5, 4))
    for f in fs:
        rr = [json.loads(l) for l in open(f)]; c, mk, lab = sty[rr[0]['model']]
        ax.plot([r['epoch'] + 1 for r in rr], [r['val_mF1'] for r in rr], color=c, lw=2, marker=mk, ms=8, mec=SURF, mew=2)
        ax.annotate(lab, (rr[-1]['epoch'] + 1, rr[-1]['val_mF1']), xytext=(8, 0), textcoords='offset points', va='center', color=INK2)
    g = max((r for r in rd('select.jsonl') if r['exp'] == 'gbm_grid'), key=lambda r: r['mF1'], default=None)
    if g:
        ax.axhline(g['mF1'], color=S1, lw=1.5, ls='--')
        ax.annotate(f"gradient boosting (best grid, {g['mF1']:.3f})", (1, g["mF1"]), xytext=(0, -13), textcoords='offset points', color=INK2, fontsize=9)
    ax.set_xlabel('epoch'); ax.set_ylabel('validation macro-F1'); ax.set_xlim(0.7, None)
    ax.set_title('Deep models: validation macro-F1 per epoch (5 s)'); save(fig, '11_deep_val_curves.png')


def gru_tuning():
    rr = rd('tune_gru.jsonl')
    if not rr: return
    by = {r['name']: r for r in rr}
    seeds = [by[k]['best_val_mF1'] for k in ('base', 'base_seed1', 'base_seed2') if k in by]
    groups = [('look-back T', [('T10', 'T=10'), ('T25', 'T=25'), ('base', 'T=50 (base)'), ('T100', 'T=100')]),
              ('GRU hidden size', [('hidden32', '32'), ('base', '64 (base)'), ('hidden128', '128')]),
              ('learning rate', [('lr3e-4', '3e-4'), ('base', '1e-3 (base)'), ('lr3e-3', '3e-3')]),
              ('dropout', [('dropout0', '0'), ('base', '0.2 (base)'), ('dropout0.5', '0.5')]),
              ('class weights', [('base', 'balanced (base)'), ('no_class_weight', 'none')])]
    rows, labels, ys, seps, y = [], [], [], [], 0
    for gname, items in groups:
        items = [(k, l) for k, l in items if k in by]
        if not items: continue
        for k, l in items:
            rows.append(by[k]); labels.append(f'{gname}: {l}'); ys.append(y); y += 1
        seps.append(y - 0.5); y += 0.6
    fig, ax = plt.subplots(figsize=(8.5, 0.42 * len(rows) + 1.6))
    ys = -np.array(ys)
    if len(seeds) > 1:
        ax.axvspan(min(seeds), max(seeds), color='#cde2fb', zorder=0)
        ax.annotate(f'base, 3 seeds: {min(seeds):.3f}\u2013{max(seeds):.3f}', (min(seeds), 0.6), xytext=(-4, 0), textcoords='offset points', color=INK2, fontsize=9, va='center', ha='right')
    xs = [r['best_val_mF1'] for r in rows]
    ax.plot(xs, ys, 'o', color=S3, ms=8, mec=SURF, mew=2, zorder=3)
    for x_, y_ in zip(xs, ys): ax.annotate(f'{x_:.3f}', (x_, y_), xytext=(8, 0), textcoords='offset points', va='center', color=INK2, fontsize=9)
    ax.set_yticks(ys); ax.set_yticklabels(labels, fontsize=9); ax.grid(axis='y', visible=False)
    ax.set_xlim(min(xs) - 0.005, max(xs) + 0.008); ax.set_ylim(ys.min() - 0.7, 1.2)
    ax.set_xlabel('best validation macro-F1 (conv+GRU, 5 s)')
    ax.set_title('conv+GRU: one setting changed at a time (validation)'); save(fig, '12_gru_tuning_val.png')


def class_weight_tradeoff():
    by = {r['name']: r for r in rd('tune_gru.jsonl')}
    if 'base' not in by or 'no_class_weight' not in by: return
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for k, lab, c, mk in (('base', 'balanced class weights', S3, 'o'), ('no_class_weight', 'no class weights', S2, 's')):
        r = by[k]; ax.plot([r['best_val_acc']], [r['best_val_mF1']], mk, color=c, ms=11, mec=SURF, mew=2)
        ax.annotate(lab, (r['best_val_acc'], r['best_val_mF1']), xytext=(10, 0), textcoords='offset points', va='center', color=INK2)
    m = next((r for r in rd('select.jsonl') if r['exp'] == 'horizon' and r['H'] == 5 and r['model'] == 'majority'), None)
    if m:
        ax.plot([m['acc']], [m['mF1']], 'D', color=S4, ms=10, mec=SURF, mew=2)
        ax.annotate('always "flat"', (m['acc'], m['mF1']), xytext=(10, 0), textcoords='offset points', va='center', color=INK2)
    ax.set_xlabel('validation accuracy'); ax.set_ylabel('validation macro-F1')
    ax.margins(x=0.25, y=0.15); ax.set_title('Class weights trade accuracy for macro-F1'); save(fig, '13_class_weight_tradeoff.png')


def next_move():
    rr = {r['block']: r for r in rd('next_move.jsonl')}
    if 'test' not in rr: return
    r = rr['test']
    items = [('majority', 'Always the more common side', S4), ('imb1_rule', 'Rule: bigger best-level queue wins', S2),
             ('logreg', 'Logistic regression', '#9ec5f4'), ('gbm', 'Gradient boosting', S1)]
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw=dict(width_ratios=[1.2, 1]))
    ax = axs[0]; y = np.arange(len(items))[::-1]
    v = [r[k]['acc'] for k, _, _ in items]
    ax.barh(y, v, color=[c for _, _, c in items], height=0.6, edgecolor=SURF, linewidth=2)
    for yy, x in zip(y, v): ax.text(x + 0.01, yy, f'{x:.3f}', va='center', color=INK2)
    ax.set_yticks(y); ax.set_yticklabels([l for _, l, _ in items]); ax.set_xlim(0, 0.8); ax.grid(axis='y', visible=False)
    ax.set_xlabel('test accuracy (up vs down)'); ax.set_title('Direction of the next mid-price change (test)')
    ax = axs[1]; q = list(r['gbm_top_conf_acc'].items())
    xs = [float(k[:-1]) for k, _ in q]; ys = [a for _, a in q]
    ax.plot(xs, ys, color=S1, lw=2, marker='o', ms=8, mec=SURF, mew=2)
    for x_, y_ in zip(xs, ys): ax.annotate(f'{y_:.2f}', (x_, y_), xytext=(0, 9), textcoords='offset points', ha='center', color=INK2, fontsize=9)
    ax.axhline(0.5, color=INK2, lw=1, ls='--'); ax.annotate('coin flip', (100, 0.5), xytext=(4, 5), textcoords='offset points', color=INK2, fontsize=9)
    ax.set_xscale('log'); ax.invert_xaxis(); ax.set_xticks(xs); ax.set_xticklabels([f'{int(x)}%' for x in xs]); ax.set_ylim(0.45, 0.95)
    ax.set_xlabel('share of snapshots acted on (most confident first)'); ax.set_ylabel('accuracy'); ax.set_title('Gradient boosting by confidence')
    save(fig, '14_next_move_test.png')


def norm_wf():
    v = [r for r in rd('norm_wf.jsonl') if r['block'] == 'val']
    if not v: return
    lab = {'fixed': 'fixed (train once)', 'wf_expanding': 'refit every 5 days, all past', 'wf_last20': 'refit every 5 days, last 20'}
    fig, ax = plt.subplots(figsize=(8, 3.6)); ys = np.arange(3)[::-1]
    for norm, c, mk, nm in ((False, S1, 'o', 'raw features'), (True, S2, 's', 'price-level normalised')):
        xs = [next(r['mF1'] for r in v if r['scheme'] == s and r['norm'] == norm) for s in lab]
        ax.plot(xs, ys, mk, color=c, ms=10, mec=SURF, mew=2, label=nm, ls='none')
        for x_, y_ in zip(xs, ys): ax.annotate(f'{x_:.3f}', (x_, y_), xytext=(0, 10), textcoords='offset points', ha='center', color=INK2, fontsize=9)
    ax.set_yticks(ys); ax.set_yticklabels(list(lab.values())); ax.set_ylim(-0.6, 2.7); ax.grid(axis='y', visible=False)
    ax.legend(frameon=False, loc='upper left', bbox_to_anchor=(0, -0.22), ncol=2); ax.set_xlabel('validation macro-F1 (gradient boosting, 5 s)'); ax.set_xlim(0.37, 0.52)
    ax.set_title('Normalisation and rolling refits did not help (validation)'); save(fig, '15_norm_rolling_val.png')


def trade_groups():
    b = [r for r in rd('trades.jsonl') if r.get('stage') == 'B' and r['name'].startswith('drop_')]
    if not b: return
    lab = {'book': 'order book (47)', 'flow': 'book order flow (15)', 'trade_tw': 'trades: 1/5/30 s volume, count (12)',
           'time': 'time of day (3)', 'price': 'past price (6)', 'trade_ev': 'trades: last 5/20/100 imbalance (3)',
           'trade_last': 'trades: last trade info (5)', 'trade_cum': 'trades: cumulative / strength (4)'}
    b = sorted(b, key=lambda r: r['contribution']); ys = np.arange(len(b))
    fig, ax = plt.subplots(figsize=(9, 4.2))
    for y_, r in zip(ys, b):
        keep = r['contribution'] > 0.001; c = S1 if keep else S4
        ax.plot([0, r['contribution']], [y_, y_], color=GRID, lw=2, zorder=1)
        ax.plot(r['contribution'], y_, 'o' if keep else 'X', color=c, ms=10, mec=SURF, mew=2, zorder=2)
        ax.annotate(f"{r['contribution']:+.4f}  {'kept' if keep else 'dropped'}", (r['contribution'], y_), xytext=(10, -4),
                    textcoords='offset points', color=INK2, fontsize=9, bbox=dict(fc=SURF, ec='none', pad=1))
    ax.axvline(0.001, color=INK2, lw=1, ls='--'); ax.annotate('rule: drop if <= 0.001', (0.001, len(b) - 0.6), xytext=(4, 0),
                                                          textcoords='offset points', color=INK2, fontsize=9)
    ax.set_xscale('symlog', linthresh=0.002); ax.set_xlim(-0.001, 0.08)
    ax.set_yticks(ys); ax.set_yticklabels([lab[r['group']] for r in b]); ax.grid(axis='y', visible=False)
    ax.set_xlabel('validation macro-F1 lost when the group is removed (5 s)')
    ax.set_title('Which feature groups matter (validation, gradient boosting)'); save(fig, '16_trade_groups_val.png')


def trades_test():
    f = {r['H']: r for r in rd('final.jsonl')}; t = {r['H']: r for r in rd('trades.jsonl') if r.get('stage') == 'final' and 'H' in r}
    if not t: return
    Hs = sorted(t); fig, axs = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw=dict(width_ratios=[1, 1.1]))
    ax = axs[0]; x = np.arange(len(Hs))
    for vals, c, mk, nm, dx, oy in (([f[h]['gbm']['mF1'] for h in Hs], S1, 'o', 'order book only', -0.12, 9),
                                ([t[h]['mF1'] for h in Hs], S2, 's', 'order book + trades', 0.12, -15)):
        ax.plot(x + dx, vals, mk, color=c, ms=10, mec=SURF, mew=2, ls='none', label=nm)
        for x_, v in zip(x + dx, vals): ax.annotate(f'{v:.3f}', (x_, v), xytext=(0, oy), textcoords='offset points', ha='center', color=INK2, fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels([f'{h} s' for h in Hs]); ax.set_xlabel('horizon'); ax.set_ylabel('test macro-F1')
    ax.set_ylim(0.40, 0.52); ax.legend(frameon=False, loc='lower left'); ax.set_title('Test: trades did not add accuracy')
    a = np.load(os.path.join(R, 'test_pred_gbm_H5.npz')); b = np.load(os.path.join(R, 'test_pred_trades_gbm_H5.npz'))
    ds = np.unique(a['day']); d = np.array([f1_score(b['y'][a['day'] == k], b['prob'][a['day'] == k].argmax(1), average='macro') -
                                            f1_score(a['y'][a['day'] == k], a['prob'][a['day'] == k].argmax(1), average='macro') for k in ds])
    ax = axs[1]; xx = np.arange(len(ds))
    ax.bar(xx, d, color=[S2 if v > 0 else S1 for v in d], width=0.6, edgecolor=SURF, linewidth=2)
    ax.axhline(0, color=INK2, lw=1); ax.axhline(d.mean(), color=INK2, lw=1, ls='--')
    ax.annotate(f'mean {d.mean():+.4f}', (-0.5, d.mean()), xytext=(2, 4), textcoords='offset points', ha='left', color=INK2, fontsize=9)
    ax.set_xticks(xx); ax.set_xticklabels([str(k)[4:6] + '-' + str(k)[6:] for k in ds], rotation=45); ax.grid(axis='x', visible=False)
    ax.set_ylabel('macro-F1: with trades - book only'); ax.set_title(f'5 s, per test day (trades better on {int((d > 0).sum())} of {len(d)})')
    save(fig, '17_trades_test.png')


if __name__ == '__main__':
    for fn in (coverage, horizon_val, horizon_curve, ablation, gbm_grid, per_day, coverage_curve, time_of_day, confusion, cross, dl_curves, gru_tuning, class_weight_tradeoff, next_move, norm_wf, trade_groups, trades_test):
        fn()
