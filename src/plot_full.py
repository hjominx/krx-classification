import os
import sys, json, numpy as np, matplotlib; matplotlib.use('Agg')
sys.argv = ['plots.py']; exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plots.py')).read().split('\ndef ')[0])
import matplotlib.pyplot as plt
R = [json.loads(l) for l in open(os.path.join(C.ROOT, 'results', 'gru_full.jsonl')) if '"config"' in l]
fig, axs = plt.subplots(1, 2, figsize=(10.5, 3.9))
for r, c, mk in zip(R, (S1, S2), ('o', 's')):
    ep = [x['epoch'] + 1 for x in r['curve']]; lab = 'hidden 64' if 'h64' in r['name'] else 'hidden 128'
    axs[0].plot(ep, [x['train_loss'] for x in r['curve']], color=c, marker=mk, lw=2, ms=7, label=lab)
    axs[1].plot(ep, [x['val_mF1'] for x in r['curve']], color=c, marker=mk, lw=2, ms=7, label=lab)
    axs[1].annotate(f"{r['curve'][-1]['val_mF1']:.3f}", (ep[-1], r['curve'][-1]['val_mF1']), xytext=(8, 0), textcoords='offset points', va='center', fontsize=10, color=INK2)
axs[0].set_title('Training loss keeps falling'); axs[1].set_title('Validation macro-F1 falls after epoch 1')
for a in axs: a.set_xlabel('epoch (each = all 4.04 M training snapshots)'); a.set_xticks(range(1, 6)); a.legend(frameon=False)
axs[0].set_ylabel('train loss (weighted cross-entropy)'); axs[1].set_ylabel('validation macro-F1 (5 s)'); axs[1].set_xlim(0.7, 5.6)
fig.tight_layout(); fig.savefig(os.path.join(C.ROOT, 'figures', '19_full_training.png'), dpi=200)
