import os
import sys, numpy as np, matplotlib; matplotlib.use('Agg')
sys.argv = ['plots.py']
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plots.py')).read().split('\ndef ')[0])  # style header only
import matplotlib.pyplot as plt
z = np.load(os.path.join(C.ROOT, 'results', 'test_pred_vol_gru.npz')); y, p = z['y'], z['prob'].argmax(1)
cm = np.array([[np.mean(p[y == i] == j) for j in range(3)] for i in range(3)])
fig, ax = plt.subplots(figsize=(5.6, 4.6))
from matplotlib.colors import LinearSegmentedColormap
ax.imshow(cm, cmap=LinearSegmentedColormap.from_list('b', ['#f3f7fd', S1]), vmin=0, vmax=1)
lab = ['calm', 'normal', 'turbulent']
for i in range(3):
    for j in range(3):
        ax.text(j, i, f'{cm[i, j]:.1%}', ha='center', va='center', fontsize=13, color='white' if cm[i, j] > 0.5 else INK, weight='bold' if i == j else 'normal')
ax.set_xticks(range(3), lab); ax.set_yticks(range(3), lab); ax.set_xlabel('predicted'); ax.set_ylabel('actual (next 60 s)')
ax.set_title('conv+GRU on TEST: share of each actual class', fontsize=11, color=INK, loc='left'); ax.grid(False)
for s in ax.spines.values(): s.set_visible(False)
fig.tight_layout(); fig.savefig(os.path.join(C.ROOT, 'figures', '18_vol_confusion.png'), dpi=200); print(cm.round(3))
